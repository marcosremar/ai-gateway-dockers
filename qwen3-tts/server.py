"""Qwen3-TTS Service — FastAPI wrapper pro ai-gateway.

OpenAI-compatible endpoints so the gateway pipeline can swap qwen3-tts in
for kokoro-tts or Modal qwen3-tts without caller changes:

  GET  /health                — readiness probe.
  GET  /v1/audio/voices       — list built-in speaker presets.
  POST /v1/audio/speech       — JSON body {input, voice, response_format}.
  POST /v1/audio/speech/clone — multipart {text, reference_audio, ref_text}.
                                Voice-cloned synthesis using the 24kHz mono
                                reference clip; ref_text is the transcript
                                the model conditions the prosody encoder on.

Env:
  IDLE_TIMEOUT_MIN     — auto-shutdown after no requests (default 15).
  QWEN3_TTS_MODEL      — HF repo id (default Qwen/Qwen3-TTS).
  PORT                 — listen port (default 8000).
"""

import asyncio
import io
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
import uvicorn

from idle_watchdog import add_idle_middleware, start_watchdog, touch_activity

MODEL_REPO = os.environ.get("QWEN3_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
SAMPLE_RATE = 24_000

inference_lock = asyncio.Lock()
boot_ts = time.time()
model = None
load_error: Optional[str] = None
load_traceback: Optional[str] = None

# Built-in voices — actual list comes from the model. We expose a static
# fallback for the listing endpoint when the model fails to load so the
# gateway autoscaler can still call /health and trigger a redeploy.
DEFAULT_VOICES = [
    {"id": "Ryan",     "language": "en", "gender": "male"},
    {"id": "Vivian",   "language": "en", "gender": "female"},
    {"id": "Cherry",   "language": "en", "gender": "female"},
    {"id": "Ethan",    "language": "en", "gender": "male"},
    {"id": "Serena",   "language": "zh", "gender": "female"},
    {"id": "Sunny",    "language": "zh", "gender": "female"},
]


def _load_model() -> None:
    global model, load_error, load_traceback
    try:
        from qwen_tts import Qwen3TTSModel  # type: ignore
        # qwen-tts handles dtype + device internally; pass `device` when the
        # constructor accepts it, otherwise fall back to whatever ships in
        # this version of the wheel and skip the .to() / .eval() calls that
        # only exist on `nn.Module` subclasses.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = Qwen3TTSModel.from_pretrained(
                MODEL_REPO, torch_dtype=torch.float16, device=device,
            )
        except TypeError:
            model = Qwen3TTSModel.from_pretrained(MODEL_REPO, torch_dtype=torch.float16)
        for fn in ("to", "eval"):
            method = getattr(model, fn, None)
            if not callable(method):
                continue
            try:
                method("cuda") if fn == "to" else method()
            except Exception:
                pass
    except Exception as exc:
        import traceback
        load_error = str(exc)
        load_traceback = traceback.format_exc()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.get_event_loop().run_in_executor(None, _load_model)
    asyncio.create_task(start_watchdog())
    yield


app = FastAPI(title="Qwen3-TTS (ai-gateway)", lifespan=lifespan)
add_idle_middleware(app)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "healthy" if load_error is None else "degraded",
        "model_loaded": model is not None,
        "model": MODEL_REPO,
        "uptime_s": round(time.time() - boot_ts, 1),
        "gpu_available": torch.cuda.is_available(),
        "load_error": load_error,
    }


@app.get("/v1/audio/voices")
async def list_voices() -> dict:
    return {"object": "list", "data": DEFAULT_VOICES}


class SpeechRequest(BaseModel):
    model: Optional[str] = None
    input: str
    voice: str = "Ryan"
    response_format: str = "wav"
    speed: float = 1.0


def _encode(audio: np.ndarray, fmt: str) -> tuple[bytes, str]:
    buf = io.BytesIO()
    if fmt in ("wav", "pcm"):
        sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return buf.getvalue(), "audio/wav"
    if fmt == "flac":
        sf.write(buf, audio, SAMPLE_RATE, format="FLAC")
        return buf.getvalue(), "audio/flac"
    if fmt == "mp3":
        sf.write(buf, audio, SAMPLE_RATE, format="MP3")
        return buf.getvalue(), "audio/mpeg"
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue(), "audio/wav"


def _synth(text: str, voice: str, speed: float = 1.0,
           reference_audio: Optional[np.ndarray] = None,
           reference_sr: Optional[int] = None,
           ref_text: Optional[str] = None) -> np.ndarray:
    if model is None:
        raise RuntimeError(load_error or "model not loaded")

    kwargs = {"speaker": voice, "speed": speed}
    if reference_audio is not None:
        kwargs["reference_audio"] = reference_audio
        if reference_sr:
            kwargs["reference_sample_rate"] = reference_sr
        if ref_text:
            kwargs["reference_text"] = ref_text

    with torch.inference_mode():
        out = model.generate(text=text, **kwargs)

    audio = out["audio"] if isinstance(out, dict) else out
    if hasattr(audio, "cpu"):
        audio = audio.cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest) -> Response:
    touch_activity()
    if not req.input.strip():
        raise HTTPException(400, "input must be non-empty")
    async with inference_lock:
        try:
            audio = await asyncio.get_event_loop().run_in_executor(
                None, _synth, req.input, req.voice, req.speed,
            )
        except Exception as exc:
            raise HTTPException(500, f"synthesis failed: {exc}")
    body, ctype = _encode(audio, req.response_format)
    return Response(content=body, media_type=ctype)


@app.post("/v1/audio/speech/clone")
async def speech_clone(
    text: str = Form(...),
    reference_audio: UploadFile = File(...),
    ref_text: str = Form(""),
    response_format: str = Form("wav"),
    speed: float = Form(1.0),
) -> Response:
    """Voice-cloned synthesis. `reference_audio` is a 5-30s clip (any sample
    rate; we resample to 24 kHz mono internally). `ref_text` is the literal
    transcript of that clip — required by Qwen3-TTS-Base for prosody
    conditioning. Returns the synthesized audio in the requested format."""
    touch_activity()
    if not text.strip():
        raise HTTPException(400, "text must be non-empty")

    raw = await reference_audio.read()
    try:
        ref_audio, ref_sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception as exc:
        raise HTTPException(400, f"could not decode reference_audio: {exc}")
    if ref_audio.ndim > 1:
        ref_audio = ref_audio.mean(axis=1)

    async with inference_lock:
        try:
            audio = await asyncio.get_event_loop().run_in_executor(
                None, _synth, text, "Ryan", speed, ref_audio, ref_sr, ref_text,
            )
        except Exception as exc:
            raise HTTPException(500, f"voice-clone synthesis failed: {exc}")

    body, ctype = _encode(audio, response_format)
    return Response(content=body, media_type=ctype)


@app.get("/v1/demo", response_class=HTMLResponse)
async def demo() -> str:
    return """<!doctype html><html><body style="font-family:sans-serif;max-width:560px;margin:40px auto">
<h2>Qwen3-TTS — quick demo</h2>
<form method="post" action="/v1/audio/speech" enctype="application/json">
  <textarea name="input" rows="3" cols="60" placeholder="Texto..."></textarea><br>
  <select name="voice"><option>Ryan</option><option>Vivian</option><option>Cherry</option></select>
  <button>Sintetizar</button>
</form>
<h3>Voice cloning (multipart):</h3>
<form method="post" action="/v1/audio/speech/clone" enctype="multipart/form-data">
  <input type="text" name="text" placeholder="Texto a sintetizar" size=60><br><br>
  <input type="text" name="ref_text" placeholder="Transcrição da reference_audio" size=60><br><br>
  <input type="file" name="reference_audio" accept="audio/*"><br><br>
  <button>Clonar voz</button>
</form>
</body></html>"""


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        log_level="info",
    )
