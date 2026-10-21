"""whisper-align — WhisperX word-level forced alignment service.

FastAPI wrapper exposing a single content endpoint so the canal-dark /
project-philosofi pipeline can map narration audio to per-word timings:

  GET  /health              — readiness probe.
  POST /v1/audio/align      — multipart {audio, text?, language?}.
                              `audio` is mandatory; `text` (optional)
                              forces the transcript so the alignment
                              uses the script.json wording instead of
                              re-transcribing. `language` (optional, ISO
                              code) skips language detection.

Returns:
  {
    "language": "en",
    "duration": 245.6,
    "segments": [{"start": 0.0, "end": 4.2, "text": "Marcus..."}, ...],
    "words":    [{"word": "Marcus", "start": 0.12, "end": 0.45, "score": 0.92}, ...]
  }

Env:
  IDLE_TIMEOUT_MIN — auto-shutdown after no requests (default 15).
  WHISPERX_MODEL   — whisper repo id (default large-v3).
  PORT             — listen port (default 8000).
"""

import asyncio
import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
import uvicorn

from idle_watchdog import add_idle_middleware, start_watchdog, touch_activity

MODEL_NAME = os.environ.get("WHISPERX_MODEL", "large-v3")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# WhisperX faster-whisper backend wants float16 on GPU and int8 on CPU.
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
BATCH_SIZE = 16 if DEVICE == "cuda" else 4

inference_lock = asyncio.Lock()
boot_ts = time.time()
asr_model = None  # whisperx.load_model(...)
align_cache: dict = {}  # language_code → (model, metadata)
load_error: Optional[str] = None


def _load_asr_model() -> None:
    global asr_model, load_error
    try:
        import whisperx  # type: ignore
        asr_model = whisperx.load_model(
            MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE,
        )
    except Exception as exc:
        import traceback
        load_error = f"{exc}\n{traceback.format_exc()}"


def _get_align_model(language_code: str):
    """Lazy-load + cache wav2vec2 align model per language. WhisperX ships
    a default per-language map (en→wav2vec2-large-960h-lv60-self, etc.)."""
    if language_code in align_cache:
        return align_cache[language_code]
    import whisperx  # type: ignore
    model, metadata = whisperx.load_align_model(
        language_code=language_code, device=DEVICE,
    )
    align_cache[language_code] = (model, metadata)
    return model, metadata


async def _load_asr_async() -> None:
    await asyncio.get_event_loop().run_in_executor(None, _load_asr_model)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Fire-and-forget weight download — uvicorn must accept /health
    # immediately so the gateway boot poller doesn't time out.
    asyncio.create_task(_load_asr_async())
    asyncio.create_task(start_watchdog())
    yield


app = FastAPI(title="whisper-align (ai-gateway)", lifespan=lifespan)
add_idle_middleware(app)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "healthy" if load_error is None else "degraded",
        "model_loaded": asr_model is not None,
        "model": MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "uptime_s": round(time.time() - boot_ts, 1),
        "gpu_available": torch.cuda.is_available(),
        "load_error": load_error,
    }


def _run_align(audio_path: str, text: Optional[str], lang_hint: Optional[str]) -> dict:
    """Transcribe + force-align. If `text` is provided, we still run whisper
    transcription to obtain segment boundaries (whisperx.align needs segments
    with rough timestamps), then overlay the user-supplied transcript onto
    each segment when text length matches; otherwise we just return the
    whisper transcription aligned to itself. Returns the JSON dict."""
    import whisperx  # type: ignore

    if asr_model is None:
        raise RuntimeError(load_error or "ASR model not loaded")

    audio = whisperx.load_audio(audio_path)
    duration = float(len(audio)) / 16_000.0

    transcribe_kwargs = {"batch_size": BATCH_SIZE}
    if lang_hint:
        transcribe_kwargs["language"] = lang_hint
    result = asr_model.transcribe(audio, **transcribe_kwargs)
    language = result.get("language") or lang_hint or "en"
    segments = result.get("segments") or []

    # If caller supplied the target transcript, replace whisper's text with
    # it so the alignment uses the script's exact wording. We naively split
    # the user transcript across segments by character ratio — for canal-dark
    # this is fine because the audio was generated FROM that script.
    if text:
        text_clean = text.strip()
        if text_clean and segments:
            total_chars = sum(max(1, len(s.get("text") or "")) for s in segments)
            cursor = 0
            for seg in segments:
                seg_chars = max(1, len(seg.get("text") or ""))
                share = seg_chars / total_chars
                end_cursor = min(len(text_clean), cursor + max(1, int(round(share * len(text_clean)))))
                seg["text"] = text_clean[cursor:end_cursor].strip()
                cursor = end_cursor
            if cursor < len(text_clean):
                segments[-1]["text"] = (segments[-1].get("text") or "") + " " + text_clean[cursor:].strip()

    align_model, align_meta = _get_align_model(language)
    aligned = whisperx.align(
        segments, align_model, align_meta, audio, DEVICE,
        return_char_alignments=False,
    )

    out_segments = []
    for s in aligned.get("segments") or []:
        out_segments.append({
            "start": float(s.get("start", 0.0)),
            "end":   float(s.get("end", 0.0)),
            "text":  (s.get("text") or "").strip(),
        })
    out_words = []
    for w in aligned.get("word_segments") or []:
        if w.get("start") is None or w.get("end") is None:
            continue
        out_words.append({
            "word":  w.get("word") or "",
            "start": float(w["start"]),
            "end":   float(w["end"]),
            "score": float(w.get("score") or 0.0),
        })

    return {
        "language": language,
        "duration": round(duration, 3),
        "segments": out_segments,
        "words":    out_words,
    }


@app.post("/v1/audio/align")
async def align_audio(
    audio: UploadFile = File(...),
    text: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
) -> dict:
    """Forced word-level alignment. `audio` is the mandatory clip (any
    format ffmpeg can decode). `text` is the optional target transcript;
    when provided we overlay it onto whisper's segments so the alignment
    matches the script wording exactly. `language` is an optional ISO
    code (en, pt, fr...) — when omitted whisper detects it."""
    touch_activity()
    if asr_model is None:
        raise HTTPException(503, f"model not loaded yet (err={load_error or 'booting'})")

    raw = await audio.read()
    if not raw:
        raise HTTPException(400, "audio file is empty")

    suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        async with inference_lock:
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, _run_align, tmp_path, text, language,
                )
            except Exception as exc:
                raise HTTPException(500, f"alignment failed: {exc}")
        return result
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        log_level="info",
    )
