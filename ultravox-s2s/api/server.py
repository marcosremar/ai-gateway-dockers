"""
Ultravox Speech-to-Speech Server

Pipeline: Audio → Ultravox (audio-native LLM) → Qwen3-TTS → Audio

Ultravox processes speech directly without separate STT — the audio
embeddings are projected into the LLM's latent space via a multimodal
projector (Whisper encoder → LLM).

Endpoints:
  GET  /health              - Health + model status
  POST /v1/speech           - Audio in → full pipeline → audio out (JSON)
  POST /api/stream-audio    - Audio in → SSE streaming
  POST /api/offer           - WebRTC SDP signaling
  WS   /ws/stream           - WebSocket bidirectional audio
  POST /v1/tts              - Text → TTS → audio
"""

# ── Cold-start optimizations ────────────────────────────────────────────────
# Run BEFORE any torch import so torch.compile picks up the cache env vars.
# Catch ALL exceptions — never let an OPTIONAL optimization block startup.
import os, sys
sys.path.insert(0, "/app")  # for coldstart.py
try:
    from coldstart import bootstrap, prefetch_safetensors
    bootstrap(torch_cache_dir=os.environ.get("TORCHINDUCTOR_CACHE_DIR", "/app/.torch-cache"))
except Exception as _coldstart_err:
    print(
        f"[server] coldstart unavailable ({type(_coldstart_err).__name__}: "
        f"{_coldstart_err}) — running without optimizations",
        file=sys.stderr,
    )
    prefetch_safetensors = lambda *a, **k: 0.0  # no-op stub

import asyncio
import base64
import io
import re
import time
import threading
import json
import struct
from typing import Optional, Generator

import numpy as np
import soundfile as sf
import torch
import librosa
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Configuration ──
ULTRAVOX_MODEL = os.environ.get("ULTRAVOX_MODEL", "fixie-ai/ultravox-v0_6-llama-3_1-8b")
TTS_MODEL_BASE = os.environ.get("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
TTS_MODEL_CLONE = os.environ.get("TTS_MODEL_CLONE", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT",
    "You are a friendly and helpful Portuguese language tutor named Marcela. "
    "Respond naturally in Portuguese. Keep responses concise (1-2 sentences). "
    "If the student speaks in another language, gently guide them back to Portuguese."
)
TTS_SPEAKER = os.environ.get("TTS_SPEAKER", "serena")
SAMPLE_RATE = 24000

# ── Global state ──
ultravox_pipe = None
tts_model = None
loading_state = {"ultravox": False, "tts": False, "error": None}
start_time = time.time()
conversation_history = []

# ── WebRTC ──
try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, MediaStreamTrack
    from aiortc.contrib.media import MediaRelay
    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False
    MediaStreamTrack = object

pcs = set()


def load_ultravox():
    """Load Ultravox model (background thread)."""
    global ultravox_pipe
    import transformers
    print(f"[load] Loading Ultravox: {ULTRAVOX_MODEL}...")

    # NOTE: We previously called prefetch_safetensors() here, but real GPU
    # benchmarking on Vast.ai RTX 4090 (2026-04-08) showed it ADDED ~37ms of
    # overhead — NVMe is fast enough that the second read from page cache
    # doesn't beat the first read by enough to justify the extra pass.
    # The Ultravox safetensors file is also small (~1.3 GB) so the speedup
    # is bounded by ~0.2s even with fastsafetensors.
    # Kept the import for forward compat but don't call it.
    _ = prefetch_safetensors  # silence unused warning

    t0 = time.time()
    try:
        ultravox_pipe = transformers.pipeline(
            model=ULTRAVOX_MODEL,
            trust_remote_code=True,
            device="cuda:0",
            torch_dtype=torch.bfloat16,
        )
        loading_state["ultravox"] = True
        print(f"[load] Ultravox loaded in {time.time()-t0:.1f}s")
    except Exception as e:
        loading_state["error"] = f"Ultravox failed: {e}"
        print(f"[load] Ultravox load error: {e}")
        import traceback; traceback.print_exc()


def load_tts():
    """Load Qwen3-TTS via faster_qwen3_tts (background thread)."""
    global tts_model
    t0 = time.time()
    try:
        from faster_qwen3_tts import FasterQwen3TTS
        print(f"[load] Loading TTS: {TTS_MODEL_BASE}...")
        tts_model = FasterQwen3TTS.from_pretrained(TTS_MODEL_BASE, device="cuda:0")
        # Pre-warmup: capture CUDA graphs NOW (at startup) so they don't conflict
        # with the LLM background thread during requests.
        print("[load] Pre-warming TTS CUDA graphs...")
        _warmup_t0 = time.time()
        for _chunk in tts_model.generate_custom_voice_streaming(
            text="Hello.", speaker=TTS_SPEAKER, language="Portuguese", chunk_size=4,
        ):
            pass  # Just run through to trigger warmup
        print(f"[load] TTS warmup done in {time.time()-_warmup_t0:.1f}s")
        loading_state["tts"] = True
        print(f"[load] TTS loaded in {time.time()-t0:.1f}s")
    except Exception as e:
        loading_state["error"] = f"TTS failed: {e}"
        print(f"[load] TTS load error: {e}")
        import traceback; traceback.print_exc()


def load_models_background():
    """Load all models sequentially in background."""
    if not torch.cuda.is_available():
        loading_state["error"] = "No CUDA available"
        print("[load] No CUDA — skipping model loading")
        return
    torch.set_float32_matmul_precision("high")
    load_ultravox()
    if loading_state.get("error"):
        return
    load_tts()
    vram = torch.cuda.memory_allocated(0) / 1024**3
    print(f"[load] All models loaded. VRAM: {vram:.1f}GB")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start model loading on startup."""
    print("=" * 60)
    print("Ultravox Speech-to-Speech — Starting")
    print(f"  Ultravox: {ULTRAVOX_MODEL}")
    print(f"  TTS:      {TTS_MODEL_BASE}")
    print(f"  Speaker:  {TTS_SPEAKER}")
    print("  Server is READY — models loading in background...")
    print("=" * 60)
    bg = threading.Thread(target=load_models_background, daemon=True)
    bg.start()
    yield
    # Cleanup WebRTC
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


app = FastAPI(title="Ultravox S2S", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Helpers ──

def numpy_to_wav_bytes(audio_np: np.ndarray, sr: int) -> bytes:
    """Convert a numpy chunk to WAV bytes."""
    audio_f32 = audio_np.astype(np.float32)
    if audio_f32.ndim > 1:
        audio_f32 = audio_f32.mean(axis=1)
    if np.abs(audio_f32).max() <= 1.0:
        audio_i16 = (audio_f32 * 32767).clip(-32768, 32767).astype(np.int16)
    else:
        audio_i16 = audio_f32.clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, audio_i16, sr or SAMPLE_RATE, format="WAV")
    return buf.getvalue()


def decode_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode audio bytes to numpy array at 16kHz mono."""
    buf = io.BytesIO(audio_bytes)
    audio, sr = sf.read(buf)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    return audio.astype(np.float32), sr


def run_ultravox(audio: np.ndarray, sr: int = 16000) -> str:
    """Run Ultravox inference: audio → text response."""
    if ultravox_pipe is None:
        raise RuntimeError("Ultravox not loaded yet")

    turns = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in conversation_history[-6:]:  # last 3 turns
        turns.append(h)
    turns.append({"role": "user", "content": "<|audio|>"})

    result = ultravox_pipe(
        {"audio": audio, "turns": turns, "sampling_rate": sr},
        max_new_tokens=100,
    )
    text = result[0]["generated_text"] if isinstance(result, list) else str(result)
    return text


# Phrase boundary: sentence-ending punctuation or comma followed by a space
_PHRASE_BREAK = re.compile(r'(?<=[.!?…])\s|[,;]\s+(?=\S)')
_MIN_PHRASE_CHARS = 10  # don't start TTS on tiny fragments


def run_ultravox_streaming(audio: np.ndarray, sr: int = 16000, system_prompt: str = None) -> Generator[str, None, None]:
    """Streaming Ultravox: yields text tokens as the LLM generates them.

    Uses TextIteratorStreamer so model.generate() runs in a background thread
    and tokens are produced in real-time without waiting for the full response.
    """
    from transformers import TextIteratorStreamer

    if ultravox_pipe is None:
        raise RuntimeError("Ultravox not loaded yet")

    prompt = system_prompt or SYSTEM_PROMPT
    turns = [{"role": "system", "content": prompt}]
    for h in conversation_history[-6:]:
        turns.append(h)
    turns.append({"role": "user", "content": "<|audio|>"})

    model_inputs = ultravox_pipe.preprocess({"audio": audio, "turns": turns, "sampling_rate": sr})
    device = ultravox_pipe.model.device
    model_inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in model_inputs.items()}

    streamer = TextIteratorStreamer(
        ultravox_pipe.tokenizer,
        skip_special_tokens=True,
        skip_prompt=True,
    )

    terminators = [ultravox_pipe.tokenizer.eos_token_id]
    if "<|eot_id|>" in ultravox_pipe.tokenizer.added_tokens_encoder:
        terminators.append(ultravox_pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"))

    gen_kwargs = {
        **model_inputs,
        "max_new_tokens": 100,
        "repetition_penalty": 1.1,
        "eos_token_id": terminators,
        "streamer": streamer,
    }
    t = threading.Thread(target=ultravox_pipe.model.generate, kwargs=gen_kwargs, daemon=True)
    t.start()

    for token in streamer:
        yield token


def run_tts(text: str, speaker: str = TTS_SPEAKER) -> bytes:
    """Run TTS: text → WAV bytes via FasterQwen3TTS."""
    if tts_model is None:
        raise RuntimeError("TTS not loaded yet")

    audio_data, sr = tts_model.generate_custom_voice(
        text=text,
        speaker=speaker,
        language="Portuguese",
    )

    # audio_data is a list of numpy arrays
    if isinstance(audio_data, list):
        audio_np = np.concatenate(audio_data) if len(audio_data) > 1 else audio_data[0]
    elif isinstance(audio_data, torch.Tensor):
        audio_np = audio_data.cpu().numpy()
    else:
        audio_np = np.array(audio_data)

    audio_np = audio_np.astype(np.float32)
    if audio_np.max() <= 1.0:
        audio_np = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
    else:
        audio_np = audio_np.clip(-32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    sf.write(buf, audio_np, sr or SAMPLE_RATE, format="WAV")
    return buf.getvalue()


# ── Endpoints ──

@app.get("/health")
async def health():
    uptime = int(time.time() - start_time)
    ready = loading_state["ultravox"] and loading_state["tts"]
    vram = torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0

    # Build transport URLs from request
    host = os.environ.get("VAST_HOST", "")
    port = os.environ.get("VAST_PORT", "8000")

    return {
        "status": "ok" if ready else "loading",
        "phase": "ready" if ready else "loading",
        "uptime_s": uptime,
        "error": loading_state.get("error"),
        "services": {
            "ultravox": "loaded" if loading_state["ultravox"] else "loading",
            "tts": "loaded" if loading_state["tts"] else "loading",
        },
        "models": {
            "ultravox": ULTRAVOX_MODEL,
            "tts": TTS_MODEL_BASE,
        },
        "vram_gb": round(vram, 1),
        "streaming": "sse",
        "transports": {
            "sse": {"endpoint": "/api/stream-audio"},
            "websocket": {"url": "/ws/stream"},
            "webrtc": {"signalingUrl": "/api/offer"} if AIORTC_AVAILABLE else None,
        },
    }


class SpeechRequest(BaseModel):
    audio: str  # base64 WAV
    language: str = "pt"
    speaker: str = TTS_SPEAKER


@app.post("/api/reset")
async def reset_history():
    """Clear conversation history."""
    conversation_history.clear()
    return {"status": "ok", "message": "History cleared"}


@app.post("/v1/speech")
async def api_speech(req: SpeechRequest):
    """Full pipeline: audio → Ultravox → TTS → audio (JSON response)."""
    audio_bytes = base64.b64decode(req.audio)
    audio_np, sr = decode_audio(audio_bytes)

    t0 = time.time()
    response_text = await asyncio.to_thread(run_ultravox, audio_np, sr)
    llm_ms = int((time.time() - t0) * 1000)

    t1 = time.time()
    audio_out = await asyncio.to_thread(run_tts, response_text, req.speaker)
    tts_ms = int((time.time() - t1) * 1000)

    # Update history
    conversation_history.append({"role": "user", "content": ""})
    conversation_history.append({"role": "assistant", "content": response_text})

    return {
        "response": response_text,
        "audio": base64.b64encode(audio_out).decode(),
        "content_type": "audio/wav",
        "timing": {
            "ultravox_ms": llm_ms,
            "tts_ms": tts_ms,
            "total_ms": llm_ms + tts_ms,
        },
    }


@app.post("/api/stream-audio")
async def stream_audio(request: Request):
    """SSE streaming: audio → Ultravox → TTS streaming → chunked audio."""
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        form = await request.form()
        audio_field = form.get("audio")
        audio_bytes = await audio_field.read() if hasattr(audio_field, "read") else bytes(audio_field)
        custom_system_prompt = form.get("system_prompt") or None
        req_language = form.get("language") or "pt"
        history_json = form.get("history") or None
    else:
        audio_bytes = await request.body()
        custom_system_prompt = None
        req_language = "pt"
        history_json = None

    audio_np, sr = decode_audio(audio_bytes)

    # Map ISO language code to TTS language name
    _LANG_MAP = {
        "pt": "Portuguese", "en": "English", "fr": "French",
        "es": "Spanish", "de": "German", "it": "Italian",
        "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
    }
    tts_language = _LANG_MAP.get(req_language, req_language if len(req_language) > 2 else "Portuguese")
    active_prompt = custom_system_prompt or SYSTEM_PROMPT
    # Fresh history when a custom prompt is sent (language switch etc.)
    if custom_system_prompt:
        conversation_history.clear()

    def generate_sse():
        """
        Pipelined streaming: LLM tokens arrive in real-time, TTS starts on
        the first complete phrase rather than waiting for the full response.

        Timeline (target):
          0ms  → LLM starts (audio encoder + first token ~100-200ms)
          ~200ms → first phrase complete ("Muito bem!")
          ~500ms → TTS first chunk for that phrase → audio sent to client
          meanwhile LLM keeps generating the rest of the response
        """
        if tts_model is None:
            yield f"event: error\ndata: {json.dumps({'message': 'TTS not loaded yet'})}\n\n"
            return

        yield f"event: status\ndata: {json.dumps({'stage': 'ultravox'})}\n\n"

        t0 = time.time()
        buf = ""
        full_text = ""
        first_audio = True

        def _tts_phrase(phrase: str):
            nonlocal first_audio
            # CUDA graphs are pre-captured at startup, so replaying them here
            # does NOT conflict with the LLM background thread.
            for c_np, c_sr, _ in tts_model.generate_custom_voice_streaming(
                text=phrase,
                speaker=TTS_SPEAKER,
                language=tts_language,
                chunk_size=4,
            ):
                d: dict = {"chunk": base64.b64encode(numpy_to_wav_bytes(c_np, c_sr)).decode()}
                if first_audio:
                    d["ttfc_ms"] = int((time.time() - t0) * 1000)
                    first_audio = False
                yield f"event: audio\ndata: {json.dumps(d)}\n\n"

        try:
            for token in run_ultravox_streaming(audio_np, sr, system_prompt=active_prompt):
                buf += token
                full_text += token

                # Send TTS as soon as we have a meaningful phrase boundary
                if len(buf) >= _MIN_PHRASE_CHARS and _PHRASE_BREAK.search(buf):
                    # Split on last boundary to keep remainder for next phrase
                    m = list(_PHRASE_BREAK.finditer(buf))[-1]
                    phrase, buf = buf[:m.end()].strip(), buf[m.end():]
                    if phrase:
                        yield from _tts_phrase(phrase)

            # Flush remaining buffer (end of LLM generation)
            if buf.strip():
                yield from _tts_phrase(buf.strip())

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
            return

        total_ms = int((time.time() - t0) * 1000)
        yield f"event: response\ndata: {json.dumps({'response': full_text, 'total_ms': total_ms})}\n\n"
        yield f"event: complete\ndata: {json.dumps({'response': full_text, 'timing': {'total_ms': total_ms}})}\n\n"

        conversation_history.append({"role": "user", "content": ""})
        conversation_history.append({"role": "assistant", "content": full_text})

    return StreamingResponse(generate_sse(), media_type="text/event-stream")


class TTSRequest(BaseModel):
    text: str
    speaker: str = TTS_SPEAKER


@app.post("/v1/tts")
async def api_tts(req: TTSRequest):
    """Text → TTS → audio."""
    t0 = time.time()
    audio_out = await asyncio.to_thread(run_tts, req.text, req.speaker)
    tts_ms = int((time.time() - t0) * 1000)
    return {
        "audio": base64.b64encode(audio_out).decode(),
        "content_type": "audio/wav",
        "tts_ms": tts_ms,
    }




# ── WebSocket ──

@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    """Bidirectional audio WebSocket."""
    await ws.accept()
    try:
        while True:
            data = await ws.receive_bytes()
            if len(data) < 100:
                continue

            audio_np, sr = decode_audio(data)

            # Ultravox
            response_text = await asyncio.to_thread(run_ultravox, audio_np, sr)
            await ws.send_json({"type": "response", "text": response_text})

            # TTS
            audio_out = await asyncio.to_thread(run_tts, response_text)
            audio_b64 = base64.b64encode(audio_out).decode()
            await ws.send_json({"type": "audio", "audio": audio_b64, "content_type": "audio/wav"})

            conversation_history.append({"role": "user", "content": ""})
            conversation_history.append({"role": "assistant", "content": response_text})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ws] error: {e}")


# ── WebRTC ──

@app.post("/api/offer")
async def webrtc_offer(request: Request):
    """WebRTC SDP signaling endpoint."""
    if not AIORTC_AVAILABLE:
        return JSONResponse({"error": "aiortc not installed"}, status_code=501)

    body = await request.json()
    offer = RTCSessionDescription(sdp=body["sdp"], type=body["type"])

    pc = RTCPeerConnection(
        configuration=RTCConfiguration(
            iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
        )
    )
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_state():
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        if track.kind != "audio":
            return

        async def recv_audio():
            frames = []
            while True:
                try:
                    frame = await asyncio.wait_for(track.recv(), timeout=5.0)
                    frames.append(frame)
                    if len(frames) >= 50:  # ~1s of audio
                        # Process accumulated audio
                        pcm = b"".join(f.to_ndarray().tobytes() for f in frames)
                        audio_np = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                        if len(audio_np) > 0:
                            response = await asyncio.to_thread(run_ultravox, audio_np, 16000)
                            # Send response via data channel if available
                        frames = []
                except asyncio.TimeoutError:
                    if frames:
                        frames = []
                except Exception:
                    break

        asyncio.ensure_future(recv_audio())

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    })
