"""
BabelCast Subtitle Server — Faster Whisper STT + TranslateGemma 4B LLM (llama.cpp)
No TTS. Runs on all NVIDIA GPUs (Blackwell sm_120 through Ampere sm_80).

LLM runs as a llama.cpp subprocess with GGUF quantized model for maximum
inference speed. Flash attention + large batch size for optimal throughput.

Endpoints:
  GET  /health              — Service status + model readiness
  GET  /version             — Image version info
  POST /v1/transcribe       — STT (multipart audio)
  POST /v1/translate/text   — LLM translation (JSON)
  POST /v1/audio/transcriptions — OpenAI-compatible STT
  POST /v1/chat/completions     — OpenAI-compatible LLM
"""

import os
import time
import logging
import asyncio
import subprocess
import tempfile
from contextlib import asynccontextmanager

import torch
import httpx
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("babelcast-subtitle")

# ── Config ───────────────────────────────────────────────────────────────────

COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16")
STT_MODEL = os.environ.get("STT_MODEL", "large-v3-turbo")
LLM_REPO = os.environ.get("LLM_REPO", "bullerwins/translategemma-4b-it-GGUF")
LLM_FILENAME = os.environ.get("LLM_FILENAME", "translategemma-4b-it-Q8_0.gguf")
LLM_PORT = 8002
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# llama.cpp tuning — overridable via env
N_CTX = int(os.environ.get("N_CTX", "1024"))
N_BATCH = int(os.environ.get("N_BATCH", "1024"))
N_UBATCH = int(os.environ.get("N_UBATCH", "512"))
FLASH_ATTN = os.environ.get("FLASH_ATTN", "true").lower() == "true"

# ── Global model state ───────────────────────────────────────────────────────

stt_model = None
llm_client: httpx.AsyncClient | None = None
boot_time = time.time()

service_status = {
    "whisper": "pending",
    "llm": "pending",
}


def _detect_compute_type() -> str:
    """Auto-detect optimal compute type based on GPU architecture."""
    if DEVICE != "cuda":
        return "float32"
    cap = torch.cuda.get_device_capability()
    arch = cap[0] * 10 + cap[1]
    if arch >= 120:
        log.info(f"Detected Blackwell GPU (sm_{arch}) — using float16")
        return "float16"
    return COMPUTE_TYPE


# ── Background model loading ────────────────────────────────────────────────

async def _load_stt():
    """Load Faster Whisper STT model."""
    global stt_model
    try:
        service_status["whisper"] = "downloading"
        log.info(f"Loading Faster Whisper ({STT_MODEL}) on {DEVICE}...")
        from faster_whisper import WhisperModel
        stt_model = await asyncio.to_thread(
            WhisperModel, STT_MODEL, device=DEVICE, compute_type=_detect_compute_type()
        )
        service_status["whisper"] = "loaded"
        log.info(f"STT ready ({STT_MODEL})")
    except Exception as e:
        service_status["whisper"] = f"error: {e}"
        log.error(f"STT load failed: {e}")


async def _load_llm():
    """Download GGUF model, start llama.cpp subprocess, wait for ready."""
    global llm_client
    try:
        service_status["llm"] = "downloading"
        log.info(f"Downloading GGUF: {LLM_REPO}/{LLM_FILENAME}...")

        from huggingface_hub import hf_hub_download, try_to_load_from_cache
        # Fast path: check cache first
        cached = await asyncio.to_thread(try_to_load_from_cache, LLM_REPO, LLM_FILENAME)
        if cached:
            gguf_path = cached
            log.info(f"GGUF found in cache: {gguf_path}")
        else:
            gguf_path = await asyncio.to_thread(hf_hub_download, LLM_REPO, LLM_FILENAME)
            log.info(f"GGUF downloaded: {gguf_path}")
    except Exception as e:
        service_status["llm"] = f"download_failed: {e}"
        log.error(f"GGUF download failed: {e}")
        return

    # Start llama.cpp server subprocess
    service_status["llm"] = "starting"
    cmd = [
        "python3", "-m", "llama_cpp.server",
        "--host", "127.0.0.1", "--port", str(LLM_PORT),
        "--model", gguf_path,
        "--n_gpu_layers", "99",
        "--n_ctx", str(N_CTX),
        "--n_batch", str(N_BATCH),
        "--n_ubatch", str(N_UBATCH),
    ]
    if FLASH_ATTN:
        cmd += ["--flash_attn", "true"]

    log.info(f"Starting llama.cpp: flash_attn={FLASH_ATTN}, n_batch={N_BATCH}, n_ctx={N_CTX}")
    try:
        with open("/tmp/llama.log", "w") as log_f:
            subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    except Exception as e:
        service_status["llm"] = f"start_failed: {e}"
        log.error(f"Failed to start llama.cpp: {e}")
        return

    # Poll until ready (up to 5 min)
    llm_client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{LLM_PORT}", timeout=10.0)
    for attempt in range(60):
        await asyncio.sleep(3 if attempt < 10 else 5)
        try:
            r = await llm_client.get("/health")
            if r.status_code == 200:
                service_status["llm"] = "ready"
                log.info(f"llama.cpp ready after {attempt * 3}s")
                return
        except Exception:
            pass
    service_status["llm"] = "failed: timeout"
    log.error("llama.cpp failed to start within timeout")


# ── App ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(_load_stt())
    asyncio.create_task(_load_llm())
    yield

app = FastAPI(title="BabelCast Subtitle", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "uptime": round(time.time() - boot_time, 1),
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
        "compute_type": _detect_compute_type(),
        "services": service_status,
        "model_warmth": {
            "stt": {"warm": service_status["whisper"] == "loaded"},
            "llm": {"warm": service_status["llm"] == "ready"},
        },
    }


@app.get("/version")
async def version():
    cap = torch.cuda.get_device_capability() if DEVICE == "cuda" else (0, 0)
    return {
        "version": "subtitle-2.0",
        "type": "subtitle",
        "cuda_arch": f"sm_{cap[0]*10+cap[1]}",
        "compute_type": _detect_compute_type(),
        "models": {"stt": STT_MODEL, "llm": f"{LLM_REPO}/{LLM_FILENAME}"},
        "llm_config": {"flash_attn": FLASH_ATTN, "n_batch": N_BATCH, "n_ctx": N_CTX},
    }


# ── STT Endpoints ────────────────────────────────────────────────────────────

@app.post("/v1/transcribe")
@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(None),
):
    if stt_model is None:
        return JSONResponse(status_code=503, content={"error": "STT model not loaded yet"})

    audio_bytes = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()

        t0 = time.time()
        segments, info = stt_model.transcribe(
            tmp.name, language=language, beam_size=1, vad_filter=True,
        )
        text = " ".join(s.text.strip() for s in segments)
        latency_ms = round((time.time() - t0) * 1000)

    return {
        "text": text,
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration": round(info.duration, 2),
        "latency_ms": latency_ms,
    }


# ── LLM Endpoints ───────────────────────────────────────────────────────────

@app.post("/v1/translate/text")
async def translate_text(request: Request):
    if llm_client is None or service_status["llm"] != "ready":
        return JSONResponse(status_code=503, content={"error": "LLM not ready yet"})

    body = await request.json()
    text = body.get("text", "")
    source_lang = body.get("source_lang", "en")
    target_lang = body.get("target_lang", "es")

    prompt = f"Translate from {source_lang} to {target_lang}: {text}"

    t0 = time.time()
    r = await llm_client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0,
    })
    data = r.json()
    translated = data["choices"][0]["message"]["content"].strip()
    latency_ms = round((time.time() - t0) * 1000)

    return {
        "translated_text": translated,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "latency_ms": latency_ms,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if llm_client is None or service_status["llm"] != "ready":
        return JSONResponse(status_code=503, content={"error": "LLM not ready yet"})

    body = await request.json()
    t0 = time.time()
    r = await llm_client.post("/v1/chat/completions", json={
        "messages": body.get("messages", []),
        "max_tokens": body.get("max_tokens", 512),
        "temperature": body.get("temperature", 0),
    })
    data = r.json()
    latency_ms = round((time.time() - t0) * 1000)

    data["latency_ms"] = latency_ms
    return data


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    log.info(f"Starting BabelCast Subtitle server on :{port}")
    if DEVICE == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability()
        log.info(f"CUDA arch: sm_{cap[0]*10+cap[1]}, compute_type: {_detect_compute_type()}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
