"""
BabelCast Subtitle Server — Faster Whisper STT + TranslateGemma 12B LLM
No TTS. Runs on all NVIDIA GPUs (Blackwell sm_120 through Ampere sm_80).

Endpoints:
  GET  /health              — Service status + model readiness
  GET  /version             — Image version info
  POST /v1/transcribe       — STT (multipart audio)
  POST /v1/translate/text   — LLM translation (JSON)
  POST /v1/audio/transcriptions — OpenAI-compatible STT
  POST /v1/chat/completions     — OpenAI-compatible LLM
"""

import os
import io
import time
import json
import logging
import asyncio
import tempfile
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("babelcast-subtitle")

# ── Config ───────────────────────────────────────────────────────────────────

COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16")
STT_MODEL = os.environ.get("STT_MODEL", "large-v3-turbo")
LLM_MODEL = os.environ.get("LLM_MODEL", "google/translate-gemma-12b-it")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Global model state ───────────────────────────────────────────────────────

stt_model = None
llm_model = None
llm_tokenizer = None
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
        # Blackwell (sm_120): INT8 tensor cores have kernel issues, use float16
        log.info(f"Detected Blackwell GPU (sm_{arch}) — using float16")
        return "float16"
    if COMPUTE_TYPE != "float16":
        return COMPUTE_TYPE
    return "float16"


async def _load_models():
    """Load STT and LLM models in background."""
    global stt_model, llm_model, llm_tokenizer

    # Load STT (Faster Whisper)
    try:
        service_status["whisper"] = "downloading"
        log.info(f"Loading Faster Whisper ({STT_MODEL}) on {DEVICE} with {_detect_compute_type()}...")
        from faster_whisper import WhisperModel
        stt_model = WhisperModel(
            STT_MODEL,
            device=DEVICE,
            compute_type=_detect_compute_type(),
        )
        service_status["whisper"] = "loaded"
        log.info(f"STT ready ({STT_MODEL})")
    except Exception as e:
        service_status["whisper"] = f"error: {e}"
        log.error(f"STT load failed: {e}")

    # Load LLM (TranslateGemma)
    try:
        service_status["llm"] = "downloading"
        log.info(f"Loading {LLM_MODEL} on {DEVICE}...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        service_status["llm"] = "loaded"
        log.info(f"LLM ready ({LLM_MODEL})")
    except Exception as e:
        service_status["llm"] = f"error: {e}"
        log.error(f"LLM load failed: {e}")


# ── App ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(asyncio.to_thread(_load_models_sync))
    yield

def _load_models_sync():
    asyncio.run(_load_models())

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
            "llm": {"warm": service_status["llm"] == "loaded"},
        },
    }


@app.get("/version")
async def version():
    cap = torch.cuda.get_device_capability() if DEVICE == "cuda" else (0, 0)
    return {
        "version": "subtitle-1.0",
        "type": "subtitle",
        "cuda_arch": f"sm_{cap[0]*10+cap[1]}",
        "compute_type": _detect_compute_type(),
        "models": {"stt": STT_MODEL, "llm": LLM_MODEL},
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
            tmp.name,
            language=language,
            beam_size=1,
            vad_filter=True,
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
    if llm_model is None or llm_tokenizer is None:
        return JSONResponse(status_code=503, content={"error": "LLM model not loaded yet"})

    body = await request.json()
    text = body.get("text", "")
    source_lang = body.get("source_lang", "en")
    target_lang = body.get("target_lang", "es")

    prompt = f"Translate from {source_lang} to {target_lang}: {text}"
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)

    t0 = time.time()
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )
    translated = llm_tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    latency_ms = round((time.time() - t0) * 1000)

    return {
        "translated_text": translated.strip(),
        "source_lang": source_lang,
        "target_lang": target_lang,
        "latency_ms": latency_ms,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if llm_model is None or llm_tokenizer is None:
        return JSONResponse(status_code=503, content={"error": "LLM model not loaded yet"})

    body = await request.json()
    messages = body.get("messages", [])

    # Build prompt from messages
    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prompt_parts.append(f"{role}: {content}")
    prompt = "\n".join(prompt_parts)

    inputs = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)

    t0 = time.time()
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=body.get("max_tokens", 512),
            do_sample=body.get("temperature", 0) > 0,
            temperature=body.get("temperature", 1.0) if body.get("temperature", 0) > 0 else 1.0,
        )
    response_text = llm_tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    latency_ms = round((time.time() - t0) * 1000)

    return {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": LLM_MODEL,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text.strip()},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": inputs["input_ids"].shape[1], "completion_tokens": len(outputs[0]) - inputs["input_ids"].shape[1]},
        "latency_ms": latency_ms,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    log.info(f"Starting BabelCast Subtitle server on :{port}")
    if DEVICE == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability()
        log.info(f"CUDA arch: sm_{cap[0]*10+cap[1]}, compute_type: {_detect_compute_type()}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
