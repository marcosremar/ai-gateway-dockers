# ── BabelCast Subtitle — Universal GPU Image ────────────────────────────────
# Single image for ALL GPU architectures: Blackwell (sm_120), Ada (sm_89),
# Ampere (sm_80/86), Hopper (sm_90). No separate Blackwell variant needed.
#
# CUDA 12.8.1 is the minimum for RTX 5090/5080 (sm_120) and includes kernels
# for all older architectures. One image runs everywhere.
#
# LLM: TranslateGemma 4B Q8_0 via llama.cpp (flash_attn + n_batch=1024)
# STT: Faster Whisper large-v3-turbo via CTranslate2
# ─────────────────────────────────────────────────────────────────────────────

ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    COMPUTE_TYPE=float16 \
    LLM_REPO=bullerwins/translategemma-4b-it-GGUF \
    LLM_FILENAME=translategemma-4b-it-Q8_0.gguf \
    N_CTX=1024 \
    N_BATCH=1024 \
    N_UBATCH=512 \
    FLASH_ATTN=true

# System deps — minimal for inference only (no build tools needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    ffmpeg libsndfile1 curl && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Step 1: PyTorch from CUDA 12.8 index
ARG CUDA_INDEX=https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir torch==2.11.0 --index-url ${CUDA_INDEX}

# Step 2: llama-cpp-python — installed at runtime in start.sh using pre-built CUDA wheel.
# Cannot install at build time because buildx platform detection prevents finding the right wheel.

# Step 3: Everything else from PyPI
RUN pip install --no-cache-dir \
    "faster-whisper>=1.2.1" \
    "ctranslate2>=4.7.1" \
    httpx \
    huggingface-hub \
    python-multipart \
    uvicorn[standard] \
    fastapi

# Pre-download GGUF model into image cache (saves 2-5 min on cold boot)
RUN python3 -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download('bullerwins/translategemma-4b-it-GGUF', 'translategemma-4b-it-Q8_0.gguf')" \
    || echo "GGUF pre-download skipped (will download at boot)"

COPY server.py .
COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["/app/start.sh"]
