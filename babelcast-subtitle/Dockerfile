# ── BabelCast Subtitle — Universal GPU Image ────────────────────────────────
# Single image for ALL GPU architectures: Blackwell (sm_120), Ada (sm_89),
# Ampere (sm_80/86), Hopper (sm_90). No separate Blackwell variant needed.
#
# Build:
#   docker build -t marcosremar/babelcast-subtitle:latest .
#   docker build --build-arg BASE_IMAGE=nvidia/cuda:13.0.2-cudnn-runtime-ubuntu22.04 -t ... .
#
# CUDA 12.8.1 is the minimum for RTX 5090/5080 (sm_120) and includes kernels
# for all older architectures. One image runs everywhere.
# ─────────────────────────────────────────────────────────────────────────────

ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Default to float16 — safe on all GPUs including Blackwell.
    # INT8 tensor cores have issues on sm_120; float16 is universally reliable.
    COMPUTE_TYPE=float16

# System deps — minimal for inference only
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    ffmpeg libsndfile1 curl && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps — CTranslate2 4.7.1+ has Blackwell sm_120 fixes
# PyTorch cu128 wheels work on CUDA 12.8+ (Blackwell + older GPUs)
ARG CUDA_INDEX=https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir \
    faster-whisper>=1.2.1 \
    "ctranslate2>=4.7.1" \
    torch==2.11.0 --index-url ${CUDA_INDEX} && \
    pip install --no-cache-dir \
    transformers>=4.48.0 \
    accelerate \
    sentencepiece \
    protobuf \
    uvicorn[standard] \
    fastapi

COPY server.py .
COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8000

# Health check — gateway polls /health to know when models are loaded
HEALTHCHECK --interval=10s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["/app/start.sh"]
