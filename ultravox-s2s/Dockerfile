# Ultravox Speech-to-Speech — Standard GPU image (Ampere/Ada)
#
# Base: runpod/pytorch (proven on RunPod, Vast.ai, TensorDock)
# GPUs: RTX 3090/4090, L40S, A100 (driver 525+)
# Blackwell: see Dockerfile.blackwell

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# ── System dependencies ──
# aria2 + zstd: multi-connection downloads (~30% faster than curl/single-stream)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 sox libsox-dev curl aria2 zstd \
    libgomp1 libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Install hf-download helper (aria2c-based, with HF Hub fallback)
COPY hf-download /usr/local/bin/hf-download
RUN chmod +x /usr/local/bin/hf-download

# ── PyTorch (CUDA 12.4) ──
RUN pip3 install --no-cache-dir --upgrade torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# ── Ultravox deps ──
RUN pip install --no-cache-dir \
    "transformers>=4.45.0" "accelerate>=1.12.0" \
    peft librosa soundfile numpy scipy

# ── Web framework ──
# hf_xet replaces the deprecated hf_transfer — see HF_XET_* env vars below.
RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" "uvicorn[standard]>=0.32.0" \
    python-multipart httpx websockets \
    "huggingface-hub>=1.0.0" "hf_xet>=1.4.0" \
    "pydantic-settings>=2.0" \
    "aiortc>=1.9.0" av

# ── Qwen3-TTS ──
RUN pip install --no-cache-dir einops onnxruntime sox && \
    pip install --no-cache-dir "faster-qwen3-tts>=0.2.4"

RUN rm -rf /root/.cache/pip /tmp/pip-*

ENV HF_HOME=/app/.cache/huggingface \
    HF_XET_HIGH_PERFORMANCE=1 \
    HF_XET_FIXED_DOWNLOAD_CONCURRENCY=50

COPY api/ /app/api/
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh && chmod -R 777 /app/.cache 2>/dev/null || true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/app/start.sh"]
