# ── Wan 2.1 Image-to-Video — 360° Background Animator ───────────────────────
# Anima frames estáticos de panoramas 360° gerando vídeos MP4 de 3-5 segundos.
#
# Model: Wan-AI/Wan2.1-I2V-14B-480P (~28GB)
# VRAM:  16GB+ com CPU offload  |  24GB nativo
# GPU:   RTX 4090, A100 40GB, H100 80GB
#
# Build:
#   docker build -t marcosremar/wan-i2v:latest \
#     --build-arg HF_TOKEN=hf_... .
#
# Run (local):
#   docker run --gpus all -p 8095:8095 \
#     -v /workspace/hf-cache:/root/.cache/huggingface \
#     -e IDLE_TIMEOUT_MIN=15 \
#     marcosremar/wan-i2v:latest
# ─────────────────────────────────────────────────────────────────────────────

ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_ID=Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
    DEFAULT_WIDTH=832 \
    DEFAULT_HEIGHT=480 \
    DEFAULT_FRAMES=81 \
    DEFAULT_FPS=16 \
    CPU_OFFLOAD=1 \
    IDLE_TIMEOUT_MIN=15 \
    HF_HOME=/workspace/.cache/huggingface \
    TORCHINDUCTOR_CACHE_DIR=/workspace/.torch-cache \
    TORCHINDUCTOR_FX_GRAPH_CACHE=1 \
    TORCHINDUCTOR_AUTOGRAD_CACHE=1

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    build-essential git curl aria2 zstd \
    ffmpeg libgl1 libglib2.0-0 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    rm -rf /var/lib/apt/lists/*

# ── hf-download helper (multi-conn, +24% vs curl single-stream) ───────────────
COPY hf-download /usr/local/bin/hf-download
RUN chmod +x /usr/local/bin/hf-download

WORKDIR /app

# ── PyTorch CUDA 12.4 ─────────────────────────────────────────────────────────
# torch 2.5.1: fixes infer_schema incompatibility with diffusers Wan autoencoder
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# ── Diffusers + HF stack ──────────────────────────────────────────────────────
# diffusers 0.33+ required for WanImageToVideoPipeline
RUN pip install --no-cache-dir \
    "diffusers==0.33.1" \
    "transformers>=4.45.0,<5.0" \
    "accelerate>=0.34.0,<2.0" \
    "huggingface-hub>=0.27.0,<2.0" \
    sentencepiece tokenizers \
    safetensors

# ── Server deps ───────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" "uvicorn[standard]>=0.30.0" \
    pillow "imageio>=2.34.0" imageio-ffmpeg \
    "numpy<2.0"

# ── App files ─────────────────────────────────────────────────────────────────
# Model (~28GB) is downloaded on first boot to the HF cache volume.
# GitHub Actions runners don't have enough disk for pre-baking.
# Mount -v wan-i2v-hf-cache:/root/.cache/huggingface to persist across restarts.
COPY server.py       /app/server.py
COPY start.sh        /app/start.sh
COPY idle_watchdog.py /app/idle_watchdog.py
RUN chmod +x /app/start.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/app/start.sh"]
