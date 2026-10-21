# ── FLUX.1-dev — Open-Source Text-to-Image (ai-gateway) ───────────────────
# Wraps Black Forest Labs FLUX.1-dev (Apache 2.0). Exposes OpenAI-compatible
# /v1/images/generations endpoint so canal-dark / project-philosofi pipelines
# can route text-to-image identically to other gateway providers.
#
# Model:   black-forest-labs/FLUX.1-dev (~24GB FP16 weights, lazy-pulled)
# VRAM:    24GB+ (RTX 4090 24GB tight; A100 40GB confortável)
# License: Apache 2.0 (FLUX.1-dev — non-commercial OK; for commercial use
#          FLUX.1-schnell is fully Apache 2.0 — toggle FLUX_MODEL env)
#
# Build:
#   docker build -t marcosremar/flux:latest .
# Run:
#   docker run --gpus all -p 8000:8000 -p 22:22 \
#     -v /workspace/hf-cache:/workspace/.cache/huggingface \
#     -e IDLE_TIMEOUT_MIN=15 \
#     -e FLUX_MODEL=black-forest-labs/FLUX.1-schnell \
#     marcosremar/flux:latest
# ─────────────────────────────────────────────────────────────────────────────

ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    IDLE_TIMEOUT_MIN=15 \
    FLUX_MODEL=black-forest-labs/FLUX.1-schnell \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        git wget curl ca-certificates \
        ffmpeg libsm6 libxext6 libgl1 libsndfile1 \
        openssh-server \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd \
    && sed -i 's/#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config \
    && sed -i 's/#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config

WORKDIR /app

# PyTorch + diffusers stack. Pin to CUDA 12.4 wheels matching base.
# FLUX requires diffusers >= 0.30 (FluxPipeline) and transformers >= 4.45.
RUN pip install --upgrade pip \
    && pip install \
         "torch==2.5.1+cu124" "torchvision==0.20.1+cu124" \
         --index-url https://download.pytorch.org/whl/cu124 \
    && pip install \
         "diffusers>=0.31" "transformers>=4.45" "accelerate>=1.0" \
         "sentencepiece>=0.2" "protobuf>=4.25" \
         "fastapi>=0.115" "uvicorn[standard]>=0.32" python-multipart \
         hf_transfer huggingface_hub Pillow numpy einops

COPY server.py            /app/server.py
COPY idle_watchdog.py     /app/idle_watchdog.py
COPY start.sh             /app/start.sh
RUN chmod +x /app/start.sh

RUN mkdir -p /app/models /app/results /workspace/.cache/huggingface

EXPOSE 8000 22

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/app/start.sh"]
