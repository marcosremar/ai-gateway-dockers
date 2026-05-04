# ── LatentSync — Audio-Conditioned Latent Diffusion Lip-Sync ──────────────
# Base: ByteDance LatentSync (https://github.com/bytedance/LatentSync)
# We add ai-gateway conventions on top of the upstream inference code:
#   - sshd (PUBKEY auth) so the gateway can patch/iterate the running pod
#   - FastAPI uvicorn :8000 with /health, /v1/lipsync (multipart in → MP4)
#   - idle_watchdog auto-stop (IDLE_TIMEOUT_MIN minutes of no requests)
#   - HF cache + weights persisted to /workspace when available
#
# Model:  ByteDance/LatentSync-1.5  (8GB VRAM)  or  LatentSync-1.6 (18GB)
# GPU:    RTX 3070+ for 1.5  |  RTX 4090 / A100 for 1.6 / 512x512 quality
# License: Apache 2.0 (LatentSync code) — see upstream LICENSE
#
# Build:
#   docker build -t marcosremar/latentsync:latest .
# Run:
#   docker run --gpus all -p 8000:8000 -p 22:22 \
#     -v /workspace/hf-cache:/workspace/.cache/huggingface \
#     -e LATENTSYNC_VARIANT=1.5 \
#     -e IDLE_TIMEOUT_MIN=15 \
#     marcosremar/latentsync:latest
# ─────────────────────────────────────────────────────────────────────────────

ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    IDLE_TIMEOUT_MIN=15 \
    LATENTSYNC_VARIANT=1.5 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# OS deps: ffmpeg (read/write video+audio), git+wget for clone+download,
# openssh-server for the gateway SSH-patch loop, build tools for any pip
# wheels that fall back to source.
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        git wget curl ca-certificates \
        ffmpeg libsm6 libxext6 libgl1 \
        openssh-server \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd \
    && sed -i 's/#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config \
    && sed -i 's/#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config

WORKDIR /app

# Clone upstream LatentSync. Pin to a known-good commit when promoting to
# production (replace `main` with a SHA below).
ARG LATENTSYNC_REF=main
RUN git clone --depth 1 --branch ${LATENTSYNC_REF} \
        https://github.com/bytedance/LatentSync.git /app/LatentSync

# PyTorch + the LatentSync requirements pinned by upstream. We let the
# upstream `requirements.txt` resolve transitively; the gateway only needs
# fastapi/uvicorn/python-multipart/hf_transfer on top for the HTTP shim.
RUN pip install --upgrade pip \
    && pip install \
         "torch==2.5.1+cu124" "torchvision==0.20.1+cu124" "torchaudio==2.5.1+cu124" \
         --index-url https://download.pytorch.org/whl/cu124 \
    && pip install -r /app/LatentSync/requirements.txt \
    && pip install \
         "fastapi>=0.115" "uvicorn[standard]>=0.32" python-multipart \
         hf_transfer huggingface_hub soundfile

# AI-gateway conventions (server, idle watchdog, start script).
COPY server.py            /app/server.py
COPY idle_watchdog.py     /app/idle_watchdog.py
COPY start.sh             /app/start.sh
RUN chmod +x /app/start.sh

# Persistent dirs (overlaid by `/workspace` if Vast/RunPod/Hyperstack mounts it)
RUN mkdir -p /app/models /app/data /app/results /workspace/.cache/huggingface

EXPOSE 8000 22

HEALTHCHECK --interval=30s --timeout=10s --start-period=240s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/app/start.sh"]
