# Generic finetune base image for ai-gateway gpu finetune.
# Works for ANY preset (text/audio/custom). Per-preset pipDeps installed at
# runtime from manifest.json. NO model-specific weights baked.
#
# Build:
#   docker build -t ghcr.io/<user>/aigw-finetune-base:latest \
#                -f dockers/aigw-finetune-base/Dockerfile \
#                dockers/aigw-finetune-base
#
# Use (auto-selected by ai-gateway when type=preset):
#   ai-gateway gpu finetune submit -f train.yaml
#
# Goal: cut 3-4 min boot per job vs apt-get + pip install on bare cuda image.
# Strategy:
#   1. Bake heavy stuff (torch + cuda runtime + common ML deps).
#   2. Skip model weights — pulled at runtime per-spec into HF_HOME volume.
#   3. /opt/cache mount → preserve HF + torch.compile cache between jobs.

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_XET_HIGH_PERFORMANCE=1 \
    HF_XET_FIXED_DOWNLOAD_CONCURRENCY=50 \
    HF_HOME=/opt/cache/hf \
    HUGGINGFACE_HUB_CACHE=/opt/cache/hf \
    TRANSFORMERS_CACHE=/opt/cache/hf \
    TORCHINDUCTOR_CACHE_DIR=/opt/cache/torch-inductor

# System deps — superset that covers text + audio presets.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        pkg-config build-essential \
        libsentencepiece-dev libsndfile1 ffmpeg \
        git curl ca-certificates rsync openssh-client \
    && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Torch first (large, separate layer for cache hits).
# CUDA 12.1 → torch 2.1.2 cu121 validated on RTX 4090 / 5090 / A100.
RUN pip install --upgrade pip && \
    pip install torch==2.1.2 torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cu121

# Numpy <2 (pocket-tts + many older audio libs incompat with numpy 2.x).
RUN pip install 'numpy<2'

# HF stack + xet for fast downloads.
RUN pip install 'huggingface-hub>=1.0.0' 'hf_xet>=1.4.0' hf_transfer

# Common ML deps shared across presets.
RUN pip install safetensors soundfile sentencepiece \
                accelerate datasets evaluate \
                wandb optuna \
                openai-whisper jiwer

# Pre-import to materialize .pyc + first-run side effects.
RUN python -c "import torch, torchaudio, safetensors, datasets, accelerate; print('[bake] base imports ok')"

# Inductor cache + HF cache live under /opt/cache (mountable as persistent volume).
RUN mkdir -p /opt/cache/hf /opt/cache/torch-inductor

WORKDIR /workspace

# Per-preset pipDeps installed at runtime by ai-gateway from manifest.json.
# Trainer scripts rsynced from preset bundle. Dataset/model pulled by job.
CMD ["bash"]
