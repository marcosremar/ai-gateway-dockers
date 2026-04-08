# ── DiT360 — 360° Panoramic Image Generation ─────────────────────────────────
# Single image for ALL GPU architectures: Blackwell (sm_120), Ada (sm_89),
# Ampere (sm_80/86), Hopper (sm_90).
#
# CUDA 12.8.1 runtime is the minimum for RTX 5090/5080 (sm_120).
#
# Model: FLUX.1-dev (12B) + DiT360 LoRA (Insta360-Research)
# Output: 1024×2048 equirectangular panoramic images from text prompts
# VRAM: ~37GB without offload, ~22GB with cpu_offload (slower)
# ─────────────────────────────────────────────────────────────────────────────

ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_ID=black-forest-labs/FLUX.1-dev \
    LORA_ID=Insta360-Research/DiT360-Panorama-Image-Generation \
    DEFAULT_STEPS=28 \
    DEFAULT_GUIDANCE=2.8 \
    DEFAULT_WIDTH=2048 \
    DEFAULT_HEIGHT=1024 \
    CPU_OFFLOAD=auto \
    HF_XET_HIGH_PERFORMANCE=1 \
    HF_XET_FIXED_DOWNLOAD_CONCURRENCY=50 \
    TORCHINDUCTOR_CACHE_DIR=/workspace/.torch-cache \
    TORCHINDUCTOR_FX_GRAPH_CACHE=1 \
    TORCHINDUCTOR_AUTOGRAD_CACHE=1

# System deps
# aria2 + zstd: multi-connection downloads (~30% faster than curl/single-stream)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    curl git aria2 zstd libxcb1 libgl1-mesa-glx libglib2.0-0 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    rm -rf /var/lib/apt/lists/*

# Install hf-download helper (aria2c-based, with HF Hub fallback)
COPY hf-download /usr/local/bin/hf-download
RUN chmod +x /usr/local/bin/hf-download

WORKDIR /app

# Step 1: PyTorch from CUDA 12.8 index
ARG CUDA_INDEX=https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir torch==2.11.0 torchvision --index-url ${CUDA_INDEX}

# Step 2: Diffusers (from git for latest FLUX support) + deps + Real-ESRGAN for 8K upscale
# Pin huggingface-hub >= 1.0 to get hf_xet as automatic dep (bench: +24% download speed)
RUN pip install --no-cache-dir \
    "diffusers[torch]>=0.32.0" \
    "transformers>=4.48.0" \
    "accelerate>=1.2.0" \
    "peft>=0.14.0" \
    "safetensors>=0.4.0" \
    sentencepiece \
    protobuf \
    "huggingface-hub>=1.0.0" \
    "hf_xet>=1.4.0" \
    uvicorn[standard] \
    fastapi \
    python-multipart \
    httpx \
    Pillow \
    realesrgan \
    "basicsr>=1.4.2" \
    opencv-python-headless

# Step 3: Pre-download LoRA weights (small, ~200MB) into HF cache.
# FLUX.1-dev base model (~24GB fp16) is too large for CI runners,
# so it downloads at first boot (runtime benefits from HF_XET_* env vars above).
# The LoRA is small enough to pre-bake.
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('Insta360-Research/DiT360-Panorama-Image-Generation', max_workers=8)" \
    || echo "LoRA pre-download skipped (will download at boot)"

COPY server.py .
COPY idle_watchdog.py .
COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8000

# Longer start period — FLUX.1-dev is ~24GB and may need to download on first boot
HEALTHCHECK --interval=15s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["/app/start.sh"]
