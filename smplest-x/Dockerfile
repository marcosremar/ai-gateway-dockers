# ── SMPLest-X — Whole-Body SMPL-X with Hand Tracking ─────────────────────────
# Full hand articulation (15 joints/hand) — critical for LIBRAS.
# Repo: github.com/MotrixLab/SMPLest-X (TPAMI 2025)
#
# GPU: RTX 3090 / A4000+ (24GB+). ~8GB VRAM for inference.
# Deploy: bunx ai-gateway gpu deploy --image marcosremar/smplest-x:latest

# CUDA 12.1 — matches hybrik-x, widely supported on Vast.ai
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3.10-venv \
    build-essential git wget curl unzip \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    python3-opengl libosmesa6 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── PyTorch 2.1.2 + CUDA 12.1 (same as hybrik-x) ───────────────────────────
RUN pip install "numpy<2.0" "setuptools<70" wheel cython
RUN pip install torch==2.1.2 torchvision==0.16.2 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# ── OpenMMLab — PRE-BUILT WHEELS (no compilation!) ──────────────────────────
RUN pip install openmim && \
    pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html && \
    mim install mmengine "mmdet>=3.0.0" "mmpose>=1.0.0"

# ── Clone SMPLest-X ─────────────────────────────────────────────────────────
RUN git clone --depth 1 https://github.com/MotrixLab/SMPLest-X.git /app/SMPLest-X

# ── Install SMPLest-X deps ──────────────────────────────────────────────────
WORKDIR /app/SMPLest-X
RUN pip install -r requirements.txt 2>/dev/null || true
RUN pip install smplx scipy ultralytics pyopengl==3.1.4 huggingface_hub

# ── Download ALL models at build time (body models + checkpoint + YOLO) ─────
# Script avoids inline `from` which Docker misparses as FROM instruction.
COPY download_models.py /tmp/download_models.py
RUN python3 /tmp/download_models.py && \
    rm /tmp/download_models.py && \
    echo "=== Body model files ===" && \
    find human_models -type f -exec ls -lh {} \; && \
    echo "=== Checkpoint ===" && \
    ls -lh pretrained_models/smplest_x_h/

# ── FastAPI server deps ─────────────────────────────────────────────────────
RUN pip install fastapi "uvicorn[standard]" python-multipart Pillow opencv-python-headless httpx

# ── Pin numpy LAST (avoid scipy/mmcv incompatibility) ───────────────────────
RUN pip install "numpy<2.0" --force-reinstall --quiet

# ── Copy server files ───────────────────────────────────────────────────────
WORKDIR /app
COPY server.py /app/server.py
COPY idle_watchdog.py /app/idle_watchdog.py

ENV PYTHONPATH="/app/SMPLest-X:${PYTHONPATH}" \
    SMPLESTX_ROOT="/app/SMPLest-X"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python3", "/app/server.py"]
