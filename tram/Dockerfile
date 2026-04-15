# ── TRAM — Tracking and Reconstruction of Arbitrary 3D Humans ────────────────
# Global trajectory recovery with multi-person tracking.
# Pipeline: Detectron2 + SAM + DEVA + DROID-SLAM + HMR2
#
# Paper: "TRAM: Global Trajectory and Motion of 3D Humans from in-the-Wild Videos" (2024)
# Repo:  github.com/yufu-wang/tram
#
# Checkpoints: downloaded via scripts/download_models.sh
#
# Deploy via ai-gateway (never build locally):
#   bunx ai-gateway gpu deploy --image marcosremar/tram:latest

ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3.10-venv \
    build-essential git wget curl ffmpeg \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── PyTorch ──────────────────────────────────────────────────────────────────
RUN pip install "numpy<2.0" "setuptools<70" wheel
RUN pip install torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu124

# ── Detectron2 (required by TRAM) ───────────────────────────────────────────
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# ── Clone TRAM with submodules ───────────────────────────────────────────────
RUN git clone --recursive --depth 1 https://github.com/yufu-wang/tram.git /app/TRAM
WORKDIR /app/TRAM

# ── Build DROID-SLAM ─────────────────────────────────────────────────────────
RUN cd thirdparty/DROID-SLAM && python3 setup.py install 2>/dev/null || \
    echo "[tram] DROID-SLAM build may need manual intervention"

# ── Install TRAM deps ────────────────────────────────────────────────────────
RUN pip install -e . 2>/dev/null || pip install -r requirements.txt 2>/dev/null || true
RUN pip install smplx scipy einops timm segment-anything

# ── Download pretrained models ───────────────────────────────────────────────
RUN bash scripts/download_models.sh 2>/dev/null || \
    echo "[tram] download_models.sh may need SMPL credentials — mount checkpoints at runtime"

# ── SMPL body model placeholder ─────────────────────────────────────────────
RUN mkdir -p /app/body_models/smpl

# ── FastAPI server ───────────────────────────────────────────────────────────
RUN pip install fastapi "uvicorn[standard]" python-multipart Pillow opencv-python-headless

WORKDIR /app
COPY server.py /app/server.py
COPY idle_watchdog.py /app/idle_watchdog.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

ENV TRAM_CHECKPOINT=/app/TRAM/data/tram_ckpt.pth

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/app/start.sh"]
