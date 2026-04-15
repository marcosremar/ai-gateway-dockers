# ── GVHMR — World-Grounded Video Human Motion Recovery ───────────────────────
# Drift-free global trajectory estimation from monocular video.
# Uses gravity-view coordinates for world-grounded motion recovery.
#
# Paper: "GVHMR: World-Grounded Human Motion Recovery via Gravity-View Coordinates" (SIGGRAPH Asia 2024)
# Repo:  github.com/zju3dv/GVHMR
#
# Checkpoints: gvhmr_siga24_release.ckpt + dpvo + hmr2 + vitpose + yolo
# Google Drive: https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD
#
# Deploy via ai-gateway (never build locally):
#   bunx ai-gateway gpu deploy --image marcosremar/gvhmr:latest

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

# ── Clone GVHMR ──────────────────────────────────────────────────────────────
RUN git clone --depth 1 https://github.com/zju3dv/GVHMR.git /app/GVHMR
WORKDIR /app/GVHMR
RUN pip install -e . 2>/dev/null || true

# ── Additional deps ──────────────────────────────────────────────────────────
RUN pip install smplx scipy einops timm ultralytics

# ── Download checkpoints via gdown ───────────────────────────────────────────
RUN pip install gdown && \
    mkdir -p inputs/checkpoints/gvhmr inputs/checkpoints/hmr2 \
             inputs/checkpoints/vitpose inputs/checkpoints/yolo \
             inputs/checkpoints/dpvo inputs/checkpoints/body_models && \
    gdown --folder "1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD" -O inputs/checkpoints/ 2>/dev/null || \
    echo "[gvhmr] WARNING: gdown folder failed — checkpoints must be mounted at runtime"

# ── SMPL body model placeholder ─────────────────────────────────────────────
RUN mkdir -p inputs/checkpoints/body_models/smplx inputs/checkpoints/body_models/smpl

# ── FastAPI server ───────────────────────────────────────────────────────────
RUN pip install fastapi "uvicorn[standard]" python-multipart Pillow opencv-python-headless

WORKDIR /app
COPY server.py /app/server.py
COPY idle_watchdog.py /app/idle_watchdog.py

ENV GVHMR_CHECKPOINT=/app/GVHMR/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python3", "/app/server.py"]
