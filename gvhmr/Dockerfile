# ── GVHMR — World-Grounded Video Human Motion Recovery ───────────────────────
# Drift-free trajectory via gravity-view coordinates. SIGGRAPH Asia 2024.
# Repo: github.com/zju3dv/GVHMR
#
# Deploy: bunx ai-gateway gpu deploy --image marcosremar/gvhmr:latest

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

# ── PyTorch (matching GVHMR's requirements: 2.3.0+cu121, but 2.4+cu124 works) ─
RUN pip install "numpy==1.23.5" "setuptools>=68.0" wheel cython
RUN pip install torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu124

# ── Clone GVHMR ──────────────────────────────────────────────────────────────
RUN git clone --depth 1 https://github.com/zju3dv/GVHMR.git /app/GVHMR

# ── GVHMR requirements (from requirements.txt) ──────────────────────────────
WORKDIR /app/GVHMR
RUN pip install \
    "timm==0.9.12" "lightning==2.3.0" \
    hydra-core==1.3 hydra-zen hydra_colorlog rich \
    matplotlib opencv-python ffmpeg-python scikit-image \
    termcolor einops "imageio==2.34.1" "av==13.0.0" joblib \
    trimesh chumpy smplx ultralytics cython_bbox lapx \
    tensorboardX gdown

# ── pytorch3d (pre-built wheel for py310+cu121+pt2.3) ───────────────────────
RUN pip install pytorch3d 2>/dev/null || \
    pip install "git+https://github.com/facebookresearch/pytorch3d.git" 2>/dev/null || \
    echo "[gvhmr] pytorch3d install skipped"

# ── Install GVHMR package ───────────────────────────────────────────────────
RUN pip install -e . 2>/dev/null || true
ENV PYTHONPATH="/app/GVHMR:${PYTHONPATH}"

# ── Checkpoint dirs ──────────────────────────────────────────────────────────
RUN mkdir -p inputs/checkpoints/gvhmr inputs/checkpoints/hmr2 \
             inputs/checkpoints/vitpose inputs/checkpoints/yolo \
             inputs/checkpoints/dpvo inputs/checkpoints/body_models/smplx \
             inputs/checkpoints/body_models/smpl

# ── FastAPI server ───────────────────────────────────────────────────────────
RUN pip install fastapi "uvicorn[standard]" python-multipart Pillow scipy

# ── Pin numpy LAST ───────────────────────────────────────────────────────────
RUN pip install "numpy==1.23.5" --force-reinstall --quiet

WORKDIR /app
COPY server.py /app/server.py
COPY idle_watchdog.py /app/idle_watchdog.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

ENV GVHMR_CHECKPOINT=/app/GVHMR/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/app/start.sh"]
