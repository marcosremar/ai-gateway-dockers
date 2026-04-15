# ── SMPLest-X — Whole-Body SMPL-X Estimation with Hand Tracking ──────────────
# Expressive pose estimation: body + hands + face from a single image.
# Key advantage: full hand articulation (15 joints/hand) — critical for LIBRAS.
#
# Paper: "SMPLest-X: Ultimate Scaling for Expressive Human Pose and Shape"
# Repo:  github.com/MotrixLab/SMPLest-X (TPAMI 2025)
#
# Checkpoint: smplest_x_h.pth.tar (8.2GB) from HuggingFace waanqii/SMPLest-X
#
# Deploy via ai-gateway (never build locally):
#   bunx ai-gateway gpu deploy --image marcosremar/smplest-x:latest

ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3.10-venv \
    build-essential git wget curl \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── PyTorch ──────────────────────────────────────────────────────────────────
RUN pip install "numpy<2.0" "setuptools<70" wheel cython
RUN pip install torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu124

# ── OpenMMLab stack ──────────────────────────────────────────────────────────
RUN pip install openmim && \
    mim install mmengine "mmcv>=2.0.0" "mmdet>=3.0.0" "mmpose>=1.0.0"

# ── Clone SMPLest-X ─────────────────────────────────────────────────────────
RUN git clone --depth 1 https://github.com/MotrixLab/SMPLest-X.git /app/SMPLest-X

# ── Install SMPLest-X deps ──────────────────────────────────────────────────
WORKDIR /app/SMPLest-X
RUN pip install smplx scipy ultralytics

# ── Download checkpoint from HuggingFace (8.2GB) ────────────────────────────
RUN pip install huggingface_hub && \
    python3 -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download('waanqii/SMPLest-X', 'smplest_x_h32.pth.tar', \
    local_dir='/app/SMPLest-X/pretrained_models/smplest_x_h'); \
print('[smplest-x] Checkpoint downloaded')"

# ── SMPL-X body model placeholder (mount or bake) ───────────────────────────
RUN mkdir -p /app/SMPLest-X/human_models/human_model_files/smplx /app/SMPLest-X/human_models/human_model_files/smpl

# ── FastAPI server ───────────────────────────────────────────────────────────
RUN pip install fastapi "uvicorn[standard]" python-multipart Pillow opencv-python-headless

WORKDIR /app
COPY server.py /app/server.py
COPY idle_watchdog.py /app/idle_watchdog.py

ENV SMPLESTX_CONFIG=/app/SMPLest-X/main/config/config_smpler_x_h32.py \
    SMPLESTX_CHECKPOINT=/app/SMPLest-X/pretrained_models/smplest_x_h/smplest_x_h32.pth.tar

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python3", "/app/server.py"]
