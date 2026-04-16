# ── SMPLest-X — Whole-Body SMPL-X with Hand Tracking ─────────────────────────
# Full hand articulation (15 joints/hand) — critical for LIBRAS.
# Repo: github.com/MotrixLab/SMPLest-X (TPAMI 2025)
# Existing Docker ref: wcwcw/smplerx_inference:v0.2 (SMPLer-X, predecessor)
#
# Deploy: bunx ai-gateway gpu deploy --image marcosremar/smplest-x:latest

# Use CUDA 12.1 (matches hybrik-x, widely supported on Vast.ai hosts)
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3.10-venv \
    build-essential git wget curl \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    python3-opengl libosmesa6 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── PyTorch ──────────────────────────────────────────────────────────────────
RUN pip install "numpy<2.0" "setuptools<70" wheel cython
RUN pip install torch==2.1.2 torchvision==0.16.2 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# ── OpenMMLab stack ──────────────────────────────────────────────────────────
RUN pip install openmim && \
    mim install mmengine "mmcv>=2.0.0" "mmdet>=3.0.0" "mmpose>=1.0.0"

# ── Clone SMPLest-X ─────────────────────────────────────────────────────────
RUN git clone --depth 1 https://github.com/MotrixLab/SMPLest-X.git /app/SMPLest-X

# ── Install SMPLest-X deps ──────────────────────────────────────────────────
WORKDIR /app/SMPLest-X
RUN pip install -r requirements.txt 2>/dev/null || true
RUN pip install smplx scipy ultralytics pyopengl==3.1.4
ENV PYTHONPATH="/app/SMPLest-X:${PYTHONPATH}"

# ── Checkpoint placeholder (downloaded at startup via start.sh) ──────────────
RUN pip install huggingface_hub && \
    mkdir -p /app/SMPLest-X/pretrained_models/smplest_x_h

# ── SMPL-X body model placeholder ───────────────────────────────────────────
RUN mkdir -p /app/SMPLest-X/human_models/human_model_files/smplx \
             /app/SMPLest-X/human_models/human_model_files/smpl

# ── FastAPI server ───────────────────────────────────────────────────────────
RUN pip install fastapi "uvicorn[standard]" python-multipart Pillow opencv-python-headless

# ── Pin numpy LAST ───────────────────────────────────────────────────────────
RUN pip install "numpy<2.0" --force-reinstall --quiet

WORKDIR /app
COPY server.py /app/server.py
COPY idle_watchdog.py /app/idle_watchdog.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

ENV SMPLESTX_CONFIG=/app/SMPLest-X/configs/config_smplest_x_h.py \
    SMPLESTX_CHECKPOINT=/app/SMPLest-X/pretrained_models/smplest_x_h/smplest_x_h.pth.tar

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use python directly — Vast.ai doesn't always honor CMD/start.sh
# Checkpoint download is handled inside server.py load_model()
CMD ["python3", "/app/server.py"]
