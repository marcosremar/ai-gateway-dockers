# ── WHAM — World-grounded Humans with Accurate Motion ────────────────────────
# Temporal video-based pose estimation with SLAM integration.
# 57mm MPJPE on 3DPW — strong temporal coherence, less jitter than per-frame.
#
# Paper: "WHAM: Reconstructing World-grounded Humans with Accurate 3D Motion" (CVPR 2024)
# Repo:  github.com/yohanshin/WHAM
#
# Checkpoint: wham_vit_w_3dpw.pth.tar from Google Drive
#
# Deploy via ai-gateway (never build locally):
#   bunx ai-gateway gpu deploy --image marcosremar/wham:latest

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

# ── Clone WHAM ───────────────────────────────────────────────────────────────
RUN git clone --depth 1 https://github.com/yohanshin/WHAM.git /app/WHAM
WORKDIR /app/WHAM
RUN pip install -e . 2>/dev/null || pip install -r requirements.txt 2>/dev/null || true

# ── Additional deps ──────────────────────────────────────────────────────────
RUN pip install smplx scipy joblib plyfile yacs einops

# ── Checkpoint deps (downloaded at startup via start.sh) ─────────────────────
RUN pip install gdown && mkdir -p /app/WHAM/checkpoints

# ── SMPL body model placeholder ─────────────────────────────────────────────
RUN mkdir -p /app/body_models/smpl

# ── FastAPI server ───────────────────────────────────────────────────────────
RUN pip install fastapi "uvicorn[standard]" python-multipart Pillow opencv-python-headless

WORKDIR /app
COPY server.py /app/server.py
COPY idle_watchdog.py /app/idle_watchdog.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

ENV WHAM_CONFIG=/app/WHAM/configs/yamls/demo.yaml \
    WHAM_CHECKPOINT=/app/WHAM/checkpoints/wham_vit_w_3dpw.pth.tar

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/app/start.sh"]
