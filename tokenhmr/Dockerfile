# ── TokenHMR — Occlusion-Robust Human Mesh Recovery ──────────────────────────
# Tokenized pose representation for partial occlusion. CVPR 2024.
# Repo: github.com/saidwivedi/TokenHMR
#
# Deploy: bunx ai-gateway gpu deploy --image marcosremar/tokenhmr:latest

ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3.10-venv \
    build-essential git wget curl unzip \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── PyTorch (TokenHMR tested with 2.1.0+cu118, but 2.4+cu124 works) ─────────
RUN pip install "numpy<2.0" "setuptools<70" wheel
RUN pip install torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu124

# ── Clone TokenHMR ───────────────────────────────────────────────────────────
RUN git clone --depth 1 https://github.com/saidwivedi/TokenHMR.git /app/TokenHMR

# ── Install TokenHMR + deps ──────────────────────────────────────────────────
WORKDIR /app/TokenHMR
RUN pip install -r requirements.txt 2>/dev/null || true
RUN pip install -e . 2>/dev/null || true

# ── Detectron2 (for demo/inference) ──────────────────────────────────────────
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git' 2>/dev/null || true

# ── Core deps ────────────────────────────────────────────────────────────────
RUN pip install smplx scipy einops timm yacs

ENV PYTHONPATH="/app/TokenHMR:${PYTHONPATH}"

# ── Checkpoint + SMPL placeholder ────────────────────────────────────────────
RUN mkdir -p /app/checkpoints /app/body_models/smpl

# ── FastAPI server ───────────────────────────────────────────────────────────
RUN pip install fastapi "uvicorn[standard]" python-multipart Pillow opencv-python-headless

# ── Pin numpy LAST ───────────────────────────────────────────────────────────
RUN pip install "numpy<2.0" --force-reinstall --quiet

WORKDIR /app
COPY server.py /app/server.py
COPY idle_watchdog.py /app/idle_watchdog.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

ENV TOKENHMR_CHECKPOINT=/app/checkpoints/tokenhmr_model_latest.ckpt

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/app/start.sh"]
