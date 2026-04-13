# ── WiLoR-mini API — MANO hand pose rotations as a service ───────────────────
# FastAPI + WebSocket server returning 15 MANO axis-angle rotations per hand,
# plus global_orient (wrist rotation) and betas (hand shape).
#
# Uses wilor_mini pip package (lightweight fork of WiLoR CVPR 2025).
# Auto-enables FP16 on CUDA for ~2x inference speedup.
#
# Build via GitHub Actions (ai-gateway-dockers workflow), never locally.
# Deploy via: bunx ai-gateway gpu deploy --image marcosremar/wilor-api:latest
# ────���─────────────────────────────────────────────────────────��──────────────

ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip python3.10-venv \
        build-essential git wget libgl1 libglib2.0-0 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps ──────────────────────────────────────────────────────────────
RUN pip install "numpy<2.0" "setuptools<70" wheel
RUN pip install --no-build-isolation 'git+https://github.com/mattloper/chumpy.git'
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
RUN pip install git+https://github.com/warmshao/WiLoR-mini.git
RUN pip install fastapi uvicorn pillow dill

# Pre-download/verify wilor_mini models at build time
RUN python3 -c "from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline; print('Import OK')" || true

# ── Copy API server ──────────────────────────��──────────────────────────────
COPY server_mini.py /app/server_mini.py

EXPOSE 8000

# Health check for ai-gateway readiness probe
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["python3", "/app/server_mini.py"]
