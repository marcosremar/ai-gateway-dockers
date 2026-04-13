# ── WiLoR-mini API — MANO hand pose rotations as a service ───────────────────
# FastAPI + WebSocket server returning 15 MANO axis-angle rotations per hand,
# plus global_orient (wrist rotation) and betas (hand shape).
#
# Uses wilor_mini pip package (lightweight fork of WiLoR CVPR 2025).
# Auto-enables FP16 on CUDA for ~2x inference speedup.
#
# Build via GitHub Actions (ai-gateway-dockers workflow), never locally.
# Deploy via: bunx ai-gateway gpu deploy
# ─────────────────────────────────────────────────────────────────────────────

ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3.10-venv \
        build-essential git wget curl ca-certificates libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Make python3.10 the default and install pip
RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/local/bin/python3 && \
    wget -qO /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py && \
    python3.10 /tmp/get-pip.py && \
    rm /tmp/get-pip.py

WORKDIR /app

# ── Python deps ──────────────────────────────────────────────────────────────
RUN pip install "numpy<2.0" "setuptools<70" wheel
RUN pip install --no-build-isolation 'git+https://github.com/mattloper/chumpy.git'
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-build-isolation 'git+https://github.com/warmshao/WiLoR-mini.git'
RUN pip install fastapi uvicorn pillow dill

# Pre-download models at build time so startup is instant (no HuggingFace download)
RUN python3 -c "\
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline; \
import torch; \
device = torch.device('cpu'); \
p = WiLorHandPose3dEstimationPipeline(device=device, dtype=torch.float32); \
print('WiLoR-mini models pre-downloaded OK'); \
"

# ── Copy API server + startup script ─────────────────────────────────────────
COPY server_mini.py /app/server_mini.py

# Vast.ai uses /app/start.sh instead of Docker CMD. Create it so the server
# starts automatically on any platform (Vast.ai, RunPod, plain Docker).
RUN printf '#!/bin/bash\nexec python3 /app/server_mini.py\n' > /app/start.sh && \
    chmod +x /app/start.sh

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=5s --start-period=120s \
    CMD curl -sf http://localhost:8080/health || exit 1

CMD ["python3", "/app/server_mini.py"]
