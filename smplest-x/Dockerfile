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

# ── Download SMPL-X/SMPL/MANO body models from HuggingFace ─────────────────
# These are REQUIRED for model initialization. Without them, load_model fails.
# Source: lithiumice/models_hub (mirrors of official SMPL-X body models)
RUN python3 -c "
from huggingface_hub import hf_hub_download
import shutil, os

# SMPL-X body models (.npz)
os.makedirs('human_models/human_model_files/smplx', exist_ok=True)
for f in ['SMPLX_NEUTRAL.npz', 'SMPLX_MALE.npz', 'SMPLX_FEMALE.npz']:
    path = hf_hub_download('lithiumice/models_hub', f'4_SMPLhub/SMPLX/X_npz/{f}')
    shutil.copy(path, f'human_models/human_model_files/smplx/{f}')
    print(f'  OK: smplx/{f}')

# SMPL body models (.pkl)
os.makedirs('human_models/human_model_files/smpl', exist_ok=True)
for f in ['SMPL_NEUTRAL.pkl', 'SMPL_MALE.pkl', 'SMPL_FEMALE.pkl']:
    path = hf_hub_download('lithiumice/models_hub', f'4_SMPLhub/SMPL/X_pkl/{f}')
    shutil.copy(path, f'human_models/human_model_files/smpl/{f}')
    print(f'  OK: smpl/{f}')

# MANO hand models (.pkl) — needed for hand joint mapping
os.makedirs('human_models/human_model_files/mano', exist_ok=True)
for f in ['MANO_LEFT.pkl', 'MANO_RIGHT.pkl']:
    path = hf_hub_download('lithiumice/models_hub', f'4_SMPLhub/MANO/pkl/{f}')
    shutil.copy(path, f'human_models/human_model_files/mano/{f}')
    print(f'  OK: mano/{f}')

print('All body models downloaded!')
" && \
    echo "Body model files:" && \
    find human_models -type f -exec ls -lh {} \;

# ── Download SMPLest-X checkpoint + config (8.2GB) ──────────────────────────
# Baked into image like HybrIK does — no runtime download needed.
RUN mkdir -p pretrained_models/smplest_x_h && \
    python3 -c "
from huggingface_hub import hf_hub_download
import shutil
for f in ['smplest_x_h.pth.tar', 'config_base.py']:
    path = hf_hub_download('waanqii/SMPLest-X', f)
    shutil.copy(path, f'pretrained_models/smplest_x_h/{f}')
    print(f'  OK: {f}')
print('Checkpoint + config ready!')
"

# ── Pre-download YOLOv8x detector ───────────────────────────────────────────
RUN python3 -c "
from ultralytics import YOLO
import shutil, os
model = YOLO('yolov8x.pt')
os.makedirs('pretrained_models', exist_ok=True)
if os.path.exists('yolov8x.pt'):
    shutil.move('yolov8x.pt', 'pretrained_models/yolov8x.pt')
print('YOLOv8x downloaded!')
" 2>/dev/null || echo "YOLO download skipped (will auto-download at runtime)"

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
