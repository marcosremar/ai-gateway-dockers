# ── WiLoR — End-to-end 3D Hand Localization & Reconstruction ─────────────────
# Runs WiLoR (rolpotamias/WiLoR, CVPR 2025) in a CUDA 12.4 container with
# Python 3.10 and PyTorch 2.4. Tested target: NVIDIA RTX 4090 (sm_89).
#
# Source: https://github.com/rolpotamias/WiLoR
# Paper:  https://arxiv.org/abs/2409.12259
#
# What's pre-baked at build time:
#   - WiLoR repo (cloned with --recursive for submodules)
#   - Python deps from requirements.txt
#   - detector.pt and wilor_final.ckpt from the official HF Spaces mirror
#
# What you must mount at runtime (NOT in image — license/registration required):
#   - MANO_RIGHT.pkl  →  /app/mano_data/MANO_RIGHT.pkl
#     Get it from https://mano.is.tue.mpg.de/ (free registration)
#
# Build:
#   docker compose build
# Run:
#   docker compose up -d
#   open http://localhost:7860
# ─────────────────────────────────────────────────────────────────────────────

ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# ── System deps ──────────────────────────────────────────────────────────────
# python3.10: WiLoR pins this version
# build-essential + python3-dev: chumpy and pyrender need C extensions
# libgl1, libglib2.0-0: opencv-python runtime
# libosmesa6, libegl1: pyrender headless rendering
# git + git-lfs: clone WiLoR + handle large checkpoints
# wget: download pretrained weights
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        build-essential git git-lfs wget ca-certificates \
        libgl1 libglib2.0-0 libosmesa6 libegl1 libgles2 \
        libxrender1 libsm6 libxext6 ffmpeg && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install

# Make python3.10 the default `python` and `pip`
RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/local/bin/python3 && \
    python -m ensurepip --upgrade && \
    python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# ── Clone WiLoR (with submodules) ────────────────────────────────────────────
RUN git clone --recursive https://github.com/rolpotamias/WiLoR.git /app/WiLoR

WORKDIR /app/WiLoR

# ── PyTorch with CUDA 12.4 (RTX 4090 native) ────────────────────────────────
# WiLoR docs say cu117 + torch 2.0 but that's too old for RTX 4090 — use cu124.
# torch 2.4.x is the stable line that works with both cu124 and the WiLoR
# pytorch-lightning version range.
RUN pip install \
        torch==2.4.1+cu124 \
        torchvision==0.19.1+cu124 \
        --index-url https://download.pytorch.org/whl/cu124

# ── WiLoR Python deps ────────────────────────────────────────────────────────
# chumpy needs numpy installed first or its setup.py blows up
RUN pip install "numpy<2.0" "Cython>=0.29"
RUN pip install -r requirements.txt

# ── Pre-bake pretrained weights from the official HF Spaces mirror ───────────
# detector.pt        ≈ 11 MB  (YOLOv8 hand detector)
# wilor_final.ckpt   ≈ 700 MB (ViT + MANO regression head)
# model_config.yaml  ≈ 4 KB   (model architecture config)
RUN mkdir -p /app/WiLoR/pretrained_models && \
    wget -q https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt \
        -O /app/WiLoR/pretrained_models/detector.pt && \
    wget -q https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt \
        -O /app/WiLoR/pretrained_models/wilor_final.ckpt && \
    wget -q https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/model_config.yaml \
        -O /app/WiLoR/pretrained_models/model_config.yaml

# Sanity check — fail the build if any weight is missing or empty
RUN test -s /app/WiLoR/pretrained_models/detector.pt && \
    test -s /app/WiLoR/pretrained_models/wilor_final.ckpt && \
    test -s /app/WiLoR/pretrained_models/model_config.yaml

# ── MANO_RIGHT.pkl placeholder ───────────────────────────────────────────────
# Cannot ship inside the image (license restriction). Mount via volume instead.
RUN mkdir -p /app/WiLoR/mano_data

EXPOSE 7860

# Default to the gradio web UI on :7860. Override CMD to run the CLI demo:
#   docker run ... python demo.py --img_folder demo_img --out_folder out --fast --save_mesh
CMD ["python", "gradio_demo.py"]
