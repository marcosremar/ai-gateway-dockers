# ── HaMeR — Reconstructing Hands in 3D with Transformers ────────────────────
# Runs HaMeR (geopavlakos/hamer, ICCV 2023) in a CUDA 12.4 container with
# Python 3.10 and PyTorch 2.4. Tested target: NVIDIA RTX 4090 (sm_89).
#
# Source: https://github.com/geopavlakos/hamer
# Paper:  https://arxiv.org/abs/2312.05251
# Page:   https://geopavlakos.github.io/hamer/
#
# Compared to the official Dockerfile (cuda 12.6 + cu118 wheels), this image:
#   - Uses a stable CUDA 12.4 base (better PyTorch wheel compat)
#   - Pre-bakes the demo data tarball (~6 GB) at build time, so the first
#     `docker run` is instant — no `bash fetch_demo_data.sh` step
#   - Uses the direct UT Austin mirror (wget) instead of gdown to avoid the
#     "this file is too large for Google Drive virus scan" interactive auth
#   - Pre-installs ViTPose third-party dep so `import vitpose_model` works
#
# What's pre-baked:
#   - HaMeR repo with submodules (--recursive)
#   - PyTorch 2.4 + cu124 wheels
#   - All Python deps from `pip install -e .[all]`
#   - hamer_demo_data.tar.gz (~6 GB extracted to _DATA/)
#   - ViTPose model from third-party submodule
#
# What you must mount at runtime (NOT in image — license/registration required):
#   - MANO_RIGHT.pkl  →  /app/hamer/_DATA/data/mano/MANO_RIGHT.pkl
#     Get it from https://mano.is.tue.mpg.de/ (free registration)
#
# Build:
#   docker compose build           # ~10-15 min, ~10 GB image (6 GB is demo data)
# Run CLI demo (default):
#   mkdir -p input output mano_data
#   cp /path/to/MANO_RIGHT.pkl mano_data/
#   cp your_photo.jpg input/
#   docker compose run --rm hamer
#   # results in ./output/
# Or interactive shell:
#   docker compose run --rm --entrypoint /bin/bash hamer
# ─────────────────────────────────────────────────────────────────────────────

ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYOPENGL_PLATFORM=egl

# ── System deps ──────────────────────────────────────────────────────────────
# python3.10: HaMeR is tested on this version
# build-essential + python3-dev: detectron2, chumpy, mmcv all need C/C++ extensions
# libgl1, libglib2.0-0, libsm6, libxext6: opencv-python runtime
# libosmesa6, libegl1, libgles2-mesa, libglfw3: pyrender headless rendering
# ffmpeg: video I/O
# git + git-lfs: clone HaMeR + submodules + handle large tracked files
# wget + ca-certificates: download the demo data tarball
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        build-essential gcc g++ make git git-lfs wget ca-certificates \
        libgl1 libglib2.0-0 libosmesa6 libegl1 libgles2-mesa libglfw3 \
        libxrender1 libsm6 libxext6 ffmpeg libsndfile1-dev espeak-ng && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install

# Make python3.10 the default `python` and `pip`.
# Cannot use `ensurepip` — Debian/Ubuntu disables it for system Python.
# Use the official get-pip.py bootstrap instead, which works on every Python.
RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/local/bin/python3 && \
    wget -qO /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py && \
    python3.10 /tmp/get-pip.py && \
    rm /tmp/get-pip.py && \
    python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# ── Clone HaMeR (with submodules: third-party/ViTPose etc.) ─────────────────
RUN git clone --recursive https://github.com/geopavlakos/hamer.git /app/hamer

WORKDIR /app/hamer

# ── PyTorch with CUDA 12.4 (RTX 4090 native) ────────────────────────────────
# HaMeR's official Dockerfile uses torch 2.2 + cu118; we bump to 2.4 + cu124
# for native RTX 4090 (sm_89) support and broader wheel availability.
RUN pip install \
        torch==2.4.1+cu124 \
        torchvision==0.19.1+cu124 \
        --index-url https://download.pytorch.org/whl/cu124

# ── Numpy pin (chumpy + mmcv 1.3.9 break with numpy 2.x) ────────────────────
RUN pip install "numpy<2.0" "Cython>=0.29"

# ── HaMeR + all extras (installs detectron2, chumpy, mmcv, hydra, etc.) ─────
# `pip install -e .[all]` runs the package's own setup.py which pulls
# detectron2 + chumpy from GitHub and installs every needed dependency.
RUN pip install -e ".[all]"

# ── Install ViTPose submodule in editable mode ──────────────────────────────
# HaMeR uses ViTPose for 2D pose estimation. The submodule is at
# vitpose_model/ when you clone with --recursive. Some HaMeR forks place it
# at third-party/ViTPose — try both.
RUN if [ -d /app/hamer/vitpose_model ]; then \
        pip install -e /app/hamer/vitpose_model; \
    elif [ -d /app/hamer/third-party/ViTPose ]; then \
        pip install -e /app/hamer/third-party/ViTPose; \
    fi

# ── Pre-bake the demo data (~6 GB) ──────────────────────────────────────────
# The tarball includes pretrained checkpoints, ViTPose weights, demo images,
# and config files. Without this step the first `docker run` would need to
# download 6 GB before producing any output.
RUN wget --progress=dot:giga \
        https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz \
        -O /tmp/hamer_demo_data.tar.gz && \
    tar --warning=no-unknown-keyword --exclude=".*" \
        -xf /tmp/hamer_demo_data.tar.gz -C /app/hamer && \
    rm /tmp/hamer_demo_data.tar.gz

# Sanity check — fail the build if expected artifacts are missing
RUN test -d /app/hamer/_DATA && \
    find /app/hamer/_DATA -name "*.ckpt" -o -name "*.pth" | head -1 | grep -q . || \
    (echo "ERROR: no checkpoint files found in _DATA after extraction" && exit 1)

# ── MANO_RIGHT.pkl placeholder ───────────────────────────────────────────────
# Cannot ship inside the image (license restriction). Mount via volume instead.
RUN mkdir -p /app/hamer/_DATA/data/mano

# Default working dirs for input/output (will be replaced by volume mounts)
RUN mkdir -p /app/hamer/example_data /app/hamer/demo_out

# Default CMD: run the canonical demo against /app/hamer/example_data
# Override at run time:
#   docker compose run --rm --entrypoint /bin/bash hamer
CMD ["python", "demo.py", \
     "--img_folder", "example_data", \
     "--out_folder", "demo_out", \
     "--batch_size=48", \
     "--side_view", \
     "--save_mesh", \
     "--full_frame"]
