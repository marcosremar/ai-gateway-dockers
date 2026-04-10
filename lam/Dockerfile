# LAM — Large Avatar Model (SIGGRAPH 2025)
# One-shot animatable Gaussian head from a single photo → MP4
#
# Paper:  https://arxiv.org/abs/2502.17796
# Repo:   https://github.com/aigc3d/LAM
# Models: 3DAIGC/LAM-20K  +  3DAIGC/LAM-assets  (HuggingFace)
#
# VRAM:   ~24GB (RTX 3090 / A100 / L40S recommended)
# Speed:  ~1s reconstruction + ~10ms/frame render on A100
#
# Build:  docker build -t marcosremar/lam:latest .
# Run:    docker run --gpus all -p 8000:8000 marcosremar/lam:latest

# devel required: diff-gaussian-rasterization, nvdiffrast, simple-knn need nvcc
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_XET_HIGH_PERFORMANCE=1 \
    HF_XET_FIXED_DOWNLOAD_CONCURRENCY=50 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX" \
    FORCE_CUDA=1 \
    MODEL_ID=3DAIGC/LAM-20K \
    ASSETS_ID=3DAIGC/LAM-assets

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        git curl aria2 zstd \
        build-essential cmake ninja-build \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
        ffmpeg wget ca-certificates && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# ── hf-download helper ───────────────────────────────────────────────────────
COPY hf-download /usr/local/bin/hf-download
RUN chmod +x /usr/local/bin/hf-download

WORKDIR /app

# ── PyTorch 2.3 + xformers (CUDA 12.1) ──────────────────────────────────────
RUN pip install --no-cache-dir \
    torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir xformers==0.0.26.post1 \
    --index-url https://download.pytorch.org/whl/cu121

# ── pytorch3d — build from source (pre-built wheel URLs are unreliable) ──────
RUN pip install --no-cache-dir fvcore iopath && \
    pip install --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git"

# ── Clone LAM ────────────────────────────────────────────────────────────────
RUN git clone --depth 1 https://github.com/aigc3d/LAM.git /app/lam

# ── Python deps (pinned, no git+ entries) ────────────────────────────────────
RUN pip install --no-cache-dir \
    "pymcubes==0.1.6" \
    "omegaconf==2.3.0" \
    "moviepy==1.0.3" \
    "pandas==2.2.3" \
    "transformers==4.41.2" \
    "numpy==1.23.0" \
    "trimesh==4.4.9" \
    "opencv-python-headless==4.11.0.86" \
    "scikit-image==0.20.0" \
    "huggingface-hub==0.23.2" \
    "hf_xet>=1.4.0" \
    "safetensors" \
    "accelerate" \
    "tyro" \
    "einops" \
    "diffusers" \
    "plyfile" \
    "jaxtyping" \
    "typeguard==2.13.3" \
    "chumpy" \
    "loguru" \
    "ninja" \
    "Cython" \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.32.0" \
    "python-multipart" \
    "aiofiles"

# ── TensorFlow (for face-detection-tflite) ───────────────────────────────────
RUN pip install --no-cache-dir \
    "tensorflow-cpu==2.12.0" \
    "face-detection-tflite==0.6.0"

# ── JAX CPU (preprocessing only) ─────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "jax[cpu]" "jaxlib==0.4.30"

# ── CUDA extensions from source ──────────────────────────────────────────────
# diff-gaussian-rasterization (3DGS renderer)
RUN pip install --no-cache-dir \
    "git+https://github.com/ashawkey/diff-gaussian-rasterization/"

# nvdiffrast with backface culling
RUN pip install --no-cache-dir \
    "nvdiffrast @ git+https://github.com/ShenhanQian/nvdiffrast@backface-culling"

# simple-knn
RUN pip install --no-cache-dir \
    "git+https://github.com/camenduru/simple-knn/"

# ── Build FaceBoxesV2 C extension ─────────────────────────────────────────────
RUN cd /app/lam/external/landmark_detection/FaceBoxesV2/utils && bash make.sh

# ── Download + extract LAM assets (LAM_assets.tar + thirdparty_models.tar) ──
# Must be extracted inside /app/lam — tars contain pretrained_models/, ckpts/, etc.
RUN huggingface-cli download 3DAIGC/LAM-assets \
        LAM_assets.tar thirdparty_models.tar \
        --local-dir /app/lam/tmp && \
    cd /app/lam && \
    tar -xf tmp/LAM_assets.tar      && rm tmp/LAM_assets.tar && \
    tar -xf tmp/thirdparty_models.tar && rm tmp/thirdparty_models.tar && \
    rm -rf tmp

# ── Download LAM-20K model to the path LAM expects ──────────────────────────
RUN huggingface-cli download 3DAIGC/LAM-20K \
        --local-dir /app/lam/model_zoo/lam_models/releases/lam/lam-20k/step_045500/

# ── App files ────────────────────────────────────────────────────────────────
COPY server.py start.sh /app/
RUN chmod +x /app/start.sh

EXPOSE 8000

# First run downloads model (~2GB) → allow 3min before first health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["/app/start.sh"]
