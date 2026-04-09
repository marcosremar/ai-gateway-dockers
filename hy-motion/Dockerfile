# HY-Motion-1.0 — Tencent Hunyuan text-to-3D-motion model wrapped in a FastAPI
# server, packaged for ai-gateway Vast.ai deployment.
#
# Targets RTX 5090 (Blackwell, sm_120) with CUDA 12.8 + PyTorch 2.7. Older
# CUDA/PyTorch versions don't have Blackwell kernels, so do NOT downgrade
# without also dropping the GPU target back to Ada/Hopper.
#
# Image is ~50GB extracted (Qwen3-8B encoder is 16GB by itself). Use a
# Vast.ai disk allocation of >=120GB to leave room for the layered FS plus
# the runtime output dir.
ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1
# Lazy weight download — the server's lifespan startup will fetch models from
# HuggingFace on first boot and cache them on the Vast.ai instance disk. This
# keeps the image at ~10 GB so it fits on GitHub Actions free runners. The
# tradeoff is a slow first /health check (~10-15 min) while ~22 GB of weights
# download. Subsequent boots on the same instance hit the cache.
ENV USE_HF_MODELS=1
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/hub

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    git git-lfs wget curl ffmpeg libgl1-mesa-glx libglib2.0-0 unzip \
    build-essential g++ \
    && rm -rf /var/lib/apt/lists/* && git lfs install

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# PyTorch 2.7 + CUDA 12.8 — first version with full RTX 5090 (sm_120) support.
RUN pip install --no-cache-dir \
    torch==2.7.0 torchvision==0.22.0 \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Clone the official HY-Motion repo (pinned to main; bump if upstream releases
# a new build that breaks this image).
RUN git clone https://github.com/Tencent-Hunyuan/HY-Motion-1.0.git /app/HY-Motion-1.0
WORKDIR /app/HY-Motion-1.0

# HY-Motion deps. We deviate from the upstream requirements.txt:
#   - bitsandbytes is in the file but never imported in the code; skipping it
#     avoids a CUDA-only wheel that breaks builds for non-x86 hosts.
#   - fbxsdkpy is the proprietary Autodesk SDK and is wrapped in try/except in
#     t2m_runtime.py. We skip it for now and emit BVH/GLTF instead of FBX.
#   - torchdiffeq, accelerate, diffusers, transformers, einops, safetensors,
#     hf_hub, omegaconf, transforms3d, scipy come from the upstream pin so the
#     model loader code matches what HY-Motion expects.
RUN pip install --no-cache-dir \
    --extra-index-url https://gitlab.inria.fr/api/v4/projects/18692/packages/pypi/simple \
    huggingface_hub==0.30.0 hf_transfer hf_xet \
    torchdiffeq==0.2.5 \
    accelerate==0.30.1 \
    diffusers==0.26.3 \
    transformers==4.53.3 \
    einops==0.8.1 \
    safetensors==0.5.3 \
    "numpy>=1.24.0,<2.0" \
    "scipy>=1.10.0" \
    transforms3d==0.4.2 \
    PyYAML==6.0 omegaconf==2.3.0 click==8.1.3 requests==2.32.4 \
    openai==1.78.1 \
    fbxsdkpy==2020.1.post2 \
    fastapi uvicorn[standard] python-multipart pydantic
# openai is required at module-import time by hymotion/prompt_engineering/
# prompt_rewrite.py, even when disable_prompt_engineering=True at runtime.
# hf_xet enables Xet storage for faster HuggingFace downloads.
# fbxsdkpy is the Autodesk FBX SDK Python bindings hosted on INRIA's GitLab
# package index (the only public source of pre-built wheels). It enables
# T2MRuntime's `output_format="fbx"` path.

# NOTE: weights are NOT pre-baked. On first boot the server will download:
#   - tencent/HY-Motion-1.0          (~4 GB DiT generator)
#   - openai/clip-vit-large-patch14  (~1.7 GB sentence encoder)
#   - Qwen/Qwen3-8B                  (~16 GB LLM text encoder)
# Total: ~22 GB cached under HF_HOME on the Vast.ai instance disk.
#
# Text2MotionPrompter (prompt rewriter / duration estimator) is intentionally
# skipped — the server runs with disable_prompt_engineering=True so callers
# pass pre-formatted English prompts.

# Retargeting: SMPL-H → Mixamo via retarget_fbx.py (from hy-motion-fbx-exporter)
# + Mixamo character FBX template for target skeleton
COPY retarget_fbx.py /app/retarget_fbx.py
COPY mixamo_template.fbx /app/mixamo_template.fbx

# FBX2glTF for converting retargeted FBX → GLB (Babylon.js native format)
# The zip contains FBX2glTF-linux-x86_64/FBX2glTF-linux-x86_64 (binary in subdir)
RUN wget -q https://github.com/godotengine/FBX2glTF/releases/download/v0.13.1/FBX2glTF-linux-x86_64.zip -O /tmp/fbx2gltf.zip && \
    cd /tmp && python3 -c "import zipfile; zipfile.ZipFile('fbx2gltf.zip').extractall('.')" && \
    mv /tmp/FBX2glTF-linux-x86_64/FBX2glTF-linux-x86_64 /usr/local/bin/FBX2glTF && \
    chmod +x /usr/local/bin/FBX2glTF && \
    rm -rf /tmp/fbx2gltf.zip /tmp/FBX2glTF-linux-x86_64

# FastAPI server + entrypoint
COPY server.py /app/server.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8000

CMD ["/app/start.sh"]
