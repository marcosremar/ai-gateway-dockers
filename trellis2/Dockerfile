# ══════════════════════════════════════════════════════════════════════════════
# TRELLIS.2 — Image/Text-to-3D (.GLB) inference server
#
# Microsoft's TRELLIS.2-4B: 4 billion parameter model that generates high-
# quality 3D meshes with PBR materials (albedo, roughness, metallic) from a
# single image. Outputs .GLB files ready for Babylon.js / Three.js.
#
# GPU: 24GB+ VRAM required (tested on A100, H100, RTX 4090)
# Base: CUDA 12.4 + cuDNN 9 (matches TRELLIS.2 setup.sh defaults)
# ══════════════════════════════════════════════════════════════════════════════

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# TRELLIS.2 env vars
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV ATTN_BACKEND=flash-attn

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3.10-venv \
    git wget curl ffmpeg \
    libgl1-mesa-glx libglib2.0-0 libgomp1 \
    build-essential g++ ninja-build cmake \
    libx11-dev libxrender-dev libxext-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# PyTorch 2.6.0 + CUDA 12.4 (matches TRELLIS.2 setup.sh)
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Clone TRELLIS.2
RUN git clone --depth 1 https://github.com/microsoft/TRELLIS.2.git /app/trellis2
WORKDIR /app/trellis2

# Core Python deps (from requirements.txt + setup.sh)
RUN pip install --no-cache-dir \
    imageio imageio-ffmpeg tqdm easydict \
    opencv-python-headless ninja trimesh \
    transformers gradio==6.0.1 \
    tensorboard pandas lpips zstandard \
    pillow-simd kornia timm utils3d \
    huggingface_hub hf-xet \
    fastapi uvicorn python-multipart httpx \
    scipy safetensors accelerate einops

# flash-attn (requires CUDA + torch already installed)
RUN pip install --no-cache-dir flash-attn==2.7.3 --no-build-isolation

# nvdiffrast (differentiable rasterizer — needed for mesh export)
RUN pip install --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git

# nvdiffrec (differentiable renderer — needed for texture baking)
RUN pip install --no-cache-dir git+https://github.com/NVlabs/nvdiffrec.git

# Install TRELLIS.2 custom extensions
# cumesh (CUDA mesh operations)
RUN cd /app/trellis2/extensions/cumesh && pip install --no-cache-dir .

# o-voxel (sparse voxel structure)
RUN cd /app/trellis2/extensions/o_voxel && pip install --no-cache-dir .

# flexgemm (flexible GEMM kernels)
RUN cd /app/trellis2/extensions/flexgemm && pip install --no-cache-dir .

# Install trellis2 package itself
RUN pip install --no-cache-dir -e .

# Pre-download the 4B model from HuggingFace so startup is instant
# This caches ~15GB into /root/.cache/huggingface/hub/
RUN python -c "\
from trellis2.pipelines import Trellis2ImageTo3DPipeline; \
pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B'); \
print('Model downloaded successfully'); \
"

# Copy our FastAPI server
COPY server.py /app/server.py

EXPOSE 8000

# Health check — /health endpoint returns 200 when model is loaded
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "/app/server.py"]
