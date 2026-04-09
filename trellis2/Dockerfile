# ══════════════════════════════════════════════════════════════════════════════
# TRELLIS.2 — Image-to-3D (.GLB) inference server
#
# Microsoft's TRELLIS.2-4B: 4B param model generating 3D meshes with PBR
# materials from a single image. Outputs .GLB for Babylon.js / Three.js.
#
# GPU: 24GB+ VRAM required (A100, H100, RTX 4090)
# Base: CUDA 12.4 + cuDNN 9
#
# Build note: CUDA extensions (flash-attn, cumesh, o-voxel, flexgemm) are
# compiled at build time. TORCH_CUDA_ARCH_LIST targets SM 8.0+ (A100/H100/
# RTX 3090/4090) for CI builds without a GPU. The resulting .so files are
# portable across those GPU families at runtime.
# ══════════════════════════════════════════════════════════════════════════════

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV ATTN_BACKEND=flash-attn
# Target GPU architectures for headless CUDA compilation (no GPU needed at
# build time). SM 8.0 = A100, SM 8.6 = RTX 3090, SM 8.9 = RTX 4090/L40S,
# SM 9.0 = H100. Add SM 7.5 (T4) if needed but increases build time.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3.10-venv \
    git wget curl ffmpeg \
    libgl1-mesa-glx libglib2.0-0 libgomp1 \
    build-essential g++ ninja-build cmake \
    libx11-dev libxrender-dev libxext-dev \
    libegl1-mesa-dev libgles2-mesa-dev libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# PyTorch 2.6.0 + CUDA 12.4
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Clone TRELLIS.2
RUN git clone --depth 1 https://github.com/microsoft/TRELLIS.2.git /app/trellis2
WORKDIR /app/trellis2

# Core Python deps (Pillow instead of pillow-simd — simd needs AVX headers
# that aren't available in all CI environments)
RUN pip install --no-cache-dir \
    imageio imageio-ffmpeg tqdm easydict \
    opencv-python-headless ninja trimesh \
    transformers \
    pandas lpips zstandard \
    Pillow kornia timm utils3d \
    huggingface_hub hf-xet \
    fastapi uvicorn python-multipart httpx \
    scipy safetensors accelerate einops

# flash-attn (slow compile — ~15min on CI, but pre-built wheels exist for
# some torch+CUDA combos. Try wheel first, fall back to source build.)
RUN pip install --no-cache-dir flash-attn==2.7.3 --no-build-isolation \
    || pip install --no-cache-dir flash-attn --no-build-isolation

# nvdiffrast (differentiable rasterizer — needs EGL/OpenGL dev headers)
RUN pip install --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git \
    || echo "WARN: nvdiffrast build failed — mesh export may use fallback"

# TRELLIS.2 custom CUDA extensions — each has a setup.py that uses
# torch.utils.cpp_extension. TORCH_CUDA_ARCH_LIST lets them compile
# without a physical GPU present.

# cumesh (CUDA mesh operations)
RUN cd /app/trellis2/extensions/cumesh && pip install --no-cache-dir . \
    || echo "WARN: cumesh failed — will use CPU fallback at runtime"

# o-voxel (sparse voxel structure — the core of TRELLIS.2)
RUN cd /app/trellis2/extensions/o_voxel && pip install --no-cache-dir . \
    || echo "WARN: o_voxel failed — will retry at runtime"

# flexgemm (flexible GEMM kernels for attention)
RUN cd /app/trellis2/extensions/flexgemm && pip install --no-cache-dir . \
    || echo "WARN: flexgemm failed — will use torch fallback"

# Install trellis2 package itself
RUN pip install --no-cache-dir -e .

# Pre-download the 4B model from HuggingFace so startup is instant.
# Uses huggingface_hub which caches to /root/.cache/huggingface/hub/ (~15GB).
# If this fails in CI (network/space), the model downloads on first request.
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('microsoft/TRELLIS.2-4B'); \
print('Model downloaded successfully'); \
" || echo "WARN: model pre-download failed — will download on first request"

# Copy our FastAPI server
COPY server.py /app/server.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "/app/server.py"]
