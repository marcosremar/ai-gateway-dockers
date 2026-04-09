# ══════════════════════════════════════════════════════════════════════════════
# TRELLIS.2 — Image-to-3D (.GLB) inference server
#
# Microsoft's TRELLIS.2-4B generates 3D meshes with PBR materials from a
# single image. Outputs .GLB for Babylon.js / Three.js.
#
# GPU: 24GB+ VRAM (A100, H100, RTX 4090)
# ══════════════════════════════════════════════════════════════════════════════

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV ATTN_BACKEND=flash-attn
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# System deps (EGL for nvdiffrast headless, eigen headers cloned by o-voxel,
# openssh-server for Vast.ai SSH tunnel access — Vast hosts forward all
# inbound traffic to the container's sshd, which we run alongside FastAPI)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    git wget curl ffmpeg \
    libgl1-mesa-glx libglib2.0-0 libgomp1 \
    build-essential g++ ninja-build cmake \
    libegl1-mesa-dev libgles2-mesa-dev libgl1-mesa-dev \
    libjpeg-dev \
    openssh-server \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd /root/.ssh \
    && chmod 700 /root/.ssh

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# PyTorch 2.6.0 + CUDA 12.4
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Clone TRELLIS.2 (with submodules for o-voxel/eigen)
RUN git clone --recursive --depth 1 https://github.com/microsoft/TRELLIS.2.git /app/trellis2
WORKDIR /app/trellis2

# ── Basic deps (from setup.sh --basic) ──
# plyfile: required by o-voxel.io.ply at import time. Without it, o-voxel
# fails to import, which silently masks itself as a HuggingFace 404 inside
# the trellis2 model loader's fallback path. Took a full traceback dump to
# find this — see ai-gateway commit "fail-fast on app errors".
RUN pip install --no-cache-dir \
    imageio imageio-ffmpeg tqdm easydict \
    opencv-python-headless ninja trimesh \
    transformers \
    pandas lpips zstandard \
    Pillow kornia timm \
    plyfile \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# Our server deps
RUN pip install --no-cache-dir \
    fastapi uvicorn python-multipart httpx \
    huggingface_hub hf-xet \
    scipy safetensors accelerate einops

# ── flash-attn (from setup.sh --flash-attn) ──
RUN pip install --no-cache-dir flash-attn==2.7.3 --no-build-isolation \
    || echo "WARN: flash-attn failed — will use xformers or sdpa fallback"

# ── nvdiffrast (from setup.sh --nvdiffrast) ──
# CRITICAL: must succeed — o-voxel imports it at module load time and the
# error gets masked as a HuggingFace 404. The --no-build-isolation flag is
# required because nvdiffrast's setup.py imports torch for CUDA arch detection.
# We also explicitly set MAX_JOBS=2 to keep RAM under CI limits.
RUN mkdir -p /tmp/extensions && \
    git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast && \
    MAX_JOBS=2 pip install --no-cache-dir /tmp/extensions/nvdiffrast --no-build-isolation && \
    python -c "import nvdiffrast; print('nvdiffrast OK')"

# ── nvdiffrec (from setup.sh --nvdiffrec, renderutils branch) ──
RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec && \
    pip install --no-cache-dir /tmp/extensions/nvdiffrec --no-build-isolation \
    || echo "WARN: nvdiffrec failed"

# ── CuMesh (from setup.sh --cumesh) ──
RUN git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh && \
    pip install --no-cache-dir /tmp/extensions/CuMesh --no-build-isolation \
    || echo "WARN: CuMesh failed — mesh ops will use CPU"

# ── FlexGEMM (from setup.sh --flexgemm) ──
RUN git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM && \
    pip install --no-cache-dir /tmp/extensions/FlexGEMM --no-build-isolation \
    || echo "WARN: FlexGEMM failed — will use torch matmul"

# ── o-voxel (from setup.sh --o-voxel, copied from repo root) ──
RUN cp -r /app/trellis2/o-voxel /tmp/extensions/o-voxel && \
    pip install --no-cache-dir /tmp/extensions/o-voxel --no-build-isolation \
    || echo "WARN: o-voxel failed"

# ── trellis2 package (add to Python path since no setup.py) ──
ENV PYTHONPATH="/app/trellis2:${PYTHONPATH}"

# Model is downloaded on first request (NOT pre-baked) to keep the image
# small (~5GB instead of ~30GB) and avoid Vast.ai pull timeouts on slow
# residential hosts. The HuggingFace download takes ~2-5 min on a GPU host
# with good bandwidth, after which subsequent requests are instant.
# Set HF_HOME to /workspace if available (Vast.ai mounts persistent storage there)
ENV HF_HOME=/root/.cache/huggingface

# Copy server + entrypoint
COPY server.py /app/server.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Expose ports: 22 for sshd (Vast.ai SSH tunnel), 8000 for FastAPI inference
EXPOSE 22 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# start.sh launches sshd in background then runs FastAPI in foreground.
# This is required for Vast.ai SSH tunnel access — without sshd in the
# container, Vast's port forwarding to :22 has nothing to talk to.
CMD ["/app/start.sh"]
