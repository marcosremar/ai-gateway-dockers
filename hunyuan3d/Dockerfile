# ══════════════════════════════════════════════════════════════════════════════
# Hunyuan3D 2.0 — Image/Text-to-3D (.GLB) inference server
#
# Tencent's Hunyuan3D-2 generates 3D meshes with PBR textures from a single
# image (image-to-3D) or a text prompt (text-to-3D via DiT shape gen + paint
# pipeline for textures). Outputs .GLB for Babylon.js / Three.js.
#
# GPU: 12GB+ VRAM for shape+texture (RTX 3080 Ti / 4070 Ti / 4090).
#      6GB shape-only (RTX 3060+).
# ══════════════════════════════════════════════════════════════════════════════

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# System deps. openssh-server is required for Vast.ai SSH tunnel access — same
# rationale as trellis2: Vast forwards :22 to the container's sshd, which we
# launch alongside FastAPI in start.sh.
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

# PyTorch 2.6+ is required by transformers for safe torch.load checkpoints.
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Clone Hunyuan3D-2 (shape + texture pipelines live in hy3dgen/).
RUN git clone --depth 1 https://github.com/Tencent/Hunyuan3D-2.git /app/Hunyuan3D-2
WORKDIR /app/Hunyuan3D-2

# Install repo's requirements (transformers, diffusers, trimesh, rembg, etc).
RUN pip install --no-cache-dir -r requirements.txt

# Custom rasterizer + differentiable renderer (compiled C++/CUDA extensions
# used by the texture paint pipeline). Both live in hy3dgen/texgen/ as
# in-tree packages with their own setup.py.
RUN cd hy3dgen/texgen/custom_rasterizer && \
    pip install --no-cache-dir . --no-build-isolation \
    || echo "WARN: custom_rasterizer build failed — texture stage will fail"
RUN cd hy3dgen/texgen/differentiable_renderer && \
    pip install --no-cache-dir . --no-build-isolation \
    || echo "WARN: differentiable_renderer build failed — texture stage will fail"

# Server deps (same set as trellis2).
RUN pip install --no-cache-dir \
    fastapi uvicorn python-multipart httpx \
    huggingface_hub hf-xet \
    accelerate einops \
    sentencepiece tiktoken

# Add hy3dgen to PYTHONPATH so server.py can import it directly.
ENV PYTHONPATH="/app/Hunyuan3D-2"

# Model is downloaded on first request (NOT pre-baked) — keeps image small
# and avoids Vast.ai pull timeouts. ~10GB download (shape + paint + DINO).
ENV HF_HOME=/root/.cache/huggingface
ENV HUNYUAN3D_MODEL=tencent/Hunyuan3D-2

COPY server.py /app/server.py
COPY idle_watchdog.py /app/idle_watchdog.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 22 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/app/start.sh"]
