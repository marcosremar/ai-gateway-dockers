FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps + Python 3.11 + X11 libs (bpy needs libXi etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    git openssh-server curl wget \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 libgomp1 \
    libxi6 libxxf86vm1 libxfixes3 libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd /root/.ssh \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch 2.5.1 for CUDA 12.1 (pinned — torch_cluster/scatter need matching wheels)
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Clone UniRig
RUN git clone --depth 1 https://github.com/VAST-AI-Research/UniRig.git /app/unirig

WORKDIR /app/unirig

# bpy (Blender Python API — pip version, not full Blender)
RUN pip install --no-cache-dir bpy

# Core Python deps
RUN pip install --no-cache-dir --ignore-installed \
    transformers==4.51.3 python-box einops omegaconf \
    pytorch_lightning lightning addict timm fast-simplification \
    trimesh open3d huggingface_hub \
    fastapi uvicorn python-multipart httpx

# spconv for CUDA 12.1
RUN pip install --no-cache-dir spconv-cu121

# torch_scatter + torch_cluster (must match torch 2.5.1 + cu121)
RUN pip install --no-cache-dir torch_scatter torch_cluster \
    -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

# Pre-download UniRig model weights (~2GB)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('VAST-AI/UniRig', local_dir='/app/unirig/checkpoints')"

# Copy server + start script
COPY server.py /app/server.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 22 8000

CMD ["/app/start.sh"]
