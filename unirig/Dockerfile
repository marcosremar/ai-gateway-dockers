FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps + Python 3.11 + Blender (provides bpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    git openssh-server curl wget software-properties-common \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 libgomp1 \
    xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd /root/.ssh \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install Blender 4.2 (headless — provides bpy module)
RUN wget -q https://mirror.clarkson.edu/blender/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz -O /tmp/blender.tar.xz \
    && tar xf /tmp/blender.tar.xz -C /opt/ \
    && rm /tmp/blender.tar.xz \
    && ln -s /opt/blender-4.2.0-linux-x64 /opt/blender

# Make Blender's bpy importable from system Python
ENV BLENDER_PATH=/opt/blender
ENV PATH="${BLENDER_PATH}:${PATH}"

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch for CUDA 12.1
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Clone UniRig
RUN git clone --depth 1 https://github.com/VAST-AI-Research/UniRig.git /app/unirig

WORKDIR /app/unirig

# Core Python deps (NO flash_attn — optional optimization, causes OOM)
RUN pip install --no-cache-dir \
    transformers==4.51.3 python-box einops omegaconf \
    pytorch_lightning lightning addict timm fast-simplification \
    trimesh open3d huggingface_hub numpy==1.26.4 \
    fastapi uvicorn python-multipart httpx

# spconv
RUN pip install --no-cache-dir spconv-cu121

# torch_scatter + torch_cluster (try multiple torch versions)
RUN pip install --no-cache-dir torch_scatter torch_cluster \
    -f https://data.pyg.org/whl/torch-2.3.1+cu121.html 2>/dev/null || \
    pip install --no-cache-dir torch_scatter torch_cluster \
    -f https://data.pyg.org/whl/torch-2.4.0+cu121.html 2>/dev/null || \
    pip install --no-cache-dir torch_scatter torch_cluster

# Patch UniRig's merge.py to use Blender's Python (which has bpy)
# The merge script calls `python -m src.inference.merge` but system python
# doesn't have bpy. We make it use Blender's bundled python instead.
RUN echo '#!/bin/bash\n/opt/blender/blender --background --python-expr "import sys; sys.argv = [\"blender\"] + \"$@\".split()" -- "$@"' > /usr/local/bin/blender-python \
    && chmod +x /usr/local/bin/blender-python

# Pre-download UniRig model weights
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('VAST-AI/UniRig', local_dir='/app/unirig/checkpoints')"

# Copy server + start script
COPY server.py /app/server.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 22 8000

CMD ["/app/start.sh"]
