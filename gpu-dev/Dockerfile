# ── GPU Dev — Minimal interactive development image ─────────────────────────
# Ubuntu 22.04 + CUDA 12.1 + Python 3.10 + SSH + uv
# For fast iteration BEFORE baking a real Docker image.
# Start with `bunx ai-gateway gpu dev start`, iterate via `exec`/`sh`/`push`,
# then `snapshot` to commit into a named image.
#
# Target size: ~2-3GB (vs 8-10GB for project-specific images).

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ── Base system ─────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3-pip python3.10-venv \
      git curl wget ca-certificates \
      ffmpeg libgl1 libglib2.0-0 \
      openssh-server \
      tmux htop nano vim \
      build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

# ── Fast Python package manager ─────────────────────────────────────────────
RUN pip install -q --upgrade pip setuptools wheel \
    && pip install -q uv

# ── Baseline python packages (commonly needed) ──────────────────────────────
RUN pip install -q \
      "numpy<2.0" scipy \
      requests httpx \
      fastapi "uvicorn[standard]" python-multipart \
      huggingface_hub \
      Pillow opencv-python-headless

# ── PyTorch 2.4.1 + CUDA 12.1 (reasonable default, can be overridden) ──────
RUN pip install -q torch==2.4.1 torchvision==0.19.1 \
      --index-url https://download.pytorch.org/whl/cu121

# ── SSH server configuration ────────────────────────────────────────────────
RUN mkdir -p /var/run/sshd /root/.ssh \
    && chmod 700 /root/.ssh \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config

# ── Minimal health server + startup ─────────────────────────────────────────
WORKDIR /app
COPY health_server.py /app/health_server.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8000 22

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/app/start.sh"]
