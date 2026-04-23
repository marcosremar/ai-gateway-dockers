# ── Ditto-TalkingHead — Real-Time Audio-Driven Avatar (TRT) ────────────────
# Wraps antgroup/ditto-talkinghead (Diffusion-Free Audio-Driven Talking-Head)
# with a FastAPI chat server that does Groq STT/LLM/TTS → Ditto inference →
# face-composite onto a body background.
#
# Modelo:  digital-avatar/ditto-talkinghead (~6GB pesos: cfg + onnx + pytorch
#          + TRT engines for Ampere+)
# VRAM:    ~6-8GB inference, 24GB+ recommended for headroom
# GPU alvo: A6000/L40S Ampere+ (TRT engines bake-in)
#
# Build:
#   docker build -t marcosremar/ditto:latest .
#
# Run:
#   docker run --gpus all -p 8000:8000 -p 22:22 \
#     -e GROQ_API_KEY=sk-... \
#     -e PUBLIC_KEY="ssh-ed25519 AAAA..." \
#     marcosremar/ditto:latest
# ─────────────────────────────────────────────────────────────────────────────

ARG BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    IDLE_TIMEOUT_MIN=15 \
    HF_HOME=/workspace/.cache/huggingface \
    PYTHONPATH=/app:/root/ditto-talkinghead

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3.10-venv python3-pip \
      git curl wget ca-certificates aria2 \
      ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 libxrender-dev libgomp1 \
      openssh-server \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# ── hf-download helper (multi-conn, mirrors musetalk pattern) ───────────────
COPY hf-download /usr/local/bin/hf-download
RUN chmod +x /usr/local/bin/hf-download

WORKDIR /app

# ── PyTorch CUDA 12.1 (compatível com TRT 8.6.1 + ditto prebuilt engines) ───
RUN pip3 install --no-cache-dir \
      torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
      --index-url https://download.pytorch.org/whl/cu121

# ── TensorRT 8.6.1 wheels (need NVIDIA pypi index) ──────────────────────────
RUN pip3 install --no-cache-dir \
      --index-url https://pypi.nvidia.com \
      --extra-index-url https://pypi.org/simple \
      tensorrt-libs==8.6.1 tensorrt-bindings==8.6.1 tensorrt==8.6.1

# ── CUDA python bindings ────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir "cuda-python<13"

# ── Ditto runtime + chat server deps ────────────────────────────────────────
RUN pip3 install --no-cache-dir \
      "numpy<2.0" scipy \
      mediapipe \
      "librosa>=0.10.0" "soundfile>=0.12.0" \
      "opencv-python>=4.8.0" \
      "imageio[ffmpeg]" ffmpeg-python \
      "fastapi>=0.100.0" "uvicorn[standard]>=0.22.0" "websockets>=12.0" \
      "python-multipart>=0.0.6" \
      "huggingface_hub>=0.20.0" hf-transfer \
      "groq>=0.11.0" httpx \
      "Pillow>=9.0" \
      omegaconf einops pyyaml tqdm requests \
      onnx onnxruntime-gpu==1.18.0 \
      "filterpy>=1.4.5" "scikit-image>=0.21.0"

# ── Clone ditto-talkinghead source ──────────────────────────────────────────
# Mantemos a árvore leve descartando .git.
RUN git clone --depth=1 https://github.com/antgroup/ditto-talkinghead /root/ditto-talkinghead && \
    rm -rf /root/ditto-talkinghead/.git

# ── Bake checkpoints (HuggingFace digital-avatar/ditto-talkinghead) ─────────
# ~6GB: ditto_cfg/ + ditto_onnx/ + ditto_pytorch/ + ditto_trt_Ampere_Plus/.
# Done at build time so cold-start is just SDK init (~5-10s).
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN mkdir -p /root/ditto-talkinghead/checkpoints && \
    python3 -c "from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='digital-avatar/ditto-talkinghead', \
local_dir='/root/ditto-talkinghead/checkpoints', \
local_dir_use_symlinks=False, max_workers=8)"

# ── MediaPipe face landmarker model ─────────────────────────────────────────
RUN curl -L -o /root/face_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# ── Avatar assets (baked-in) ────────────────────────────────────────────────
COPY charlie_ref.mp4     /root/charlie_ref.mp4
COPY charlie_body_bg.mp4 /root/charlie_body_bg.mp4

# ── Chat server + start script ──────────────────────────────────────────────
COPY chat_server.py /app/chat_server.py
COPY start.sh       /app/start.sh
RUN chmod +x /app/start.sh

# ── SSH server configuration (mirrors gpu-dev) ──────────────────────────────
RUN mkdir -p /var/run/sshd /root/.ssh \
    && chmod 700 /root/.ssh \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config

EXPOSE 8000 22

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/app/start.sh"]
