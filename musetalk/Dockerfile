# ── MuseTalk — Real-Time Audio-Driven Lip Sync ─────────────────────────────
# Baseado em https://github.com/ruxir-ig/MuseTalk-API (fork com Docker +
# FastAPI) do modelo oficial TMElyralab/MuseTalk. Adicionamos convenções do
# ai-gateway: idle_watchdog, /workspace HF cache, healthcheck FastAPI, e um
# endpoint WebSocket para streaming de áudio→frames.
#
# Modelo:  TMElyralab/MuseTalk (V1.5, ~4GB pesos no total)
# VRAM:    8-12GB para 256x256 real-time  |  16GB+ para batch maior
# GPU alvo: RTX A6000 48GB (ideal) / A4000 16GB (mínimo funcional)
#
# Build:
#   docker build -t marcosremar/musetalk:latest .
#
# Run:
#   docker run --gpus all -p 8000:8000 \
#     -v /workspace/hf-cache:/workspace/.cache/huggingface \
#     -v /workspace/musetalk-models:/app/models \
#     -e IDLE_TIMEOUT_MIN=15 \
#     marcosremar/musetalk:latest
# ─────────────────────────────────────────────────────────────────────────────

# CUDA 12.4 + cuDNN9 — requerido pra Blackwell (RTX 5090, sm_120) e pra
# torch 2.6 + mmcv 2.2. Mantém compatibilidade com Ampere (A100) e Ada (4090).
ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    IDLE_TIMEOUT_MIN=15 \
    HF_HOME=/workspace/.cache/huggingface \
    MUSETALK_MODELS_DIR=/app/models \
    PYTHONPATH=/app

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    git curl ffmpeg aria2 \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# NOTE: setuptools <80 required — mmcv build hooks import pkg_resources,
# which setuptools 80 removed by default.
RUN pip3 install --no-cache-dir --upgrade pip wheel && \
    pip3 install --no-cache-dir "setuptools<80"

# ── hf-download helper (multi-conn) ──────────────────────────────────────────
COPY hf-download /usr/local/bin/hf-download
RUN chmod +x /usr/local/bin/hf-download

WORKDIR /app

# ── PyTorch CUDA 12.4 (suporta Blackwell sm_120 / RTX 5090) ──────────────────
# torch 2.4.0 tem wheel pré-compilada de mmcv 2.2 em cu124 (versões mais
# novas exigem compilação from-source). Mantém compatibilidade com Ada/Ampere.
RUN pip3 install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ── MMPose stack (detecção de landmarks faciais) ─────────────────────────────
# Usa wheel pré-compilada do OpenMMLab (cu124/torch2.4) pra evitar compilação
# do zero (que depende de pkg_resources e costuma falhar com setuptools novo).
RUN pip3 install --no-cache-dir -U openmim && \
    mim install mmengine && \
    pip3 install --no-cache-dir "mmcv==2.2.0" \
        -f "https://download.openmmlab.com/mmcv/dist/cu124/torch2.4/index.html" && \
    mim install "mmdet==3.3.0" && \
    pip3 install --no-cache-dir --no-build-isolation "mmpose==1.3.2"

# ── MuseTalk + infra deps ────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir \
    diffusers==0.30.2 \
    accelerate==0.28.0 \
    transformers==4.39.2 \
    "huggingface_hub>=0.20.0" \
    numpy==1.23.5 \
    "scipy>=1.10.0" \
    "librosa>=0.10.0" \
    "soundfile>=0.12.0" \
    "opencv-python>=4.8.0" \
    "imageio[ffmpeg]" ffmpeg-python moviepy \
    "gfpgan>=1.3.8" "facexlib>=0.3.0" "basicsr>=1.4.2" "realesrgan>=0.3.0" \
    "fastapi>=0.100.0" "uvicorn[standard]>=0.22.0" "websockets>=12.0" \
    "python-multipart>=0.0.6" \
    omegaconf "einops>=0.7.0" "pyyaml>=6.0" tqdm gdown requests \
    "filterpy>=1.4.5" "lmdb>=1.4.0" "psutil>=5.9.0" \
    "Pillow>=9.0"

# ── Clone MuseTalk-API (fonte upstream) ──────────────────────────────────────
# Fixamos no commit HEAD no momento do build para reprodutibilidade. Caso o
# fork receba fixes relevantes, bastará rebuild. Mantemos a árvore leve
# descartando .git.
RUN git clone --depth=1 https://github.com/ruxir-ig/MuseTalk-API /tmp/musetalk-src && \
    cp -r /tmp/musetalk-src/musetalk /app/musetalk && \
    cp -r /tmp/musetalk-src/configs /app/configs && \
    cp -r /tmp/musetalk-src/api /app/api && \
    cp /tmp/musetalk-src/download_models.py /app/download_models.py && \
    cp /tmp/musetalk-src/download_weights.sh /app/download_weights.sh && \
    chmod +x /app/download_weights.sh && \
    rm -rf /tmp/musetalk-src

# ── Nossos arquivos ai-gateway ───────────────────────────────────────────────
COPY server.py        /app/server.py
COPY idle_watchdog.py /app/idle_watchdog.py
COPY start.sh         /app/start.sh
RUN chmod +x /app/start.sh

# Pasta persistente pros modelos
RUN mkdir -p /app/models /app/data /app/results /workspace/.cache/huggingface

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=240s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/app/start.sh"]
