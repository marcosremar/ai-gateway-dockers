# ── whisper-align — Word-level forced alignment (ai-gateway) ─────────────────
# Wraps WhisperX (https://github.com/m-bain/whisperX, BSD-4) so the
# canal-dark / project-philosofi pipeline can map narration audio to
# word/sentence timestamps. The Remotion render then snaps scene cuts to
# spoken sentences instead of relying on hand-tuned offsets in script.json.
#
# Pipeline: script.json (scenes[]) → qwen3-tts → voice.mp3 → whisper-align
#         → timing.json {scenes: [{start_s, end_s, text}]} → Remotion.
#
# Endpoint:
#   POST /v1/audio/align (multipart) — accepts `audio` file + optional `text`
#   form field (target transcript for forced-align). Returns segments + words.
#
# Model:    whisper large-v3 (~3 GB) + wav2vec2 align model (per-language, ~1 GB)
# VRAM:     6 GB+ (works CPU; GPU recommended for batch).
# License:  WhisperX BSD-4 — see upstream LICENSE. pyannote needs HF token
#           only for diarization (we skip diarization, so no token required
#           in default mode).
#
# Build:
#   docker build -t marcosremar/whisper-align:latest .
# Run:
#   docker run --gpus all -p 8000:8000 -p 22:22 \
#     -v /workspace/hf-cache:/workspace/.cache/huggingface \
#     -e IDLE_TIMEOUT_MIN=15 \
#     marcosremar/whisper-align:latest
# ─────────────────────────────────────────────────────────────────────────────

ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    IDLE_TIMEOUT_MIN=15 \
    WHISPERX_MODEL=large-v3 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        git wget curl ca-certificates \
        ffmpeg libsndfile1 \
        openssh-server \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd \
    && sed -i 's/#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config \
    && sed -i 's/#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config

WORKDIR /app

# PyTorch + WhisperX runtime. Pin to CUDA 12.4 wheels matching the base
# image. WhisperX brings ctranslate2/faster-whisper + wav2vec2 alignment;
# pyannote.audio is required by whisperx for VAD/diarization scaffolding
# even when we skip diarization (we never call DiarizationPipeline).
RUN pip install --upgrade pip \
    && pip install \
         "torch==2.5.1+cu124" "torchaudio==2.5.1+cu124" \
         --index-url https://download.pytorch.org/whl/cu124 \
    && pip install \
         "whisperx>=3.1.5" "pyannote.audio>=3.1" \
         "fastapi>=0.115" "uvicorn[standard]>=0.32" python-multipart \
         hf_transfer huggingface_hub soundfile numpy librosa pydantic-settings

COPY server.py            /app/server.py
COPY idle_watchdog.py     /app/idle_watchdog.py
COPY start.sh             /app/start.sh
RUN chmod +x /app/start.sh

RUN mkdir -p /app/models /app/data /app/results /workspace/.cache/huggingface

EXPOSE 8000 22

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/app/start.sh"]
