# ── Qwen3-TTS — Multilingual TTS + Voice Cloning (ai-gateway) ─────────────
# Wraps Alibaba Qwen3-TTS (https://github.com/QwenLM/Qwen3-TTS). Exposes the
# OpenAI-compatible /v1/audio/speech and /v1/audio/voices endpoints so the
# canal-dark / babelcast pipelines can route to it identically to Modal
# Qwen3-TTS or kokoro-tts.
#
# Voice cloning: POST `reference_audio` + `ref_text` along with the input
# text and the synthesizer conditions the output on the reference voice.
#
# Model:  Alibaba/Qwen3-TTS-Base (~14GB pesos lazy-pulled no primeiro start)
# VRAM:   16GB+ (RTX 4090 24GB confortável; A100 ideal pra cloning)
# License: Apache 2.0 — see upstream LICENSE
#
# Build:
#   docker build -t marcosremar/qwen3-tts:latest .
# Run:
#   docker run --gpus all -p 8000:8000 -p 22:22 \
#     -v /workspace/hf-cache:/workspace/.cache/huggingface \
#     -e IDLE_TIMEOUT_MIN=15 \
#     marcosremar/qwen3-tts:latest
# ─────────────────────────────────────────────────────────────────────────────

ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    IDLE_TIMEOUT_MIN=15 \
    QWEN3_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    QWEN3_TTS_TOKENIZER=Qwen/Qwen3-TTS-Tokenizer-12Hz \
    QWEN3_TTS_DTYPE=bfloat16 \
    QWEN3_TTS_ATTN=flash_attention_2 \
    QWEN3_TTS_LANG=English \
    QWEN3_TTS_TIMEOUT_S=600 \
    QWEN3_TTS_WARMUP=0 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        git wget curl ca-certificates \
        ffmpeg libsm6 libxext6 libgl1 libsndfile1 sox libsox-dev \
        openssh-server \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd \
    && sed -i 's/#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config \
    && sed -i 's/#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config

WORKDIR /app

# PyTorch + Qwen runtime. Pin to CUDA 12.4 wheels matching the base image.
# `qwen-tts==0.1.1` is the official Alibaba helper (see
# https://pypi.org/project/qwen-tts) — pin EXACT version because earlier
# wheels had a different generate_voice_clone signature that caused the
# request hang in the previous build (kwargs silently dropped).
# `faster-qwen3-tts` adds CUDA-graph fast paths when present.
RUN pip install --upgrade pip \
    && pip install \
         "torch==2.5.1+cu124" "torchvision==0.20.1+cu124" "torchaudio==2.5.1+cu124" \
         --index-url https://download.pytorch.org/whl/cu124 \
    && pip install packaging ninja \
    && pip install \
         "transformers>=4.57,<5" "accelerate>=1.0" \
         "qwen-tts==0.1.1" "faster-qwen3-tts>=0.2.1" \
         "fastapi>=0.115" "uvicorn[standard]>=0.32" python-multipart \
         hf_transfer huggingface_hub soundfile numpy librosa pydantic-settings \
    # flash-attn is REQUIRED for stable Qwen3-TTS-Base voice-clone generation.
    # Without it, the qwen-tts library prints
    #   "Warning: flash-attn is not installed. Will only run the manual PyTorch version."
    # and the manual path produces unreliable EOS sampling — the model
    # generates audio until it hits max_new_tokens, yielding 30-600s of
    # repeated/garbled output for short texts. Building flash-attn from
    # source so the wheel matches the torch 2.5 / cu124 ABI.
    && MAX_JOBS=2 pip install "flash-attn==2.7.4.post1" --no-build-isolation

COPY server.py            /app/server.py
COPY idle_watchdog.py     /app/idle_watchdog.py
COPY start.sh             /app/start.sh
RUN chmod +x /app/start.sh

RUN mkdir -p /app/models /app/data /app/results /workspace/.cache/huggingface

EXPOSE 8000 22

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/app/start.sh"]
