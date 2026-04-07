# Kokoro TTS 82M — Fastest open-weights TTS (Elo 1060)
# Native PyTorch, StyleTTS 2 + ISTFTNet, 24kHz, 54+ voices, 10 languages
#
# Build:  docker build -t marcosremar/kokoro-tts:latest .
# Run:    docker run --gpus all -p 8000:8000 marcosremar/kokoro-tts:latest

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_XET_HIGH_PERFORMANCE=1 \
    HF_XET_FIXED_DOWNLOAD_CONCURRENCY=50

RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        espeak-ng ffmpeg curl aria2 zstd && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    rm -rf /var/lib/apt/lists/*

# Install hf-download helper (aria2c-based, with HF Hub fallback)
COPY hf-download /usr/local/bin/hf-download
RUN chmod +x /usr/local/bin/hf-download

WORKDIR /app

# Install PyTorch + Kokoro
# Pin huggingface-hub >= 1.0 to get hf_xet as automatic dep (bench: +24% download speed)
RUN pip install --no-cache-dir \
    torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir \
    "kokoro>=0.9.4" \
    "misaki[en,ja,zh,ko,pt,es,fr,de,it,hi]>=0.9.4" \
    soundfile numpy \
    "huggingface-hub>=1.0.0" "hf_xet>=1.4.0" \
    "fastapi>=0.115.0" "uvicorn[standard]>=0.32.0"

# Pre-download model at build time
RUN python3 -c "from kokoro import KPipeline; p = KPipeline(lang_code='a'); [None for _ in p('test', voice='af_heart')]; print('OK')"

COPY server.py /app/server.py

RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
