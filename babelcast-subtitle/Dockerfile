# ── BabelCast Subtitle — Universal GPU Image ────────────────────────────────
# Single image for ALL GPU architectures: Blackwell (sm_120), Ada (sm_89),
# Ampere (sm_80/86), Hopper (sm_90). No separate Blackwell variant needed.
#
# CUDA 12.8.1 runtime is the minimum for RTX 5090/5080 (sm_120).
# llama-cpp-python uses cu124 pre-built wheel (forward-compatible with 12.8).
#
# LLM: TranslateGemma 4B Q8_0 via llama.cpp (flash_attn + n_batch=1024)
# STT: Faster Whisper large-v3-turbo via CTranslate2
# ─────────────────────────────────────────────────────────────────────────────

ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    COMPUTE_TYPE=float16 \
    LLM_REPO=bullerwins/translategemma-4b-it-GGUF \
    LLM_FILENAME=translategemma-4b-it-Q8_0.gguf \
    N_CTX=1024 \
    N_BATCH=1024 \
    N_UBATCH=512 \
    FLASH_ATTN=true

# System deps — minimal for inference only
# aria2 + zstd: multi-connection downloads (~30% faster than curl/single-stream)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    ffmpeg libsndfile1 curl aria2 zstd && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    rm -rf /var/lib/apt/lists/*

# Install hf-download helper (aria2c-based, with HF Hub fallback)
COPY hf-download /usr/local/bin/hf-download
RUN chmod +x /usr/local/bin/hf-download

WORKDIR /app

# Step 1: PyTorch from CUDA 12.8 index
ARG CUDA_INDEX=https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir torch==2.11.0 --index-url ${CUDA_INDEX}

# Step 2: llama-cpp-python — download pre-built cu124 wheel directly (no compilation).
# cu124 wheel is forward-compatible with CUDA 12.8 runtime (NVIDIA minor version compat).
# Direct curl download bypasses pip index resolution issues.
ARG LLAMA_CPP_VERSION=0.3.19
RUN curl -L -o /tmp/llama_cpp_python-${LLAMA_CPP_VERSION}-cp311-cp311-linux_x86_64.whl \
    "https://github.com/abetlen/llama-cpp-python/releases/download/v${LLAMA_CPP_VERSION}-cu124/llama_cpp_python-${LLAMA_CPP_VERSION}-cp311-cp311-linux_x86_64.whl" && \
    pip install --no-cache-dir "/tmp/llama_cpp_python-${LLAMA_CPP_VERSION}-cp311-cp311-linux_x86_64.whl[server]" && \
    rm /tmp/llama_cpp_python-*.whl

# Step 3: Everything else from PyPI
# huggingface_hub >= 0.32 auto-installs hf_xet as a dependency; we pin it
# explicitly so the Xet backend is always active (validated as fastest in bench).
RUN pip install --no-cache-dir \
    "faster-whisper>=1.2.1" \
    "ctranslate2>=4.7.1" \
    httpx \
    "huggingface-hub>=1.0.0" \
    "hf_xet>=1.4.0" \
    python-multipart \
    uvicorn[standard] \
    fastapi

# Download tuning: hf-xet with HIGH_PERFORMANCE + pinned concurrency.
# Validated on Vast.ai RTX 4090 (64GB RAM, 3 runs, 2026-04-07):
#   legacy hf_transfer:      319 MB/s avg (baseline)
#   HP + FIXED_CONCURRENCY=50: 394 MB/s avg (+24%, 11% spread)
#   FIXED_CONCURRENCY=100:   410 MB/s avg (+29%, 20% spread)
#   xet default (adaptive):  269 MB/s avg (-16%, 34% spread — adaptive is too
#                            slow to escalate on short downloads)
# HP + FIXED=50 is the best stability/speed tradeoff. HIGH_PERFORMANCE requires
# >=64GB RAM; on smaller hosts, set HF_XET_HIGH_PERFORMANCE=0 in env overrides.
ENV HF_HOME=/app/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface/hub \
    HF_XET_HIGH_PERFORMANCE=1 \
    HF_XET_FIXED_DOWNLOAD_CONCURRENCY=50

# Pre-download GGUF model. hf-xet with HIGH_PERFORMANCE + FIXED_CONCURRENCY=50
# (ENV vars above) gives 394 MB/s on Vast.ai RTX 4090 (validated 3 runs, +24%
# vs legacy hf_transfer). Using the Python SDK so the file lands in the standard
# HF cache layout that try_to_load_from_cache() expects at runtime.
RUN python3 -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download('bullerwins/translategemma-4b-it-GGUF', 'translategemma-4b-it-Q8_0.gguf')" \
    || echo "GGUF pre-download skipped (will download at boot)"

# Pre-download Whisper large-v3-turbo (~3GB) — eliminates the 2-3min runtime download
# that was the dominant cold-boot cost on Vast.ai hosts.
RUN python3 -c "\
from faster_whisper import WhisperModel; \
print('Pre-baking Whisper large-v3-turbo...'); \
m = WhisperModel('large-v3-turbo', device='cpu', compute_type='int8'); \
print('Whisper cached'); \
del m" \
    || echo "Whisper pre-download skipped (will download at boot)"

COPY server.py .
COPY idle_watchdog.py .
COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["/app/start.sh"]
