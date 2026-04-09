# ── snapgpu-runtime-babelcast — pre-baked models for BabelCast speech pipeline ─
# Extends the snapgpu-runtime base image with pre-baked models so the first boot
# only needs to start uvicorn + load weights into VRAM (~10s) instead of also
# downloading them (~2 min). After the first snapshot, subsequent boots restore
# the full CPU+GPU state in ~2-5 seconds.
#
# Models baked in:
#   - TranslateGemma 4B Q8_0 GGUF (~5GB) — LLM translation via llama.cpp
#   - Faster Whisper large-v3-turbo (~3GB) — STT via CTranslate2
#
# Total image size: ~12-15GB (base ~3GB + models ~8GB + deps ~2GB)
# ─────────────────────────────────────────────────────────────────────────────

FROM marcosremar/snapgpu-runtime:latest

# ── Install ML inference deps (matching babelcast-subtitle) ────────────────
# PyTorch is required for Whisper; llama-cpp-python for TranslateGemma.
ARG CUDA_INDEX=https://download.pytorch.org/whl/cu128
RUN pip install --break-system-packages --no-cache-dir \
    torch==2.11.0 --index-url ${CUDA_INDEX}

ARG LLAMA_CPP_VERSION=0.3.19
RUN curl -L -o /tmp/llama_cpp.whl \
    "https://github.com/abetlen/llama-cpp-python/releases/download/v${LLAMA_CPP_VERSION}-cu124/llama_cpp_python-${LLAMA_CPP_VERSION}-cp312-cp312-linux_x86_64.whl" && \
    pip install --break-system-packages --no-cache-dir "/tmp/llama_cpp.whl[server]" && \
    rm /tmp/llama_cpp.whl

RUN pip install --break-system-packages --no-cache-dir \
    "faster-whisper>=1.2.1" \
    "ctranslate2>=4.7.1" \
    httpx \
    "huggingface-hub>=1.0.0" \
    "hf_xet>=1.4.0"

# ── Pre-bake models using hf_hub_download (lands in standard HF cache) ────
# Uses the validated hf-xet backend with fixed concurrency for fastest downloads.
ENV HF_XET_HIGH_PERFORMANCE=1 \
    HF_XET_FIXED_DOWNLOAD_CONCURRENCY=50

# TranslateGemma GGUF
RUN python3 -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download('bullerwins/translategemma-4b-it-GGUF', 'translategemma-4b-it-Q8_0.gguf')"

# Faster Whisper large-v3-turbo (CTranslate2 format)
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('deepdml/faster-whisper-large-v3-turbo-ct2', max_workers=8)"

# ── Copy the babelcast server.py (speech pipeline handler) ─────────────────
# This file is registered as a snapgpu App at boot via SNAPGPU_PRELOAD_APP.
# TODO: Once the snapgpu App decorator is wired for babelcast, this becomes
# the @app.cls entry point. For now it runs as a standalone FastAPI server
# alongside the snapgpu gateway.
# COPY babelcast_server.py /app/babelcast_server.py

ENV SNAPGPU_PRELOAD_APP=babelcast
