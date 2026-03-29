#!/usr/bin/env bash
set -euo pipefail

echo "[babelcast-subtitle] Starting server..."
echo "[babelcast-subtitle] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "[babelcast-subtitle] LLM: ${LLM_REPO:-bullerwins/translategemma-4b-it-GGUF} / ${LLM_FILENAME:-translategemma-4b-it-Q8_0.gguf}"
echo "[babelcast-subtitle] flash_attn=${FLASH_ATTN:-true} n_batch=${N_BATCH:-1024} n_ctx=${N_CTX:-1024}"

# Install llama-cpp-python with CUDA at runtime (needs GPU host CUDA toolkit)
if ! python3 -c "import llama_cpp" 2>/dev/null; then
    echo "[babelcast-subtitle] Installing llama-cpp-python with CUDA..."
    CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir "llama-cpp-python[server]>=0.3.19" 2>&1 | tail -5
    echo "[babelcast-subtitle] llama-cpp-python installed"
fi

exec python3 /app/server.py
