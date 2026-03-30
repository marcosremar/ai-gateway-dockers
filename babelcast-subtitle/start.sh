#!/usr/bin/env bash

echo "[babelcast-subtitle] Starting server..."
echo "[babelcast-subtitle] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "[babelcast-subtitle] LLM: ${LLM_REPO:-bullerwins/translategemma-4b-it-GGUF} / ${LLM_FILENAME:-translategemma-4b-it-Q8_0.gguf}"
echo "[babelcast-subtitle] flash_attn=${FLASH_ATTN:-true} n_batch=${N_BATCH:-1024} n_ctx=${N_CTX:-1024}"

# Install llama-cpp-python at runtime (pre-built CUDA wheel — no compilation)
# Cannot be baked into Docker image due to buildx platform detection issues.
if ! python3 -c "import llama_cpp" 2>/dev/null; then
    echo "[babelcast-subtitle] Installing llama-cpp-python (pre-built cu128 wheel)..."
    pip install --no-cache-dir "llama-cpp-python[server]>=0.3.19" \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu128 2>&1 | tail -3
    python3 -c "import llama_cpp; print('[babelcast-subtitle] llama-cpp-python', llama_cpp.__version__)" 2>/dev/null \
        || echo "[babelcast-subtitle] WARNING: llama-cpp-python install failed — LLM will be unavailable"
fi

exec python3 /app/server.py
