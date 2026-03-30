#!/usr/bin/env bash

echo "[babelcast-subtitle] Starting server..."
echo "[babelcast-subtitle] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "[babelcast-subtitle] LLM: ${LLM_REPO:-bullerwins/translategemma-4b-it-GGUF} / ${LLM_FILENAME:-translategemma-4b-it-Q8_0.gguf}"
echo "[babelcast-subtitle] flash_attn=${FLASH_ATTN:-true} n_batch=${N_BATCH:-1024} n_ctx=${N_CTX:-1024}"

exec python3 /app/server.py
