#!/usr/bin/env bash

echo "[dit360] Starting server..."
echo "[dit360] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "[dit360] Model: ${MODEL_ID:-black-forest-labs/FLUX.1-dev}"
echo "[dit360] LoRA: ${LORA_ID:-Insta360-Research/DiT360-Panorama-Image-Generation}"
echo "[dit360] CPU offload: ${CPU_OFFLOAD:-auto}"

exec python3 /app/server.py
