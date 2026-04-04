#!/usr/bin/env bash

# Redirect all stdout/stderr to both console and log file for post-mortem debugging.
# RunPod has no log API — this file persists on container disk for SSH retrieval
# and is served via GET /debug/logs endpoint.
LOG_FILE="${LOG_FILE:-/tmp/container.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[dit360] Starting server..."
echo "[dit360] $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "[dit360] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "[dit360] Model: ${MODEL_ID:-black-forest-labs/FLUX.1-dev}"
echo "[dit360] LoRA: ${LORA_ID:-Insta360-Research/DiT360-Panorama-Image-Generation}"
echo "[dit360] CPU offload: ${CPU_OFFLOAD:-auto}"
echo "[dit360] Disk: $(df -h / | tail -1 | awk '{print $2 " total, " $4 " free"}')"

exec python3 /app/server.py
