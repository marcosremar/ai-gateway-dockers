#!/bin/bash
set -e
CKPT="/app/checkpoints/tokenhmr_model_latest.ckpt"

if [ ! -f "$CKPT" ]; then
  echo "[tokenhmr] Downloading checkpoint..."
  mkdir -p /app/checkpoints
  wget -q "https://download.is.tue.mpg.de/download.php?domain=tokenhmr&sfile=tokenhmr_model_latest.zip" \
    -O /tmp/tokenhmr.zip 2>&1 && \
    unzip -o /tmp/tokenhmr.zip -d /app/checkpoints/ && \
    rm /tmp/tokenhmr.zip && \
    echo "[tokenhmr] Checkpoint ready" || \
    echo "[tokenhmr] WARNING: checkpoint download failed (may require registration)"
fi

echo "[tokenhmr] Starting server..."
exec python3 /app/server.py
