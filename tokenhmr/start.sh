#!/bin/bash
# Start server immediately, download checkpoints in background.
# Server returns status:"loading" until model is loaded.

download_checkpoints() {
  CKPT="/app/checkpoints/tokenhmr_model_latest.ckpt"
  [ -f "$CKPT" ] && return 0
  echo "[tokenhmr] Downloading checkpoint..."
  mkdir -p /app/checkpoints
  wget -q "https://download.is.tue.mpg.de/download.php?domain=tokenhmr&sfile=tokenhmr_model_latest.zip" -O /tmp/tok.zip && unzip -o /tmp/tok.zip -d /app/checkpoints/ && rm /tmp/tok.zip || echo "[tokenhmr] Download failed"
}

echo "[startup] Starting server..."
python3 /app/server.py &
SERVER_PID=$!

echo "[startup] Downloading checkpoints in background..."
download_checkpoints &

wait $SERVER_PID
