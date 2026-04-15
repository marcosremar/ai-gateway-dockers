#!/bin/bash
# Start server immediately, download checkpoints in background.
# Server returns status:"loading" until model is loaded.

download_checkpoints() {
  [ -f "/app/TRAM/data/.done" ] && return 0
  echo "[tram] Downloading models..."
  cd /app/TRAM && bash scripts/download_models.sh || echo "[tram] Download failed"
  touch /app/TRAM/data/.done
  cd /app
}

echo "[startup] Starting server..."
python3 /app/server.py &
SERVER_PID=$!

echo "[startup] Downloading checkpoints in background..."
download_checkpoints &

wait $SERVER_PID
