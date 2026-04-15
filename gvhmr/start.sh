#!/bin/bash
# Start server immediately, download checkpoints in background.
# Server returns status:"loading" until model is loaded.

download_checkpoints() {
  [ -f "/app/GVHMR/inputs/checkpoints/.done" ] && return 0
  echo "[gvhmr] Downloading checkpoints..."
  python3 -c "import gdown; gdown.download_folder('https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD', output='/app/GVHMR/inputs/checkpoints/', quiet=False, use_cookies=False)" || echo "[gvhmr] Download failed"
  touch /app/GVHMR/inputs/checkpoints/.done
}

echo "[startup] Starting server..."
python3 /app/server.py &
SERVER_PID=$!

echo "[startup] Downloading checkpoints in background..."
download_checkpoints &

wait $SERVER_PID
