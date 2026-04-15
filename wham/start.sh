#!/bin/bash
# Start server immediately, download checkpoints in background.
# Server returns status:"loading" until model is loaded.

download_checkpoints() {
  CKPT="/app/WHAM/checkpoints/wham_vit_w_3dpw.pth.tar"
  [ -f "$CKPT" ] && return 0
  echo "[wham] Downloading checkpoint..."
  mkdir -p /app/WHAM/checkpoints
  python3 -c "import gdown; gdown.download('1Erjkho7O0bnZFawarntICRUCroaKabRE', '/app/WHAM/checkpoints/wham_vit_w_3dpw.pth.tar', quiet=False)" || echo "[wham] Download failed"
}

echo "[startup] Starting server..."
python3 /app/server.py &
SERVER_PID=$!

echo "[startup] Downloading checkpoints in background..."
download_checkpoints &

wait $SERVER_PID
