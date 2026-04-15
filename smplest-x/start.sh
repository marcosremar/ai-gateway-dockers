#!/bin/bash
# Start server immediately, download checkpoints in background.
# Server returns status:"loading" until model is loaded.

download_checkpoints() {
  CKPT="/app/SMPLest-X/pretrained_models/smplest_x_h/smplest_x_h.pth.tar"
  [ -f "$CKPT" ] && return 0
  echo "[smplest-x] Downloading checkpoint (8.2GB)..."
  mkdir -p /app/SMPLest-X/pretrained_models/smplest_x_h
  python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('waanqii/SMPLest-X', 'smplest_x_h.pth.tar', local_dir='/app/SMPLest-X/pretrained_models/smplest_x_h')" || echo "[smplest-x] Download failed"
}

echo "[startup] Starting server..."
python3 /app/server.py &
SERVER_PID=$!

echo "[startup] Downloading checkpoints in background..."
download_checkpoints &

wait $SERVER_PID
