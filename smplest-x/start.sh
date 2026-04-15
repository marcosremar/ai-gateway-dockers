#!/bin/bash
set -e
CKPT="/app/SMPLest-X/pretrained_models/smplest_x_h/smplest_x_h.pth.tar"

if [ ! -f "$CKPT" ]; then
  echo "[smplest-x] Downloading checkpoint (8.2GB)..."
  mkdir -p /app/SMPLest-X/pretrained_models/smplest_x_h
  python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('waanqii/SMPLest-X', 'smplest_x_h.pth.tar',
    local_dir='/app/SMPLest-X/pretrained_models/smplest_x_h')
print('[smplest-x] Checkpoint ready')
" 2>&1 || echo "[smplest-x] WARNING: checkpoint download failed"
fi

echo "[smplest-x] Starting server..."
exec python3 /app/server.py
