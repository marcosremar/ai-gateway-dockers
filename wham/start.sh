#!/bin/bash
set -e
CKPT="/app/WHAM/checkpoints/wham_vit_w_3dpw.pth.tar"

if [ ! -f "$CKPT" ]; then
  echo "[wham] Downloading checkpoint..."
  mkdir -p /app/WHAM/checkpoints
  python3 -c "
import gdown
gdown.download('1Erjkho7O0bnZFawarntICRUCroaKabRE', '/app/WHAM/checkpoints/wham_vit_w_3dpw.pth.tar', quiet=False)
print('[wham] Checkpoint ready')
" 2>&1 || echo "[wham] WARNING: checkpoint download failed"
fi

echo "[wham] Starting server..."
exec python3 /app/server.py
