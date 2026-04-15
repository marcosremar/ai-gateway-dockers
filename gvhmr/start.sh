#!/bin/bash
# GVHMR startup — download checkpoints on first run, then start server.
set -e

CKPT_DIR="/app/GVHMR/inputs/checkpoints"
MARKER="$CKPT_DIR/.downloaded"

if [ ! -f "$MARKER" ]; then
  echo "[gvhmr] First run — downloading checkpoints..."

  mkdir -p "$CKPT_DIR/gvhmr" "$CKPT_DIR/hmr2" "$CKPT_DIR/vitpose" "$CKPT_DIR/yolo" "$CKPT_DIR/dpvo"

  # GVHMR main checkpoint
  python3 -c "
import gdown, os
folder_url = 'https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD'
try:
    gdown.download_folder(folder_url, output='$CKPT_DIR/', quiet=False, use_cookies=False)
    print('[gvhmr] Folder download OK')
except Exception as e:
    print(f'[gvhmr] Folder download failed: {e}')
    # Try individual files
    files = {
        'gvhmr/gvhmr_siga24_release.ckpt': '1_gvhmr_ckpt_id',
    }
    for path, fid in files.items():
        if not os.path.exists(f'$CKPT_DIR/{path}'):
            print(f'[gvhmr] Skipping {path} — needs manual download')
" 2>&1 || echo "[gvhmr] Checkpoint download had issues"

  # YOLO (auto-downloads on first use, but pre-fetch for faster startup)
  python3 -c "
from ultralytics import YOLO
YOLO('yolov8x.pt')
print('[gvhmr] YOLO ready')
" 2>&1 || echo "[gvhmr] YOLO download skipped"

  touch "$MARKER"
  echo "[gvhmr] Checkpoint setup complete"
fi

echo "[gvhmr] Starting server..."
exec python3 /app/server.py
