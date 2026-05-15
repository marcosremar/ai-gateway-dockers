#!/bin/bash
set -e

CKPT=/checkpoints/sam2.1_hiera_large.pt
if [ ! -f "$CKPT" ]; then
  echo "Downloading SAM2-large weights..."
  mkdir -p /checkpoints
  wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
    -O "$CKPT"
  echo "Weights downloaded."
fi

exec python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
