#!/bin/bash
set -e

if [ ! -f "/app/TRAM/data/.downloaded" ]; then
  echo "[tram] Downloading models..."
  cd /app/TRAM
  bash scripts/download_models.sh 2>&1 || echo "[tram] WARNING: some downloads may have failed"
  touch /app/TRAM/data/.downloaded
fi

echo "[tram] Starting server..."
exec python3 /app/server.py
