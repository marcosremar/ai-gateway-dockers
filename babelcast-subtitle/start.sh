#!/usr/bin/env bash
set -euo pipefail

echo "[babelcast-subtitle] Starting server..."
echo "[babelcast-subtitle] CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-all}"
echo "[babelcast-subtitle] Compute type: ${COMPUTE_TYPE:-float16}"
echo "[babelcast-subtitle] GPU arch detection will happen at model load time"

# Run the inference server — binds 0.0.0.0:8000
exec python3 /app/server.py
