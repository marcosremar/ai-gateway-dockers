#!/usr/bin/env bash

LOG_FILE="${LOG_FILE:-/tmp/container.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[wan-i2v] Starting server..."
echo "[wan-i2v] $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "[wan-i2v] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "[wan-i2v] Model: ${MODEL_ID:-Wan-AI/Wan2.1-I2V-14B-480P}"
echo "[wan-i2v] CPU offload: ${CPU_OFFLOAD:-1}"
echo "[wan-i2v] Idle timeout: ${IDLE_TIMEOUT_MIN:-15}m"
echo "[wan-i2v] Disk: $(df -h / | tail -1 | awk '{print $2 " total, " $4 " free"}')"
echo "[wan-i2v] RAM: $(free -h | awk '/^Mem:/ {print $2 " total, " $7 " available"}')"

# RunPod: prefer /workspace for torch cache (persists across restarts on network volume)
if [ -d "/workspace" ]; then
    export TORCHINDUCTOR_CACHE_DIR=/workspace/.torch-cache
    echo "[wan-i2v] Using /workspace for torch cache"
fi

exec python3 /app/server.py
