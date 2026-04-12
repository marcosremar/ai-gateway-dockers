#!/usr/bin/env bash

LOG_FILE="${LOG_FILE:-/tmp/container.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[wan-i2v] Starting server..."
echo "[wan-i2v] $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "[wan-i2v] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "[wan-i2v] Model: ${MODEL_ID:-Wan-AI/Wan2.1-I2V-14B-480P}"
echo "[wan-i2v] CPU offload: ${CPU_OFFLOAD:-1}"
echo "[wan-i2v] Idle timeout: ${IDLE_TIMEOUT_MIN:-15}m"
echo "[wan-i2v] Disk /: $(df -h / | tail -1 | awk '{print $2 " total, " $4 " free"}')"
echo "[wan-i2v] RAM: $(free -h | awk '/^Mem:/ {print $2 " total, " $7 " available"}')"

# Prefer /workspace for all large caches (Vast.ai/RunPod persistent disk)
# Without this, HuggingFace downloads 28GB model to the tiny root FS and fails.
if [ -d "/workspace" ]; then
    export HF_HOME=/workspace/.cache/huggingface
    export TORCHINDUCTOR_CACHE_DIR=/workspace/.torch-cache
    mkdir -p "$HF_HOME" "$TORCHINDUCTOR_CACHE_DIR"
    echo "[wan-i2v] Using /workspace — HF_HOME=$HF_HOME"
    echo "[wan-i2v] Disk /workspace: $(df -h /workspace | tail -1 | awk '{print $2 " total, " $4 " free"}')"
fi

exec python3 /app/server.py
