#!/usr/bin/env bash
set -e

LOG_FILE="${LOG_FILE:-/tmp/container.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[musetalk] $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "[musetalk] GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo 'none')"
echo "[musetalk] Idle timeout: ${IDLE_TIMEOUT_MIN:-15}m"
echo "[musetalk] Disk /: $(df -h / | tail -1 | awk '{print $2 " total, " $4 " free"}')"

# Preferir /workspace para cache (persistente em Hyperstack/RunPod/Vast).
if [ -d "/workspace" ]; then
    export HF_HOME=/workspace/.cache/huggingface
    export MUSETALK_MODELS_DIR=/workspace/musetalk-models
    mkdir -p "$HF_HOME" "$MUSETALK_MODELS_DIR"
    # Symlink /app/models -> /workspace/musetalk-models (primeira vez só)
    if [ ! -L /app/models ]; then
        rm -rf /app/models
        ln -s "$MUSETALK_MODELS_DIR" /app/models
    fi
    echo "[musetalk] Using /workspace — HF_HOME=$HF_HOME"
    echo "[musetalk] Disk /workspace: $(df -h /workspace | tail -1 | awk '{print $2 " total, " $4 " free"}')"
fi

# Backup/restore + heartbeat são instalados pelo ai-gateway via SSH
# (pod-provisioner) logo depois do pod entrar em 'ready' — não precisamos
# fazer nada aqui. Os scripts vão pra /usr/local/bin/aigw-{backup,restore,agent}.

# Download dos pesos se ainda não baixados
if [ ! -f "/app/models/musetalkV15/unet.pth" ]; then
    echo "[musetalk] Downloading weights (primeira vez)..."
    cd /app && python3 download_models.py || {
        echo "[musetalk] download_models.py falhou, tentando download_weights.sh..."
        bash download_weights.sh || echo "[musetalk] AVISO: alguns pesos podem estar faltando"
    }
else
    echo "[musetalk] Weights já presentes: /app/models/musetalkV15/unet.pth"
fi

echo "[musetalk] Starting FastAPI server em 0.0.0.0:8000..."
exec python3 /app/server.py
