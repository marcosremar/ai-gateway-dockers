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

# ── Auto-backup do /workspace pro B2/S3 a cada N horas ─────────────────────
# Roda no background. No-op se B2_ACCOUNT_ID não estiver setado.
# Configurável via env: BACKUP_INTERVAL_HOURS (default 24).
INTERVAL_H="${BACKUP_INTERVAL_HOURS:-24}"
if [ -x /app/backup_workspace.sh ]; then
    echo "[musetalk] Workspace auto-backup loop: every ${INTERVAL_H}h (no-op without B2_*)"
    (
        # Bootstrap delay: primeira execução em 5 min após boot
        sleep 300
        while true; do
            /app/backup_workspace.sh || true
            sleep "$((INTERVAL_H * 3600))"
        done
    ) &
fi

echo "[musetalk] Starting FastAPI server em 0.0.0.0:8000..."
exec python3 /app/server.py
