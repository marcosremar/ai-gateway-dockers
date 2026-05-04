#!/usr/bin/env bash
# latentsync startup: SSH (PUBKEY auth, optional) + FastAPI uvicorn :8000.
# PUBLIC_KEY env var (if set) is appended to /root/.ssh/authorized_keys
# so the ai-gateway can SSH-patch the running pod for fast iteration.
set -e

LOG_FILE="${LOG_FILE:-/tmp/container.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[latentsync] $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "[latentsync] GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo 'none')"
echo "[latentsync] Variant: ${LATENTSYNC_VARIANT:-1.5}"
echo "[latentsync] Idle timeout: ${IDLE_TIMEOUT_MIN:-15}m"
echo "[latentsync] Disk /: $(df -h / | tail -1 | awk '{print $2 " total, " $4 " free"}')"

# Preferir /workspace para HF cache + checkpoints (persistente em Vast / RunPod / Hyperstack)
if [ -d "/workspace" ]; then
    export HF_HOME=/workspace/.cache/huggingface
    export LATENTSYNC_CHECKPOINT_DIR=/workspace/latentsync-checkpoints
    mkdir -p "$HF_HOME" "$LATENTSYNC_CHECKPOINT_DIR"
    if [ ! -L /app/checkpoints ]; then
        rm -rf /app/checkpoints
        ln -s "$LATENTSYNC_CHECKPOINT_DIR" /app/checkpoints
    fi
    echo "[latentsync] Using /workspace — HF_HOME=$HF_HOME"
    echo "[latentsync] Disk /workspace: $(df -h /workspace | tail -1 | awk '{print $2 " total, " $4 " free"}')"
fi

# ── SSH bootstrap (optional, gateway iteration loop relies on it) ───────────
mkdir -p /root/.ssh
chmod 700 /root/.ssh
if [ -n "${PUBLIC_KEY:-}" ]; then
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    echo "[latentsync] Injected PUBLIC_KEY into authorized_keys"
fi
if [ -n "${SSH_PUBLIC_KEY:-}" ]; then
    echo "$SSH_PUBLIC_KEY" >> /root/.ssh/authorized_keys
fi
chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true
ssh-keygen -A 2>/dev/null || true
/usr/sbin/sshd -D &
echo "[latentsync] sshd started"

# ── Download checkpoints if not present (first-run cold path) ───────────────
# LatentSync ships its weights via HF: ByteDance/LatentSync-{1.5,1.6}.
# We mirror them under /app/checkpoints (or /workspace/latentsync-checkpoints
# when /workspace is mounted) so the FastAPI server's lazy-load can find them.
VARIANT="${LATENTSYNC_VARIANT:-1.5}"
HF_REPO="ByteDance/LatentSync-${VARIANT}"
CKPT_MARKER="/app/checkpoints/latentsync_${VARIANT}.pt"

if [ ! -f "$CKPT_MARKER" ]; then
    echo "[latentsync] Downloading $HF_REPO weights (primeira vez)..."
    python3 - <<PY || echo "[latentsync] AVISO: download falhou, server tentará lazy-load"
import os
from huggingface_hub import snapshot_download
target = "/app/checkpoints"
os.makedirs(target, exist_ok=True)
snapshot_download(repo_id="${HF_REPO}", local_dir=target,
                  local_dir_use_symlinks=False, max_workers=4)
print("ok")
PY
    touch "$CKPT_MARKER" || true
else
    echo "[latentsync] Weights já presentes: $CKPT_MARKER"
fi

echo "[latentsync] Starting FastAPI server em 0.0.0.0:8000..."
exec python3 /app/server.py
