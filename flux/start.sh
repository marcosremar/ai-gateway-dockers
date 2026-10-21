#!/usr/bin/env bash
# flux startup: SSH (PUBKEY auth) + FastAPI uvicorn :8000.
# PUBLIC_KEY env var (if set) is appended to /root/.ssh/authorized_keys
# so the ai-gateway can SSH-patch the running pod for fast iteration.
set -e

LOG_FILE="${LOG_FILE:-/tmp/container.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[flux] $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "[flux] GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo 'none')"
echo "[flux] Model: ${FLUX_MODEL:-black-forest-labs/FLUX.1-schnell}"
echo "[flux] Idle timeout: ${IDLE_TIMEOUT_MIN:-15}m"

if [ -d "/workspace" ]; then
    export HF_HOME=/workspace/.cache/huggingface
    mkdir -p "$HF_HOME"
    echo "[flux] Using /workspace — HF_HOME=$HF_HOME"
fi

# ── SSH bootstrap ───────────────────────────────────────────────────────────
mkdir -p /root/.ssh && chmod 700 /root/.ssh
if [ -n "${PUBLIC_KEY:-}" ]; then
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    echo "[flux] Injected PUBLIC_KEY"
fi
if [ -n "${SSH_PUBLIC_KEY:-}" ]; then
    echo "$SSH_PUBLIC_KEY" >> /root/.ssh/authorized_keys
fi
chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true
ssh-keygen -A 2>/dev/null || true
/usr/sbin/sshd -D &
echo "[flux] sshd started"

# ── Pre-warm HF download (background) ───────────────────────────────────────
nohup python3 - >/tmp/hf_prewarm.log 2>&1 <<'PY' &
import os
from huggingface_hub import snapshot_download
repo = os.environ.get("FLUX_MODEL", "black-forest-labs/FLUX.1-schnell")
print(f"snapshot_download {repo}")
snapshot_download(repo_id=repo, max_workers=4)
print("ok")
PY
echo "[flux] HF pre-warm in background (pid=$!)"

echo "[flux] Starting FastAPI server em 0.0.0.0:8000..."
exec python3 /app/server.py
