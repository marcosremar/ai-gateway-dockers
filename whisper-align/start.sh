#!/usr/bin/env bash
# whisper-align startup: SSH (PUBKEY auth) + FastAPI uvicorn :8000.
# PUBLIC_KEY env var (if set) is appended to /root/.ssh/authorized_keys
# so the ai-gateway can SSH-patch the running pod for fast iteration.
set -e

LOG_FILE="${LOG_FILE:-/tmp/container.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[whisper-align] $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "[whisper-align] GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo 'none')"
echo "[whisper-align] Model: ${WHISPERX_MODEL:-large-v3}"
echo "[whisper-align] Idle timeout: ${IDLE_TIMEOUT_MIN:-15}m"
echo "[whisper-align] Disk /: $(df -h / | tail -1 | awk '{print $2 " total, " $4 " free"}')"

if [ -d "/workspace" ]; then
    export HF_HOME=/workspace/.cache/huggingface
    mkdir -p "$HF_HOME"
    echo "[whisper-align] Using /workspace — HF_HOME=$HF_HOME"
    echo "[whisper-align] Disk /workspace: $(df -h /workspace | tail -1 | awk '{print $2 " total, " $4 " free"}')"
fi

# ── SSH bootstrap ───────────────────────────────────────────────────────────
mkdir -p /root/.ssh && chmod 700 /root/.ssh
if [ -n "${PUBLIC_KEY:-}" ]; then
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    echo "[whisper-align] Injected PUBLIC_KEY"
fi
if [ -n "${SSH_PUBLIC_KEY:-}" ]; then
    echo "$SSH_PUBLIC_KEY" >> /root/.ssh/authorized_keys
fi
chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true
ssh-keygen -A 2>/dev/null || true
/usr/sbin/sshd -D &
echo "[whisper-align] sshd started"

# ── Pre-warm HF download (background — must NOT block uvicorn) ──────────────
# server.py lazy-loads whisper + wav2vec2 alignment weights on the first
# /v1/audio/align call. We pre-warm the whisper checkpoint here so the
# first real request is fast — but never block on it: gateway must be
# able to probe /health immediately after boot, so this runs detached
# and uvicorn takes the foreground right away.
nohup python3 - >/tmp/hf_prewarm.log 2>&1 <<'PY' &
import os
from huggingface_hub import snapshot_download
# faster-whisper / WhisperX uses the Systran/Guillaume Klein conversions
# of OpenAI's whisper checkpoints (see whisperx/asr.py).
model_id_map = {
    "tiny":       "Systran/faster-whisper-tiny",
    "base":       "Systran/faster-whisper-base",
    "small":      "Systran/faster-whisper-small",
    "medium":     "Systran/faster-whisper-medium",
    "large-v2":   "Systran/faster-whisper-large-v2",
    "large-v3":   "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
}
model = os.environ.get("WHISPERX_MODEL", "large-v3")
repo = model_id_map.get(model, model)
print(f"snapshot_download {repo}")
snapshot_download(repo_id=repo, max_workers=4)
print("ok")
PY
echo "[whisper-align] HF pre-warm in background (pid=$!)"

echo "[whisper-align] Starting FastAPI server em 0.0.0.0:8000..."
exec python3 /app/server.py
