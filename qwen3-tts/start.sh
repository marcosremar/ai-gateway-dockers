#!/usr/bin/env bash
# qwen3-tts startup: SSH (PUBKEY auth) + FastAPI uvicorn :8000.
# PUBLIC_KEY env var (if set) is appended to /root/.ssh/authorized_keys
# so the ai-gateway can SSH-patch the running pod for fast iteration.
set -e

LOG_FILE="${LOG_FILE:-/tmp/container.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[qwen3-tts] $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "[qwen3-tts] GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo 'none')"
echo "[qwen3-tts] Model: ${QWEN3_TTS_MODEL:-Qwen/Qwen3-TTS}"
echo "[qwen3-tts] Idle timeout: ${IDLE_TIMEOUT_MIN:-15}m"
echo "[qwen3-tts] Disk /: $(df -h / | tail -1 | awk '{print $2 " total, " $4 " free"}')"

if [ -d "/workspace" ]; then
    export HF_HOME=/workspace/.cache/huggingface
    mkdir -p "$HF_HOME"
    echo "[qwen3-tts] Using /workspace — HF_HOME=$HF_HOME"
    echo "[qwen3-tts] Disk /workspace: $(df -h /workspace | tail -1 | awk '{print $2 " total, " $4 " free"}')"
fi

# ── SSH bootstrap ───────────────────────────────────────────────────────────
mkdir -p /root/.ssh && chmod 700 /root/.ssh
if [ -n "${PUBLIC_KEY:-}" ]; then
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    echo "[qwen3-tts] Injected PUBLIC_KEY"
fi
if [ -n "${SSH_PUBLIC_KEY:-}" ]; then
    echo "$SSH_PUBLIC_KEY" >> /root/.ssh/authorized_keys
fi
chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true
ssh-keygen -A 2>/dev/null || true
/usr/sbin/sshd -D &
echo "[qwen3-tts] sshd started"

# ── Pre-warm HF download (background — must NOT block uvicorn) ──────────────
# server.py lazy-loads weights on first /health call; pre-warming in
# parallel avoids a long first-request stall but the gateway must be
# able to probe /health immediately after boot, so we run it in the
# background and let the foreground hand off to uvicorn right away.
nohup python3 - >/tmp/hf_prewarm.log 2>&1 <<'PY' &
import os
from huggingface_hub import snapshot_download
for env_key, default in [("QWEN3_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
                          ("QWEN3_TTS_TOKENIZER", "Qwen/Qwen3-TTS-Tokenizer-12Hz")]:
    repo = os.environ.get(env_key, default)
    print(f"snapshot_download {repo}")
    snapshot_download(repo_id=repo, max_workers=4)
print("ok")
PY
echo "[qwen3-tts] HF pre-warm in background (pid=$!)"

echo "[qwen3-tts] Starting FastAPI server em 0.0.0.0:8000..."
exec python3 /app/server.py
