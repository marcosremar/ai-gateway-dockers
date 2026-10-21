#!/usr/bin/env bash
# storyboard-whiteboard startup: SSH (PUBKEY auth) + FastAPI uvicorn :8000.
# PUBLIC_KEY env var (if set) is appended to /root/.ssh/authorized_keys
# so the ai-gateway can SSH-patch the running pod for fast iteration.
set -e

LOG_FILE="${LOG_FILE:-/tmp/container.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[storyboard-whiteboard] $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "[storyboard-whiteboard] CPU: $(nproc) cores"
echo "[storyboard-whiteboard] Idle timeout: ${IDLE_TIMEOUT_MIN:-15}m"
echo "[storyboard-whiteboard] python: $(python3 --version 2>/dev/null || echo missing)"
echo "[storyboard-whiteboard] opencv: $(python3 -c 'import cv2; print(cv2.__version__)' 2>/dev/null || echo missing)"
echo "[storyboard-whiteboard] ffmpeg: $(ffmpeg -version 2>/dev/null | head -1 || echo missing)"

# ── SSH bootstrap ───────────────────────────────────────────────────────────
mkdir -p /root/.ssh && chmod 700 /root/.ssh
if [ -n "${PUBLIC_KEY:-}" ]; then
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    echo "[storyboard-whiteboard] Injected PUBLIC_KEY"
fi
if [ -n "${SSH_PUBLIC_KEY:-}" ]; then
    echo "$SSH_PUBLIC_KEY" >> /root/.ssh/authorized_keys
fi
chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true
ssh-keygen -A 2>/dev/null || true
/usr/sbin/sshd -D &
echo "[storyboard-whiteboard] sshd started (pid=$!)"

echo "[storyboard-whiteboard] Starting FastAPI server on 0.0.0.0:8000..."
exec python3 /app/server.py
