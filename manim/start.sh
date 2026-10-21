#!/usr/bin/env bash
# manim startup: SSH (PUBKEY auth) + FastAPI uvicorn :8000.
# PUBLIC_KEY env var (if set) is appended to /root/.ssh/authorized_keys
# so the ai-gateway can SSH-patch the running pod for fast iteration.
set -e

LOG_FILE="${LOG_FILE:-/tmp/container.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[manim] $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "[manim] CPU: $(nproc) cores"
echo "[manim] Idle timeout: ${IDLE_TIMEOUT_MIN:-15}m"
echo "[manim] manim version: $(manim --version 2>/dev/null || echo 'manim missing')"
echo "[manim] ffmpeg version: $(ffmpeg -version 2>/dev/null | head -1 || echo 'ffmpeg missing')"

# ── SSH bootstrap ───────────────────────────────────────────────────────────
mkdir -p /root/.ssh && chmod 700 /root/.ssh
if [ -n "${PUBLIC_KEY:-}" ]; then
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    echo "[manim] Injected PUBLIC_KEY"
fi
if [ -n "${SSH_PUBLIC_KEY:-}" ]; then
    echo "$SSH_PUBLIC_KEY" >> /root/.ssh/authorized_keys
fi
chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true
ssh-keygen -A 2>/dev/null || true
/usr/sbin/sshd -D &
echo "[manim] sshd started (pid=$!)"

echo "[manim] Starting FastAPI server on 0.0.0.0:8000..."
exec python3 /app/server.py
