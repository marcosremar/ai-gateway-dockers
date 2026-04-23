#!/usr/bin/env bash
# ditto startup: SSH (PUBKEY auth) + chat_server (FastAPI uvicorn :8000).
# PUBLIC_KEY env var (if set) is appended to /root/.ssh/authorized_keys
# so the ai-gateway can SSH-patch the running pod for fast iteration.
set -e

LOG_FILE="${LOG_FILE:-/tmp/container.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[ditto] $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "[ditto] GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo 'none')"
echo "[ditto] Idle timeout: ${IDLE_TIMEOUT_MIN:-15}m"
echo "[ditto] Disk /: $(df -h / | tail -1 | awk '{print $2 " total, " $4 " free"}')"

# ── SSH bootstrap ───────────────────────────────────────────────────────────
mkdir -p /root/.ssh
chmod 700 /root/.ssh
if [ -n "${PUBLIC_KEY:-}" ]; then
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    echo "[ditto] Injected PUBLIC_KEY into authorized_keys"
fi
if [ -n "${SSH_PUBLIC_KEY:-}" ]; then
    echo "$SSH_PUBLIC_KEY" >> /root/.ssh/authorized_keys
fi
chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true
ssh-keygen -A 2>/dev/null || true
/usr/sbin/sshd -D &
echo "[ditto] sshd started"

# ── Sanity checks for baked artifacts ───────────────────────────────────────
for f in /root/charlie_ref.mp4 /root/charlie_body_bg.mp4 /root/face_landmarker.task; do
    if [ ! -f "$f" ]; then
        echo "[ditto] WARN: missing $f"
    fi
done
if [ ! -d /root/ditto-talkinghead/checkpoints/ditto_trt_Ampere_Plus ]; then
    echo "[ditto] ERROR: TRT engines missing — image built incorrectly"
    ls /root/ditto-talkinghead/checkpoints/ 2>/dev/null || true
    exit 1
fi

if [ -z "${GROQ_API_KEY:-}" ]; then
    echo "[ditto] WARN: GROQ_API_KEY not set — chat_server will fail on startup"
fi

# ── Launch FastAPI chat server ──────────────────────────────────────────────
echo "[ditto] Starting uvicorn chat_server on 0.0.0.0:8000 ..."
cd /app
exec python3 -m uvicorn chat_server:app --host 0.0.0.0 --port 8000 --log-level info
