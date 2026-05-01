#!/bin/bash
# ── hunyuan3d entrypoint ─────────────────────────────────────────────────────
# Same dual-process pattern as trellis2: launch sshd in the background for
# Vast.ai SSH tunnel access, then exec the FastAPI server in the foreground
# so the container lifecycle follows it.

set -x

mkdir -p /var/log
touch /var/log/app.log
exec > >(tee -a /var/log/app.log) 2>&1

echo "===== PHASE: BOOT ====="
echo "[hunyuan3d] start.sh invoked at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[hunyuan3d] hostname=$(hostname) uid=$(id -u) pwd=$(pwd)"

echo "===== PHASE: SSH KEY INSTALL ====="
if [ -n "$PUBLIC_KEY" ]; then
  echo "[hunyuan3d] Installing PUBLIC_KEY into authorized_keys"
  mkdir -p /root/.ssh
  echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
  chmod 700 /root/.ssh
  chmod 600 /root/.ssh/authorized_keys
fi
if [ -n "$SSH_PUBLIC_KEY" ]; then
  echo "[hunyuan3d] Installing SSH_PUBLIC_KEY into authorized_keys"
  mkdir -p /root/.ssh
  echo "$SSH_PUBLIC_KEY" >> /root/.ssh/authorized_keys
  chmod 700 /root/.ssh
  chmod 600 /root/.ssh/authorized_keys
fi

echo "===== PHASE: SSH HOST KEYS ====="
if [ ! -f /etc/ssh/ssh_host_ed25519_key ]; then
  ssh-keygen -A || echo "[hunyuan3d] WARN: ssh-keygen -A failed"
fi

echo "===== PHASE: SSHD CONFIG ====="
sed -i 's/^#*PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config || true
sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config || true
sed -i 's/^#*PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config || true

echo "===== PHASE: SSHD START ====="
/usr/sbin/sshd -D &
SSHD_PID=$!
echo "[hunyuan3d] sshd started (pid=$SSHD_PID)"
sleep 2
if ! kill -0 $SSHD_PID 2>/dev/null; then
  echo "[hunyuan3d] ERROR: sshd died after start"
fi

echo "===== PHASE: PYTHON PREFLIGHT ====="
if [ ! -f /app/server.py ]; then
  echo "[hunyuan3d] ERROR: /app/server.py missing!"
  ls -la /app/ || true
fi
python --version || echo "[hunyuan3d] WARN: python --version failed"
python -c "import torch; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()}')" \
  || echo "[hunyuan3d] WARN: torch import failed"

echo "===== PHASE: LAUNCH FASTAPI ====="
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
exec python -u /app/server.py
