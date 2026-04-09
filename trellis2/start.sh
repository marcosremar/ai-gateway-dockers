#!/bin/bash
# ── trellis2 entrypoint ──────────────────────────────────────────────────────
# Starts both:
#   1. sshd on port 22 (for Vast.ai SSH tunnel access)
#   2. FastAPI server on port 8000 (the actual TRELLIS.2 inference API)
#
# Vast.ai injects the user's registered SSH public keys into
# /root/.ssh/authorized_keys at boot time via the PUBLIC_KEY env var. We
# also accept keys via the standard authorized_keys file for direct Docker
# runs (`docker run -v ~/.ssh/authorized_keys:/root/.ssh/authorized_keys`).
#
# The FastAPI server runs in the foreground (PID 1 substitute) so the
# container lifecycle follows it. sshd is backgrounded.
#
# NOTE: We intentionally do NOT use `set -e` — a single transient failure
# (e.g. missing host key dir) should not take down the whole container and
# lock the user out of SSH debugging. Instead each phase logs loudly and we
# soldier on so the operator can always SSH in.

set -x

# Ensure the log file exists from the very first moment so an operator can
# `tail -f /var/log/app.log` immediately after connecting via SSH, even if
# something explodes a few lines below.
mkdir -p /var/log
touch /var/log/app.log

# Mirror ALL stdout/stderr from this script into /var/log/app.log while
# still writing to the original streams (so `docker logs` also works).
# This is a process-substitution tee: file descriptors 1 and 2 are
# redirected into `tee -a`, which appends to the log AND echoes to the
# original stdout/stderr that the container runtime captures.
exec > >(tee -a /var/log/app.log) 2>&1

echo "===== PHASE: BOOT ====="
echo "[trellis2] start.sh invoked at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[trellis2] hostname=$(hostname) uid=$(id -u) pwd=$(pwd)"
echo "[trellis2] PATH=$PATH"

echo "===== PHASE: SSH KEY INSTALL ====="
# Set up SSH if Vast.ai (or the user) injected a public key via env var
if [ -n "$PUBLIC_KEY" ]; then
  echo "[trellis2] Installing PUBLIC_KEY into authorized_keys"
  mkdir -p /root/.ssh
  echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
  chmod 700 /root/.ssh
  chmod 600 /root/.ssh/authorized_keys
else
  echo "[trellis2] PUBLIC_KEY not set, skipping"
fi

if [ -n "$SSH_PUBLIC_KEY" ]; then
  echo "[trellis2] Installing SSH_PUBLIC_KEY into authorized_keys"
  mkdir -p /root/.ssh
  echo "$SSH_PUBLIC_KEY" >> /root/.ssh/authorized_keys
  chmod 700 /root/.ssh
  chmod 600 /root/.ssh/authorized_keys
else
  echo "[trellis2] SSH_PUBLIC_KEY not set, skipping"
fi

echo "===== PHASE: SSH HOST KEYS ====="
# Generate SSH host keys if missing (first boot)
if [ ! -f /etc/ssh/ssh_host_ed25519_key ]; then
  echo "[trellis2] Generating SSH host keys..."
  ssh-keygen -A || echo "[trellis2] WARN: ssh-keygen -A failed"
else
  echo "[trellis2] Host keys already present"
fi

echo "===== PHASE: SSHD CONFIG ====="
# Allow root login via key (Vast hosts use root)
sed -i 's/^#*PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config || echo "[trellis2] WARN: sed PermitRootLogin failed"
sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config || echo "[trellis2] WARN: sed PasswordAuthentication failed"
sed -i 's/^#*PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config || echo "[trellis2] WARN: sed PubkeyAuthentication failed"

echo "===== PHASE: SSHD START ====="
echo "[trellis2] Starting sshd on port 22..."
/usr/sbin/sshd -D &
SSHD_PID=$!
echo "[trellis2] sshd started (pid=$SSHD_PID)"

# Give sshd a moment to bind then verify it's alive and listening
sleep 2
if ! kill -0 $SSHD_PID 2>/dev/null; then
  echo "[trellis2] ERROR: sshd process died after start (pid=$SSHD_PID)"
fi

echo "[trellis2] Verifying sshd is listening on :22..."
if command -v ss >/dev/null 2>&1; then
  ss -tlnp | grep :22 || echo "[trellis2] WARN: ss did not see anything on :22"
elif command -v netstat >/dev/null 2>&1; then
  netstat -tlnp | grep :22 || echo "[trellis2] WARN: netstat did not see anything on :22"
else
  echo "[trellis2] WARN: neither ss nor netstat available to verify sshd"
fi

echo "===== PHASE: PYTHON PREFLIGHT ====="
# Verify the server file exists before we try to launch it
if [ ! -f /app/server.py ]; then
  echo "[trellis2] ERROR: /app/server.py does not exist!"
  ls -la /app/ || true
fi

echo "[trellis2] Python version:"
python --version || echo "[trellis2] WARN: python --version failed"

echo "[trellis2] Import test: sys.path..."
python -c "import sys; sys.path.insert(0, '/app/trellis2'); print('sys.path OK')" \
  || echo "[trellis2] WARN: sys.path import test failed"

echo "[trellis2] Import test: torch..."
python -c "import torch; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()}')" \
  || echo "[trellis2] WARN: torch import test failed"

echo "===== PHASE: LAUNCH FASTAPI ====="
echo "[trellis2] Launching FastAPI server on port 8000..."
# Use -u for unbuffered output so logs appear in real time. We also force
# PYTHONUNBUFFERED=1 and PYTHONFAULTHANDLER=1 for belt-and-suspenders:
#   - PYTHONUNBUFFERED=1: redundant with -u but also covers child processes
#   - PYTHONFAULTHANDLER=1: dumps Python tracebacks on SIGSEGV/SIGFPE so we
#     don't lose tracebacks if the model load segfaults inside a c-ext
# Note: this whole script's stdout/stderr are already being teed into
# /var/log/app.log by the `exec > >(tee -a ...)` line at the top. Adding
# another `| tee` here would create a SECOND pipe in front of python and
# break the exit-code propagation. So we just inherit the parent script's
# already-tee'd stdout/stderr.
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
exec python -u /app/server.py
