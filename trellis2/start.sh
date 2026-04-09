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
# The FastAPI server runs in the foreground (PID 1) so the container
# lifecycle follows it. sshd is backgrounded.

set -e

echo "[trellis2] Starting container..."

# Set up SSH if Vast.ai injected a public key
if [ -n "$PUBLIC_KEY" ]; then
  echo "[trellis2] Installing PUBLIC_KEY into authorized_keys"
  mkdir -p /root/.ssh
  echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
  chmod 700 /root/.ssh
  chmod 600 /root/.ssh/authorized_keys
fi

if [ -n "$SSH_PUBLIC_KEY" ]; then
  echo "[trellis2] Installing SSH_PUBLIC_KEY into authorized_keys"
  mkdir -p /root/.ssh
  echo "$SSH_PUBLIC_KEY" >> /root/.ssh/authorized_keys
  chmod 700 /root/.ssh
  chmod 600 /root/.ssh/authorized_keys
fi

# Generate SSH host keys if missing (first boot)
if [ ! -f /etc/ssh/ssh_host_ed25519_key ]; then
  echo "[trellis2] Generating SSH host keys..."
  ssh-keygen -A
fi

# Allow root login via key (Vast hosts use root)
sed -i 's/^#*PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/^#*PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config

# Start sshd in background
echo "[trellis2] Starting sshd on port 22..."
/usr/sbin/sshd -D &
SSHD_PID=$!
echo "[trellis2] sshd started (pid=$SSHD_PID)"

# Verify sshd is alive
sleep 1
if ! kill -0 $SSHD_PID 2>/dev/null; then
  echo "[trellis2] WARN: sshd died after start"
fi

# Start the FastAPI server (foreground, PID 1 substitute)
echo "[trellis2] Starting FastAPI server on port 8000..."
exec python /app/server.py
