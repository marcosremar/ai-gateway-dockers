#!/bin/bash
# gpu-dev startup: SSH + health server.
# Authorized keys (for ssh) are expected to be injected via Vast.ai's
# default mechanism (appended to /root/.ssh/authorized_keys at boot).

set -e

# Ensure SSH host keys exist
ssh-keygen -A 2>/dev/null || true

# Start SSH daemon
/usr/sbin/sshd -D &

# Start health server in foreground (keeps container alive)
exec python3 /app/health_server.py
