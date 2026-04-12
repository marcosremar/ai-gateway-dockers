#!/bin/bash
set -x
mkdir -p /var/log
touch /var/log/app.log
exec > >(tee -a /var/log/app.log) 2>&1

echo "===== UniRig Server ====="
echo "[unirig] start.sh at $(date -u)"

# SSH setup
if [ -n "$PUBLIC_KEY" ]; then
  mkdir -p /root/.ssh
  echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
  chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys
fi
if [ -f /etc/ssh/sshd_config ]; then
  sed -i 's/^#*PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
  /usr/sbin/sshd -D &
  echo "[unirig] sshd started"
fi

echo "[unirig] Python: $(python --version)"
echo "[unirig] CUDA: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")')"

export PYTHONUNBUFFERED=1
exec python -u /app/server.py
