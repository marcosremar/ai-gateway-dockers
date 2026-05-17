#!/bin/bash
# unreal-desktop startup:
# 1. Run gezp base entrypoint in background (XFCE + NoMachine + noVNC).
# 2. Start sshd (for ai-gateway dev mode + raw SSH fallback).
# 3. Start health server in foreground (keeps container alive, idle watchdog).

set -e

ssh-keygen -A 2>/dev/null || true
/usr/sbin/sshd -D &

if [ -x /startup.sh ]; then
  /startup.sh &
elif [ -x /entrypoint.sh ]; then
  /entrypoint.sh &
fi

exec python3 /app/health_server.py
