#!/usr/bin/env bash
set -euo pipefail

# Entrypoint for the fbx2glb container. Keeps things tiny: just
# exec uvicorn on the FastAPI app. The idle watchdog runs inside
# the server via its lifespan task; it exits the process after
# IDLE_TIMEOUT_MIN minutes of inactivity so the ai-gateway / Fly.io
# can reclaim the machine.
PORT="${PORT:-8000}"
exec python3 -m uvicorn server:app \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --log-level info \
  --no-access-log
