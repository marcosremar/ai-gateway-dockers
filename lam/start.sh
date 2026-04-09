#!/usr/bin/env bash
# LAM startup script — mirrors di360/kokoro-tts pattern

LOG_FILE="${LOG_FILE:-/tmp/container.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[lam] ─────────────────────────────────────────────"
echo "[lam] Large Avatar Model (SIGGRAPH 2025)"
echo "[lam] $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "[lam] GPU:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none — CPU mode')"
echo "[lam] Model: ${MODEL_ID:-3DAIGC/LAM-20K}"
echo "[lam] Disk:  $(df -h / | tail -1 | awk '{print $2 " total, " $4 " free"}')"
echo "[lam] ─────────────────────────────────────────────"

exec python3 /app/server.py
