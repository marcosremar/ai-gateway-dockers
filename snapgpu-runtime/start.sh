#!/usr/bin/env bash
# snapgpu-runtime entrypoint
#
# 1. Verify CRIU is present (fail fast on broken builds).
# 2. Check cuda_plugin.so and NVIDIA driver version for GPU snapshot support.
# 3. Start the FastAPI gateway control plane.

set -euo pipefail

echo "[snapgpu-start] booting at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ── Tool checks ────────────────────────────────────────────────────────────
if ! command -v criu > /dev/null 2>&1; then
    echo "[snapgpu-start] ERROR: criu not installed in image" >&2
    exit 1
fi
echo "[snapgpu-start] criu version: $(criu --version 2>&1 | head -1)"

# Check CUDA plugin for GPU memory snapshots
if [ -f /usr/lib/criu/cuda_plugin.so ]; then
    echo "[snapgpu-start] cuda_plugin.so found — GPU snapshots supported"
else
    echo "[snapgpu-start] WARN: cuda_plugin.so not found — CPU-only snapshots"
    export SNAPGPU_DISABLE_GPU_SNAPSHOT=1
fi

# ── Driver version check ───────────────────────────────────────────────────
# CRIU's cuda_plugin requires driver 570+ (CUDA C/R API).
# We don't fail boot on a mismatch — the gateway can still serve requests
# with GPU snapshot DISABLED — but we log loudly so operators know.
if command -v nvidia-smi > /dev/null 2>&1; then
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1 || echo "unknown")
    echo "[snapgpu-start] NVIDIA driver: $DRIVER"
    DRIVER_MAJOR=$(echo "$DRIVER" | cut -d. -f1)
    if [[ "$DRIVER_MAJOR" =~ ^[0-9]+$ ]] && [[ "$DRIVER_MAJOR" -lt 570 ]]; then
        echo "[snapgpu-start] WARN: driver $DRIVER < 570, GPU snapshots disabled (CPU-only CRIU)"
        export SNAPGPU_DISABLE_GPU_SNAPSHOT=1
    fi
else
    echo "[snapgpu-start] WARN: nvidia-smi not found — running CPU-only?"
    export SNAPGPU_DISABLE_GPU_SNAPSHOT=1
fi

# ── Snapshot dir ───────────────────────────────────────────────────────────
mkdir -p "${SNAPGPU_SNAPSHOT_DIR:-/var/snapgpu/snapshots}"
mkdir -p "$(dirname "${SNAPGPU_DB_URL#sqlite:///}")" 2>/dev/null || true

# ── Launch FastAPI gateway ─────────────────────────────────────────────────
# uvicorn binds 0.0.0.0 so the ai-gateway can reach it from outside.
# Single worker — multi-worker breaks SnapshotManager state (per-process DB).
exec python3 -m uvicorn gateway.main:app \
    --host 0.0.0.0 \
    --port "${SNAPGPU_PORT:-8000}" \
    --workers 1 \
    --log-level info
