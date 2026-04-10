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

# ── CRIU capabilities ─────────────────────────────────────────────────────
# CRIU needs CAP_CHECKPOINT_RESTORE (or CAP_SYS_ADMIN) to dump processes.
# On Vast.ai/RunPod the container runs as root but without SYS_ADMIN in the
# host user namespace. We grant cap_checkpoint_restore directly to the binary
# via setcap — this works as long as we're root, requires no host namespace
# privilege, and persists for the lifetime of the container.
CRIU_BIN=$(command -v criu)
if setcap cap_checkpoint_restore+eip "$CRIU_BIN" 2>/dev/null; then
    echo "[snapgpu-start] granted cap_checkpoint_restore to $CRIU_BIN"
elif setcap cap_sys_admin+eip "$CRIU_BIN" 2>/dev/null; then
    echo "[snapgpu-start] granted cap_sys_admin to $CRIU_BIN (fallback)"
else
    echo "[snapgpu-start] WARN: could not grant capabilities to criu — snapshots may fail"
fi

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

# ── S3 snapshot pre-fetch (if configured) ─────────────────────────────────
# When SNAPGPU_S3_ENDPOINT + SNAPGPU_APP_NAME are set, download the latest
# checkpoint before uvicorn starts so the restore is instant on first request.
# This is non-blocking: if S3 is unavailable or no checkpoint exists, we just
# proceed with a cold start.
if [[ -n "${SNAPGPU_S3_ENDPOINT:-}" && -n "${SNAPGPU_APP_NAME:-}" ]]; then
    echo "[snapgpu-start] S3 configured — pre-fetching snapshot for app '${SNAPGPU_APP_NAME}'..."
    python3 - <<'PYEOF' || echo "[snapgpu-start] S3 pre-fetch skipped (non-fatal)"
import sys, os
sys.path.insert(0, '/app')
from gateway.builder.snapshot import SnapshotManager
mgr = SnapshotManager()
snap_id = mgr.download_from_s3(os.environ['SNAPGPU_APP_NAME'])
if snap_id:
    print(f"[snapgpu-start] S3 snapshot ready: {snap_id}", flush=True)
    # Write snapshot_id to a well-known file so the gateway can read it
    with open('/var/snapgpu/s3-snapshot-id', 'w') as f:
        f.write(snap_id)
else:
    print("[snapgpu-start] No S3 snapshot found — cold start", flush=True)
PYEOF
fi

# ── Launch FastAPI gateway ─────────────────────────────────────────────────
# uvicorn binds 0.0.0.0 so the ai-gateway can reach it from outside.
# Single worker — multi-worker breaks SnapshotManager state (per-process DB).
exec python3 -m uvicorn gateway.main:app \
    --host 0.0.0.0 \
    --port "${SNAPGPU_PORT:-8000}" \
    --workers 1 \
    --log-level info
