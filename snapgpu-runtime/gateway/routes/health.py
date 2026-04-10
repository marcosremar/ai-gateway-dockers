"""Health check endpoint — reports gateway + CRIU + cuda-checkpoint readiness."""

import os
import shutil
import subprocess
from fastapi import APIRouter

router = APIRouter()

# Cache tool availability (won't change after boot)
_criu_ok: bool | None = None
_cuda_ckpt_ok: bool | None = None


def _check_criu() -> bool:
    """Return True only if criu is installed AND executable (not just in PATH).

    A binary with file capabilities set beyond the container's bounding set will
    be found by shutil.which() but raise PermissionError (rc=126) on exec.
    """
    global _criu_ok
    if _criu_ok is not None:
        return _criu_ok
    if shutil.which("criu") is None:
        _criu_ok = False
        return False
    try:
        result = subprocess.run(
            ["criu", "--version"],
            capture_output=True, timeout=5,
        )
        _criu_ok = result.returncode == 0
    except Exception:
        _criu_ok = False
    return _criu_ok


def _check_cuda_checkpoint() -> bool:
    """Check for GPU snapshot support.

    CRIU 4.0+ ships a `cuda_plugin.so` that handles GPU memory checkpointing
    internally — no separate `cuda-checkpoint` binary is needed. We check for
    the plugin first (preferred), then fall back to the standalone binary.
    """
    global _cuda_ckpt_ok
    if _cuda_ckpt_ok is not None:
        return _cuda_ckpt_ok
    if os.environ.get("SNAPGPU_DISABLE_GPU_SNAPSHOT") == "1":
        _cuda_ckpt_ok = False
        return False
    # Check for CRIU's built-in CUDA plugin (CRIU 4.0+)
    cuda_plugin_paths = [
        "/usr/lib/criu/cuda_plugin.so",
        "/usr/local/lib/criu/cuda_plugin.so",
    ]
    for p in cuda_plugin_paths:
        if os.path.isfile(p):
            _cuda_ckpt_ok = True
            return True
    # Fallback: check for standalone cuda-checkpoint binary
    _cuda_ckpt_ok = shutil.which("cuda-checkpoint") is not None
    return _cuda_ckpt_ok


def _driver_version() -> str | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            timeout=5, text=True,
        ).strip().split("\n")[0]
        return out
    except Exception:
        return None


@router.get("/health")
async def health():
    criu = _check_criu()
    cuda_ckpt = _check_cuda_checkpoint()
    driver = _driver_version()

    snapshot_capable = criu and cuda_ckpt
    return {
        "status": "ok",
        "service": "snapgpu-gateway",
        "criu": criu,
        "cuda_checkpoint": cuda_ckpt,
        "snapshot_capable": snapshot_capable,
        "driver_version": driver,
        "snapshot_dir": os.environ.get("SNAPGPU_SNAPSHOT_DIR", "/var/snapgpu/snapshots"),
    }
