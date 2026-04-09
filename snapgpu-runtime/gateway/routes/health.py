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
    global _criu_ok
    if _criu_ok is not None:
        return _criu_ok
    _criu_ok = shutil.which("criu") is not None
    return _criu_ok


def _check_cuda_checkpoint() -> bool:
    global _cuda_ckpt_ok
    if _cuda_ckpt_ok is not None:
        return _cuda_ckpt_ok
    if os.environ.get("SNAPGPU_DISABLE_GPU_SNAPSHOT") == "1":
        _cuda_ckpt_ok = False
        return False
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
