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
    """Return True only if criu is installed, executable, AND the process has the
    required capabilities in its bounding set.

    IMPORTANT: `criu check` checks KERNEL features (namespace support, etc.), NOT
    process capabilities. A container without CAP_CHECKPOINT_RESTORE passes `criu check`
    but fails `criu dump` with "needs CAP_SYS_ADMIN or CAP_CHECKPOINT_RESTORE".

    We read /proc/self/status CapBnd directly:
      CAP_SYS_ADMIN = bit 21       (granted by --privileged)
      CAP_CHECKPOINT_RESTORE = bit 40  (granted by --cap-add CHECKPOINT_RESTORE)
    Either bit being set in CapBnd means CRIU can checkpoint/restore.
    """
    global _criu_ok
    if _criu_ok is not None:
        return _criu_ok
    if shutil.which("criu") is None:
        _criu_ok = False
        return False
    try:
        # Verify the binary is executable (catches broken setcap giving rc=126)
        ver = subprocess.run(["criu", "--version"], capture_output=True, timeout=5)
        if ver.returncode != 0:
            _criu_ok = False
            return False
        # Check process capability bounding set directly from /proc
        # `criu check` checks kernel features, NOT process caps — don't use it.
        cap_bnd = 0
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("CapBnd:"):
                    cap_bnd = int(line.split(":")[1].strip(), 16)
                    break
        CAP_SYS_ADMIN = 21
        CAP_CHECKPOINT_RESTORE = 40
        has_cap = bool(cap_bnd & (1 << CAP_CHECKPOINT_RESTORE)) or bool(cap_bnd & (1 << CAP_SYS_ADMIN))
        _criu_ok = has_cap
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
