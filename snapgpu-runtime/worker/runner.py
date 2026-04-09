"""Function executor — runs inside the GPU container.

Receives serialized functions via HTTP, executes them with GPU access,
and returns serialized results. Also manages the snapshot lifecycle.
"""

from __future__ import annotations
import os
import time
import traceback
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    raise ImportError("fastapi and uvicorn required: pip install fastapi uvicorn")


app = FastAPI(title="SnapGPU Worker")

# Worker state
_instance: Optional[object] = None  # For @app.cls() stateful classes
_ready = False
_boot_time = time.time()
_request_count = 0


class ExecuteRequest(BaseModel):
    fn_data: str       # Base64 cloudpickle'd function
    args_data: str     # Base64 cloudpickle'd (args, kwargs)


class ExecuteResponse(BaseModel):
    status: str
    result: Optional[str] = None  # Base64 pickled result
    error: Optional[str] = None
    latency_ms: float = 0


@app.get("/health")
async def health():
    return {
        "status": "ready" if _ready else "loading",
        "uptime_sec": int(time.time() - _boot_time),
        "request_count": _request_count,
        "gpu_available": _check_gpu(),
    }


@app.post("/execute")
async def execute(req: ExecuteRequest) -> ExecuteResponse:
    """Execute a serialized function with serialized arguments."""
    global _request_count
    _request_count += 1
    start = time.time()

    try:
        from snapgpu.serialization import decode_b64, deserialize_function, deserialize_args, serialize_result, encode_b64

        fn = deserialize_function(decode_b64(req.fn_data))
        args, kwargs = deserialize_args(decode_b64(req.args_data))

        result = fn(*args, **kwargs)

        return ExecuteResponse(
            status="completed",
            result=encode_b64(serialize_result(result)),
            latency_ms=(time.time() - start) * 1000,
        )
    except Exception as e:
        return ExecuteResponse(
            status="error",
            error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
            latency_ms=(time.time() - start) * 1000,
        )


@app.post("/init-class")
async def init_class(req: ExecuteRequest) -> ExecuteResponse:
    """Initialize a stateful class instance and run @enter methods."""
    global _instance, _ready
    start = time.time()

    try:
        from snapgpu.serialization import decode_b64, deserialize_function

        cls_factory = deserialize_function(decode_b64(req.fn_data))
        _instance = cls_factory()

        # Run @enter(snap=True) methods (heavy init)
        for name in dir(_instance):
            method = getattr(_instance, name, None)
            if callable(method) and hasattr(method, "_snapgpu_enter"):
                if method._snapgpu_enter.get("snap", True):
                    print(f"[worker] Running @enter(snap=True): {name}")
                    method()

        _ready = True
        return ExecuteResponse(
            status="initialized",
            latency_ms=(time.time() - start) * 1000,
        )
    except Exception as e:
        return ExecuteResponse(
            status="error",
            error=str(e),
            latency_ms=(time.time() - start) * 1000,
        )


@app.post("/post-restore")
async def post_restore() -> dict:
    """Called after snapshot restore to run @enter(snap=False) methods."""
    global _ready

    if _instance is None:
        return {"status": "error", "message": "No class instance initialized"}

    for name in dir(_instance):
        method = getattr(_instance, name, None)
        if callable(method) and hasattr(method, "_snapgpu_enter"):
            if not method._snapgpu_enter.get("snap", True):
                print(f"[worker] Running @enter(snap=False): {name}")
                method()

    _ready = True
    return {"status": "restored"}


def _check_gpu() -> bool:
    """Quick check if GPU is available."""
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def start():
    """Worker entry point."""
    global _ready
    _ready = True
    port = int(os.environ.get("WORKER_PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    start()
