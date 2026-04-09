"""HTTP endpoint decorator for SnapGPU functions."""

from __future__ import annotations
from typing import Callable, Optional


def fastapi_endpoint(
    fn: Optional[Callable] = None,
    *,
    method: str = "POST",
    path: Optional[str] = None,
    docs: bool = True,
):
    """Decorator to expose a SnapGPU function as an HTTP endpoint.

    Usage:
        @app.function(gpu="T4")
        @snapgpu.fastapi_endpoint(method="POST")
        def predict(data: dict) -> dict:
            return {"result": model(data["input"])}

    The function becomes callable at:
        POST /v1/invoke/{app_name}/{fn_name}
    """
    def decorator(func: Callable) -> Callable:
        func._snapgpu_endpoint = {
            "method": method.upper(),
            "path": path,
            "docs": docs,
        }
        return func

    if fn is not None:
        return decorator(fn)
    return decorator
