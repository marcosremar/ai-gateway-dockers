"""SnapGPU App — the core interface for defining GPU serverless functions."""

from __future__ import annotations
import asyncio
from typing import Any, Callable, Optional
from .gpu import GPU, parse_gpu
from .image import Image
from .volume import Volume
from .cls import Cls
from .serialization import serialize_function, serialize_args, encode_b64

# Default gateway URL (overridden by SNAPGPU_GATEWAY_URL env)
import os
_GATEWAY_URL = os.environ.get("SNAPGPU_GATEWAY_URL", "http://localhost:8000")


class FunctionHandle:
    """A handle to a registered SnapGPU function.

    Supports:
        result = fn.remote(*args, **kwargs)     # Sync remote call
        result = await fn.remote_async(...)      # Async remote call
        result = fn.local(*args, **kwargs)       # Local execution
        future = fn.spawn(*args, **kwargs)       # Fire-and-forget
    """

    def __init__(self, app_name: str, fn_name: str, fn: Callable, spec: dict):
        self._app_name = app_name
        self._fn_name = fn_name
        self._fn = fn
        self._spec = spec
        # Copy original function metadata
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        self.__module__ = fn.__module__

    def local(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the function locally (no container, no GPU routing)."""
        return self._fn(*args, **kwargs)

    def remote(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the function remotely on a GPU container (blocking)."""
        return asyncio.get_event_loop().run_until_complete(
            self.remote_async(*args, **kwargs)
        )

    async def remote_async(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the function remotely (async)."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required for remote calls: pip install httpx")

        payload = {
            "fn_data": encode_b64(serialize_function(self._fn)),
            "args_data": encode_b64(serialize_args(args, kwargs)),
        }

        async with httpx.AsyncClient(timeout=self._spec.get("timeout", 300)) as client:
            resp = await client.post(
                f"{_GATEWAY_URL}/v1/invoke/{self._app_name}/{self._fn_name}",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        if data.get("status") == "error":
            raise RuntimeError(f"Remote execution failed: {data.get('error')}")

        # Deserialize result
        from .serialization import decode_b64, deserialize_result
        return deserialize_result(decode_b64(data["result"]))

    async def spawn(self, *args: Any, **kwargs: Any) -> str:
        """Fire-and-forget: submit for async execution, return task ID."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required for remote calls: pip install httpx")

        payload = {
            "fn_data": encode_b64(serialize_function(self._fn)),
            "args_data": encode_b64(serialize_args(args, kwargs)),
            "async": True,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{_GATEWAY_URL}/v1/invoke/{self._app_name}/{self._fn_name}",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()["task_id"]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Direct call = local execution."""
        return self.local(*args, **kwargs)

    def __repr__(self) -> str:
        gpu_str = self._spec.get("gpu", "cpu")
        return f"FunctionHandle({self._app_name}/{self._fn_name}, gpu={gpu_str})"


class App:
    """A SnapGPU application that groups functions and classes.

    Usage:
        app = snapgpu.App("my-service")

        @app.function(gpu="A100", image=my_image)
        def generate(prompt: str) -> str:
            ...
    """

    def __init__(self, name: str, *, gateway_url: Optional[str] = None):
        self.name = name
        self._gateway_url = gateway_url or _GATEWAY_URL
        self._functions: dict[str, FunctionHandle] = {}
        self._classes: dict[str, Cls] = {}
        self._default_image: Optional[Image] = None

    def function(
        self,
        fn: Optional[Callable] = None,
        *,
        image: Optional[Image] = None,
        gpu: str | GPU | None = None,
        memory: int = 2048,
        timeout: int = 300,
        keep_warm: int = 0,
        max_containers: int = 10,
        container_idle_timeout: int = 300,
        enable_memory_snapshot: bool = False,
        volumes: Optional[dict[str, Volume]] = None,
    ) -> Callable:
        """Register a function for GPU-accelerated remote execution.

        Can be used as @app.function or @app.function(gpu="A100").
        """
        def decorator(func: Callable) -> FunctionHandle:
            parsed_gpu = parse_gpu(gpu)
            spec = {
                "gpu": str(parsed_gpu) if parsed_gpu else None,
                "memory": memory,
                "timeout": timeout,
                "keep_warm": keep_warm,
                "max_containers": max_containers,
                "container_idle_timeout": container_idle_timeout,
                "enable_memory_snapshot": enable_memory_snapshot,
                "volumes": {path: vol.name for path, vol in (volumes or {}).items()},
                "image": image.to_dockerfile() if image else None,
            }
            handle = FunctionHandle(self.name, func.__name__, func, spec)
            self._functions[func.__name__] = handle
            return handle

        if fn is not None:
            return decorator(fn)
        return decorator

    def cls(
        self,
        cls_or_none: Optional[type] = None,
        *,
        image: Optional[Image] = None,
        gpu: str | GPU | None = None,
        memory: int = 2048,
        timeout: int = 300,
        keep_warm: int = 0,
        max_containers: int = 10,
        container_idle_timeout: int = 300,
        enable_memory_snapshot: bool = True,
        volumes: Optional[dict[str, Volume]] = None,
    ) -> Callable:
        """Register a stateful class for GPU execution with lifecycle hooks.

        Usage:
            @app.cls(gpu="A100", enable_memory_snapshot=True)
            class Model:
                @snapgpu.enter(snap=True)
                def load_model(self):
                    self.model = load_heavy_model()

                @snapgpu.enter(snap=False)
                def reconnect(self):
                    self.db = connect_db()

                def predict(self, text: str) -> str:
                    return self.model(text)
        """
        def decorator(cls_type: type) -> Cls:
            wrapper = Cls(
                cls_type,
                image=image,
                gpu=gpu,
                memory=memory,
                timeout=timeout,
                keep_warm=keep_warm,
                max_containers=max_containers,
                container_idle_timeout=container_idle_timeout,
                enable_memory_snapshot=enable_memory_snapshot,
                volumes=volumes,
            )
            self._classes[cls_type.__name__] = wrapper
            return wrapper

        if cls_or_none is not None:
            return decorator(cls_or_none)
        return decorator

    def to_spec(self) -> dict:
        """Serialize the entire app to a deployable specification."""
        return {
            "name": self.name,
            "functions": {
                name: handle._spec for name, handle in self._functions.items()
            },
            "classes": {
                name: cls.to_spec() for name, cls in self._classes.items()
            },
        }

    @property
    def registered_functions(self) -> list[str]:
        return list(self._functions.keys())

    @property
    def registered_classes(self) -> list[str]:
        return list(self._classes.keys())

    def __repr__(self) -> str:
        return f"App({self.name!r}, functions={self.registered_functions}, classes={self.registered_classes})"
