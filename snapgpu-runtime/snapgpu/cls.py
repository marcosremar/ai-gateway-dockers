"""Stateful class support (@app.cls) with enter/exit lifecycle."""

from __future__ import annotations
from typing import Any, Callable, Optional
from .gpu import GPU, parse_gpu
from .image import Image
from .volume import Volume


class enter:
    """Decorator for class lifecycle methods.

    @enter(snap=True) — runs BEFORE snapshot (heavy init: load model, compile)
    @enter(snap=False) — runs AFTER snapshot restore (light init: reconnect DB)
    @exit() — runs on container shutdown
    """
    def __init__(self, *, snap: bool = True):
        self.snap = snap

    def __call__(self, fn: Callable) -> Callable:
        fn._snapgpu_enter = {"snap": self.snap}
        return fn


class exit_handler:
    """Decorator for cleanup on container shutdown."""
    def __call__(self, fn: Callable) -> Callable:
        fn._snapgpu_exit = True
        return fn


# Singleton alias
exit = exit_handler


class Cls:
    """Metadata wrapper for a SnapGPU class definition.

    Created by @app.cls(). Tracks the class, its lifecycle methods,
    and its resource requirements.
    """
    def __init__(
        self,
        cls: type,
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
    ):
        self.cls = cls
        self.image = image
        self.gpu = parse_gpu(gpu)
        self.memory = memory
        self.timeout = timeout
        self.keep_warm = keep_warm
        self.max_containers = max_containers
        self.container_idle_timeout = container_idle_timeout
        self.enable_memory_snapshot = enable_memory_snapshot
        self.volumes = volumes or {}

        # Discover lifecycle methods
        self.snap_true_methods: list[str] = []
        self.snap_false_methods: list[str] = []
        self.exit_methods: list[str] = []
        self.methods: list[str] = []

        for name in dir(cls):
            if name.startswith("_"):
                continue
            attr = getattr(cls, name, None)
            if not callable(attr):
                continue
            if hasattr(attr, "_snapgpu_enter"):
                if attr._snapgpu_enter["snap"]:
                    self.snap_true_methods.append(name)
                else:
                    self.snap_false_methods.append(name)
            elif hasattr(attr, "_snapgpu_exit"):
                self.exit_methods.append(name)
            else:
                self.methods.append(name)

    def to_spec(self) -> dict:
        """Serialize to deployable spec."""
        return {
            "class_name": self.cls.__name__,
            "module": self.cls.__module__,
            "gpu": str(self.gpu) if self.gpu else None,
            "memory": self.memory,
            "timeout": self.timeout,
            "keep_warm": self.keep_warm,
            "max_containers": self.max_containers,
            "container_idle_timeout": self.container_idle_timeout,
            "enable_memory_snapshot": self.enable_memory_snapshot,
            "volumes": {path: vol.name for path, vol in self.volumes.items()},
            "snap_true_methods": self.snap_true_methods,
            "snap_false_methods": self.snap_false_methods,
            "exit_methods": self.exit_methods,
            "methods": self.methods,
        }

    def __repr__(self) -> str:
        return f"Cls({self.cls.__name__}, gpu={self.gpu}, methods={self.methods})"
