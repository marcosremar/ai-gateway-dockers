"""GPU type definitions and resource configuration."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GPU:
    """GPU resource specification."""
    name: str
    memory_gb: int
    count: int = 1

    def __str__(self) -> str:
        suffix = f"x{self.count}" if self.count > 1 else ""
        return f"{self.name}{suffix}"


# Preset GPU types (shortcuts)
class gpu:
    """GPU presets — use as `gpu.T4`, `gpu.A100`, etc."""
    T4 = GPU("T4", memory_gb=16)
    L4 = GPU("L4", memory_gb=24)
    A10G = GPU("A10G", memory_gb=24)
    L40S = GPU("L40S", memory_gb=48)
    A100 = GPU("A100", memory_gb=80)
    A100_40 = GPU("A100-40GB", memory_gb=40)
    H100 = GPU("H100", memory_gb=80)
    RTX_4090 = GPU("RTX4090", memory_gb=24)
    RTX_5090 = GPU("RTX5090", memory_gb=32)

    @staticmethod
    def any(memory_gb: int = 16, count: int = 1) -> GPU:
        """Request any GPU with at least this much memory."""
        return GPU(name="any", memory_gb=memory_gb, count=count)


def parse_gpu(spec: str | GPU | None) -> GPU | None:
    """Parse a GPU specification from string or GPU object."""
    if spec is None:
        return None
    if isinstance(spec, GPU):
        return spec
    # String shortcut: "T4", "A100", "H100", "A100:2" (count)
    parts = spec.split(":")
    name = parts[0].upper().replace(" ", "")
    count = int(parts[1]) if len(parts) > 1 else 1
    presets = {
        "T4": gpu.T4, "L4": gpu.L4, "A10G": gpu.A10G,
        "L40S": gpu.L40S, "A100": gpu.A100, "H100": gpu.H100,
        "RTX4090": gpu.RTX_4090, "RTX5090": gpu.RTX_5090,
    }
    base = presets.get(name)
    if base:
        return GPU(name=base.name, memory_gb=base.memory_gb, count=count) if count != 1 else base
    return GPU(name=spec, memory_gb=16, count=count)
