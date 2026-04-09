"""GPU device allocation and tracking."""

from __future__ import annotations
import subprocess
import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GPUDevice:
    """A physical GPU device on the host."""
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int = 0
    utilization_pct: int = 0
    assigned_to: Optional[str] = None  # container_id

    @property
    def memory_free_mb(self) -> int:
        return self.memory_total_mb - self.memory_used_mb

    @property
    def available(self) -> bool:
        return self.assigned_to is None


class GPUAllocator:
    """Discovers and allocates GPU devices to containers."""

    def __init__(self):
        self._devices: dict[int, GPUDevice] = {}
        self._assignments: dict[str, int] = {}  # container_id → gpu_index

    def discover(self) -> list[GPUDevice]:
        """Detect available GPUs via nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return []

            devices = []
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 5:
                    continue
                dev = GPUDevice(
                    index=int(parts[0]),
                    name=parts[1],
                    memory_total_mb=int(parts[2]),
                    memory_used_mb=int(parts[3]),
                    utilization_pct=int(parts[4]),
                )
                devices.append(dev)
                self._devices[dev.index] = dev

            return devices
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []

    def allocate(self, container_id: str, gpu_type: Optional[str] = None,
                 memory_required_mb: int = 0) -> Optional[GPUDevice]:
        """Allocate a GPU device to a container.

        Args:
            container_id: Container to assign GPU to
            gpu_type: Preferred GPU type (e.g., "A100"), or None for any
            memory_required_mb: Minimum GPU memory required

        Returns:
            GPUDevice if allocated, None if no suitable GPU available
        """
        candidates = [
            d for d in self._devices.values()
            if d.available and d.memory_free_mb >= memory_required_mb
        ]

        if gpu_type:
            typed = [d for d in candidates if gpu_type.lower() in d.name.lower()]
            if typed:
                candidates = typed

        if not candidates:
            return None

        # Pick the GPU with the most free memory
        best = max(candidates, key=lambda d: d.memory_free_mb)
        best.assigned_to = container_id
        self._assignments[container_id] = best.index
        return best

    def release(self, container_id: str):
        """Release GPU assigned to a container."""
        idx = self._assignments.pop(container_id, None)
        if idx is not None and idx in self._devices:
            self._devices[idx].assigned_to = None

    def get_assignment(self, container_id: str) -> Optional[GPUDevice]:
        """Get the GPU assigned to a container."""
        idx = self._assignments.get(container_id)
        return self._devices.get(idx) if idx is not None else None

    @property
    def stats(self) -> dict:
        total = len(self._devices)
        available = sum(1 for d in self._devices.values() if d.available)
        return {
            "total_gpus": total,
            "available": available,
            "assigned": total - available,
            "devices": [
                {"index": d.index, "name": d.name, "available": d.available,
                 "memory_free_mb": d.memory_free_mb}
                for d in self._devices.values()
            ],
        }
