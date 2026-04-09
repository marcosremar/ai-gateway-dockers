"""Health check utilities for the worker container."""

from __future__ import annotations
import subprocess
from typing import Optional


def check_gpu() -> dict:
    """Get GPU health status."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return {"available": False, "error": result.stderr.strip()}

        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "name": parts[0],
                    "memory_used_mb": int(parts[1]),
                    "memory_total_mb": int(parts[2]),
                    "utilization_pct": int(parts[3]),
                    "temperature_c": int(parts[4]),
                })

        return {"available": True, "gpus": gpus}
    except FileNotFoundError:
        return {"available": False, "error": "nvidia-smi not found"}
    except subprocess.TimeoutExpired:
        return {"available": False, "error": "nvidia-smi timeout"}
