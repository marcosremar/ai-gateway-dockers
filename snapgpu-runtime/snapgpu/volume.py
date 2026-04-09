"""Persistent and ephemeral volume definitions."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Volume:
    """A named volume that can be mounted into containers.

    Usage:
        vol = snapgpu.Volume("model-cache")

        @app.function(volumes={"/models": vol})
        def inference(): ...
    """
    name: str
    size_gb: int = 50
    persistent: bool = True
    region: Optional[str] = None

    # Internal: assigned by the scheduler at deploy time
    _volume_id: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        if not self.name.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"Volume name must be alphanumeric (with - or _): {self.name}")
