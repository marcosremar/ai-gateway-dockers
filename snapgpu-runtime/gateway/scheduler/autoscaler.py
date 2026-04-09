"""Request-based autoscaler with predictive scaling."""

from __future__ import annotations
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScaleConfig:
    """Per-function scaling configuration."""
    keep_warm: int = 0
    max_containers: int = 10
    container_idle_timeout: int = 300
    scale_up_threshold: float = 0.8   # utilization % to trigger scale-up
    scale_down_threshold: float = 0.2  # utilization % to trigger scale-down
    requests_per_container: int = 1    # for batch endpoints, >1


@dataclass
class AppMetrics:
    """Real-time metrics for an app's functions."""
    active_requests: int = 0
    active_containers: int = 0
    idle_containers: int = 0
    queue_depth: int = 0
    request_history: deque = field(default_factory=lambda: deque(maxlen=360))  # 1h at 10s intervals
    last_scale_action_at: float = 0.0

    @property
    def utilization(self) -> float:
        if self.active_containers == 0:
            return 1.0 if self.active_requests > 0 else 0.0
        return self.active_requests / self.active_containers

    def record_request(self):
        self.request_history.append((time.time(), self.active_requests))


class Autoscaler:
    """Evaluates scaling decisions based on request metrics.

    Strategies:
    1. Reactive: Scale up when queue has pending requests
    2. Predictive: Scale up when utilization trend is rising
    3. Keep-warm: Maintain minimum containers always running
    """

    def __init__(self):
        self._metrics: dict[str, AppMetrics] = {}  # key: "app:function"
        self._configs: dict[str, ScaleConfig] = {}

    def configure(self, app_name: str, fn_name: str, config: ScaleConfig):
        """Set scaling config for a function."""
        key = f"{app_name}:{fn_name}"
        self._configs[key] = config

    def record_request_start(self, app_name: str, fn_name: str):
        key = f"{app_name}:{fn_name}"
        metrics = self._metrics.setdefault(key, AppMetrics())
        metrics.active_requests += 1
        metrics.record_request()

    def record_request_end(self, app_name: str, fn_name: str):
        key = f"{app_name}:{fn_name}"
        metrics = self._metrics.get(key)
        if metrics:
            metrics.active_requests = max(0, metrics.active_requests - 1)

    def update_containers(self, app_name: str, fn_name: str, active: int, idle: int):
        key = f"{app_name}:{fn_name}"
        metrics = self._metrics.setdefault(key, AppMetrics())
        metrics.active_containers = active
        metrics.idle_containers = idle

    def evaluate(self, app_name: str, fn_name: str) -> ScaleDecision:
        """Evaluate whether to scale up, down, or maintain."""
        key = f"{app_name}:{fn_name}"
        metrics = self._metrics.get(key, AppMetrics())
        config = self._configs.get(key, ScaleConfig())

        total = metrics.active_containers + metrics.idle_containers

        # Rule 1: Keep-warm minimum
        if total < config.keep_warm:
            return ScaleDecision(
                action="scale_up",
                count=config.keep_warm - total,
                reason=f"below keep_warm minimum ({total}/{config.keep_warm})",
            )

        # Rule 2: Queue has pending requests — scale up immediately
        if metrics.queue_depth > 0:
            needed = math.ceil(metrics.queue_depth / config.requests_per_container)
            headroom = config.max_containers - total
            scale = min(needed, headroom)
            if scale > 0:
                return ScaleDecision(
                    action="scale_up",
                    count=scale,
                    reason=f"queue depth {metrics.queue_depth}, adding {scale} containers",
                )

        # Rule 3: High utilization — scale up
        if metrics.utilization > config.scale_up_threshold and total < config.max_containers:
            # Calculate trend
            trend = self._calculate_trend(metrics)
            if trend > 0:
                scale = max(1, int(trend * 2))
                scale = min(scale, config.max_containers - total)
                return ScaleDecision(
                    action="scale_up",
                    count=scale,
                    reason=f"utilization {metrics.utilization:.0%}, trend +{trend:.2f}",
                )

        # Rule 4: Low utilization — scale down (but respect keep_warm)
        if metrics.utilization < config.scale_down_threshold and total > config.keep_warm:
            excess = total - max(config.keep_warm, metrics.active_requests)
            if excess > 0:
                # Cooldown: don't scale down too fast
                if time.time() - metrics.last_scale_action_at > 60:
                    return ScaleDecision(
                        action="scale_down",
                        count=min(excess, 2),  # Remove max 2 at a time
                        reason=f"utilization {metrics.utilization:.0%}, {excess} excess containers",
                    )

        return ScaleDecision(action="maintain", count=0, reason="within bounds")

    def _calculate_trend(self, metrics: AppMetrics) -> float:
        """Calculate request rate trend (positive = increasing)."""
        history = list(metrics.request_history)
        if len(history) < 6:
            return 0.0

        recent = history[-3:]
        older = history[-6:-3]

        avg_recent = sum(r for _, r in recent) / len(recent)
        avg_older = sum(r for _, r in older) / len(older)

        if avg_older == 0:
            return 1.0 if avg_recent > 0 else 0.0
        return (avg_recent - avg_older) / avg_older


@dataclass
class ScaleDecision:
    action: str  # "scale_up", "scale_down", "maintain"
    count: int
    reason: str
