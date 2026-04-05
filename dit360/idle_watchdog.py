"""
Container Idle Watchdog — auto-stops the RunPod/Vast.ai pod when idle.

Works INSIDE the container, independent of the gateway server.
Tracks last request time via FastAPI middleware and shuts down after
IDLE_TIMEOUT_MIN minutes of no requests.

Usage:
    from idle_watchdog import install_idle_watchdog
    install_idle_watchdog(app)  # app is a FastAPI instance

Env vars:
    IDLE_TIMEOUT_MIN  — minutes before auto-stop (default: 15, 0=disabled)
    IDLE_ACTION       — "stop" (default) or "exit"
    RUNPOD_POD_ID     — auto-detected from RUNPOD_POD_ID env var
    RUNPOD_API_KEY    — needed for stop via API (auto-detected)
"""

import os
import time
import asyncio
import logging

log = logging.getLogger("idle-watchdog")

# ── Config ──────────────────────────────────────────────────────────────────

IDLE_TIMEOUT_MIN = int(os.environ.get("IDLE_TIMEOUT_MIN", "15"))
IDLE_ACTION = os.environ.get("IDLE_ACTION", "stop")  # "stop" or "exit"
CHECK_INTERVAL_S = 60  # check every 60s

# ── State ───────────────────────────────────────────────────────────────────

_last_request_time = time.time()
_watchdog_task = None


def touch_activity():
    """Reset idle timer — called on every request."""
    global _last_request_time
    _last_request_time = time.time()


def idle_seconds() -> float:
    return time.time() - _last_request_time


# ── Middleware ──────────────────────────────────────────────────────────────

def install_idle_watchdog(app):
    """Install idle tracking middleware + background watchdog on a FastAPI app."""
    if IDLE_TIMEOUT_MIN <= 0:
        log.info("[idle-watchdog] Disabled (IDLE_TIMEOUT_MIN=0)")
        return

    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request

    class IdleTrackingMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            path = request.url.path
            # Only count model requests, not health/status polls
            if path not in ("/health", "/version", "/debug", "/debug/logs"):
                touch_activity()
            return await call_next(request)

    app.add_middleware(IdleTrackingMiddleware)

    # Start background watchdog
    global _watchdog_task

    async def _watchdog_loop():
        log.info(f"[idle-watchdog] Started — auto-{IDLE_ACTION} after {IDLE_TIMEOUT_MIN} min idle")
        warned = False
        while True:
            await asyncio.sleep(CHECK_INTERVAL_S)
            idle_s = idle_seconds()
            idle_min = idle_s / 60
            timeout_s = IDLE_TIMEOUT_MIN * 60

            if idle_s >= timeout_s:
                log.warning(f"[idle-watchdog] Idle {idle_min:.0f} min — executing {IDLE_ACTION}")
                await _execute_idle_action()
                return

            # Warn at 75% of timeout
            if idle_s >= timeout_s * 0.75 and not warned:
                warned = True
                remaining = int(timeout_s - idle_s)
                log.warning(f"[idle-watchdog] Warning: {remaining}s until auto-{IDLE_ACTION}")

            if idle_s < timeout_s * 0.5:
                warned = False  # Reset warning after activity

    async def _start_watchdog():
        global _watchdog_task
        _watchdog_task = asyncio.create_task(_watchdog_loop())

    # Register startup event
    from contextlib import asynccontextmanager
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def wrapped_lifespan(app_instance):
        async with original_lifespan(app_instance):
            await _start_watchdog()
            yield

    app.router.lifespan_context = wrapped_lifespan

    log.info(f"[idle-watchdog] Installed — {IDLE_TIMEOUT_MIN} min timeout, action={IDLE_ACTION}")


async def _execute_idle_action():
    """Stop or exit based on IDLE_ACTION."""
    pod_id = os.environ.get("RUNPOD_POD_ID", "")

    if IDLE_ACTION == "stop" and pod_id:
        # Try RunPod stop API
        api_key = os.environ.get("RUNPOD_API_KEY", "")
        if api_key:
            import httpx
            try:
                async with httpx.AsyncClient() as client:
                    r = await client.post(
                        f"https://rest.runpod.io/v1/pods/{pod_id}/stop",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        timeout=10,
                    )
                    log.info(f"[idle-watchdog] RunPod stop: HTTP {r.status_code}")
                    return
            except Exception as e:
                log.error(f"[idle-watchdog] RunPod stop failed: {e}")

    # Fallback: exit process (container stops, RunPod/Vast marks as exited)
    log.warning(f"[idle-watchdog] Exiting process (idle timeout)")
    os._exit(0)
