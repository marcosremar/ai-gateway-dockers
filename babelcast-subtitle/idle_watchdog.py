"""
Container Idle Watchdog — auto-stops the pod when idle.

Works INSIDE the container, independent of the gateway server.
Tracks last request time via FastAPI middleware and exits after
IDLE_TIMEOUT_MIN minutes of no model requests.

Usage in server.py:
    from idle_watchdog import add_idle_middleware, start_watchdog
    add_idle_middleware(app)

    # inside lifespan:
    asyncio.create_task(start_watchdog())

Env vars:
    IDLE_TIMEOUT_MIN  — minutes before auto-stop (default: 15, 0=disabled)
"""

import os
import time
import asyncio
import logging

log = logging.getLogger("idle-watchdog")

IDLE_TIMEOUT_MIN = int(os.environ.get("IDLE_TIMEOUT_MIN", "15"))
CHECK_INTERVAL_S = 30

_last_request_time = time.time()


def touch_activity():
    global _last_request_time
    _last_request_time = time.time()


def idle_seconds() -> float:
    return time.time() - _last_request_time


def add_idle_middleware(app):
    """Add request tracking middleware. Call BEFORE app starts."""
    if IDLE_TIMEOUT_MIN <= 0:
        return

    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request

    # Paths that do NOT count as user activity (health polls, internal llm checks)
    EXCLUDED_PATHS = frozenset([
        "/health", "/version", "/debug", "/debug/logs",
        "/v1/models",  # llama.cpp internal model list
        "/docs", "/openapi.json", "/favicon.ico",
    ])

    class IdleTrackingMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            path = request.url.path
            # Only external model requests count (STT, LLM, TTS, pipeline)
            if path not in EXCLUDED_PATHS and not path.startswith("/docs"):
                touch_activity()
                log.debug(f"[idle-watchdog] Activity: {request.method} {path}")
            return await call_next(request)

    app.add_middleware(IdleTrackingMiddleware)
    log.info(f"[idle-watchdog] Middleware installed — {IDLE_TIMEOUT_MIN} min timeout")


async def start_watchdog():
    """Background coroutine — call via asyncio.create_task() in lifespan."""
    if IDLE_TIMEOUT_MIN <= 0:
        log.info("[idle-watchdog] Disabled (IDLE_TIMEOUT_MIN=0)")
        return

    timeout_s = IDLE_TIMEOUT_MIN * 60
    log.warning(f"[idle-watchdog] Started — auto-exit after {IDLE_TIMEOUT_MIN} min idle (check every {CHECK_INTERVAL_S}s)")
    warned = False

    while True:
        await asyncio.sleep(CHECK_INTERVAL_S)
        idle_s = idle_seconds()
        log.warning(f"[idle-watchdog] Check: idle {idle_s:.0f}s / {timeout_s}s ({idle_s/timeout_s*100:.0f}%)")

        if idle_s >= timeout_s:
            log.warning(f"[idle-watchdog] Idle {idle_s/60:.0f} min — shutting down")
            # Try RunPod stop API if available
            pod_id = os.environ.get("RUNPOD_POD_ID", "")
            api_key = os.environ.get("RUNPOD_API_KEY", "")
            if pod_id and api_key:
                try:
                    import httpx
                    async with httpx.AsyncClient() as client:
                        r = await client.post(
                            f"https://rest.runpod.io/v1/pods/{pod_id}/stop",
                            headers={"Authorization": f"Bearer {api_key}",
                                     "Content-Type": "application/json"},
                            timeout=10,
                        )
                        log.info(f"[idle-watchdog] RunPod stop: HTTP {r.status_code}")
                        return
                except Exception as e:
                    log.error(f"[idle-watchdog] RunPod stop failed: {e}")
            # Fallback: exit process
            log.warning("[idle-watchdog] Exiting process")
            os._exit(0)

        if idle_s >= timeout_s * 0.75 and not warned:
            warned = True
            remaining = int(timeout_s - idle_s)
            log.warning(f"[idle-watchdog] Warning: {remaining}s until auto-stop")

        if idle_s < timeout_s * 0.5:
            warned = False
