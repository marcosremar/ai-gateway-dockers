"""
Container Idle Watchdog — auto-stops the pod when idle.

Identical to qwen3-tts/idle_watchdog.py, manim/idle_watchdog.py and
flux/idle_watchdog.py — kept per-Docker for self-contained build. No GPU
shutdown logic; CPU pods just exit the process and the gateway autoscaler
handles the rest.
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
    if IDLE_TIMEOUT_MIN <= 0:
        return

    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request

    EXCLUDED_PATHS = frozenset([
        "/health", "/version", "/debug", "/debug/logs",
        "/v1/models", "/docs", "/openapi.json", "/favicon.ico",
    ])

    class IdleTrackingMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            path = request.url.path
            if path not in EXCLUDED_PATHS and not path.startswith("/docs"):
                touch_activity()
            return await call_next(request)

    app.add_middleware(IdleTrackingMiddleware)
    log.info(f"[idle-watchdog] {IDLE_TIMEOUT_MIN} min timeout active")


async def start_watchdog():
    if IDLE_TIMEOUT_MIN <= 0:
        return
    timeout_s = IDLE_TIMEOUT_MIN * 60
    while True:
        await asyncio.sleep(CHECK_INTERVAL_S)
        idle_s = idle_seconds()
        if idle_s >= timeout_s:
            pod_id = os.environ.get("RUNPOD_POD_ID", "")
            api_key = os.environ.get("RUNPOD_API_KEY", "")
            if pod_id and api_key:
                try:
                    import httpx
                    async with httpx.AsyncClient() as client:
                        await client.post(
                            f"https://rest.runpod.io/v1/pods/{pod_id}/stop",
                            headers={"Authorization": f"Bearer {api_key}",
                                     "Content-Type": "application/json"},
                            timeout=10,
                        )
                        return
                except Exception as e:
                    log.error(f"runpod stop failed: {e}")
            os._exit(0)
