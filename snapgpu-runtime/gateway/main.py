"""SnapGPU Gateway — FastAPI control plane."""

from __future__ import annotations
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI

from .db import get_engine
from .routes import health, apps, invoke, images, snapshots


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    # Ensure DB tables exist
    get_engine()
    print("[snapgpu-gateway] Started")
    yield
    print("[snapgpu-gateway] Shutting down")


app = FastAPI(
    title="SnapGPU Gateway",
    description="GPU serverless control plane — deploy, invoke, and manage GPU functions",
    version="0.1.0",
    lifespan=lifespan,
)

# Mount routes
app.include_router(health.router)
app.include_router(apps.router)
app.include_router(invoke.router)
app.include_router(images.router)
app.include_router(snapshots.router)


@app.get("/")
async def root():
    return {
        "service": "snapgpu-gateway",
        "version": "0.1.0",
        "docs": "/docs",
    }


def start():
    """Entry point for `snapgpu gateway` or `uvicorn gateway.main:app`."""
    import uvicorn
    port = int(os.environ.get("SNAPGPU_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    start()
