"""FLUX text-to-image server — FastAPI wrapper pro ai-gateway.

OpenAI-compatible endpoints so the gateway pipeline can route image
generation identically to other providers.

  GET  /health                      — readiness probe.
  POST /v1/images/generations       — JSON body {prompt, size, n, response_format}.

Env:
  IDLE_TIMEOUT_MIN    — auto-shutdown after no requests (default 15).
  FLUX_MODEL          — HF repo id (default black-forest-labs/FLUX.1-schnell).
  PORT                — listen port (default 8000).
  FLUX_DTYPE          — bfloat16 | float16 | float32 (default bfloat16).
  FLUX_OFFLOAD        — 1 to enable CPU offload (saves VRAM).
"""

import asyncio
import base64
import io
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from idle_watchdog import add_idle_middleware, start_watchdog, touch_activity


_pipeline = None
_load_lock = asyncio.Lock()


def _resolve_dtype():
    name = os.environ.get("FLUX_DTYPE", "bfloat16").lower()
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(name, torch.bfloat16)


async def _ensure_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    async with _load_lock:
        if _pipeline is not None:
            return _pipeline
        from diffusers import FluxPipeline
        repo = os.environ.get("FLUX_MODEL", "black-forest-labs/FLUX.1-schnell")
        dtype = _resolve_dtype()
        print(f"[flux] loading {repo} dtype={dtype}")
        t0 = time.time()
        pipe = FluxPipeline.from_pretrained(repo, torch_dtype=dtype)
        if os.environ.get("FLUX_OFFLOAD", "0") == "1":
            pipe.enable_model_cpu_offload()
            print("[flux] CPU offload enabled")
        else:
            pipe = pipe.to("cuda")
        print(f"[flux] loaded in {time.time()-t0:.1f}s")
        _pipeline = pipe
        return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(start_watchdog())
    # fire-and-forget pipeline load — first request waits if not ready
    asyncio.create_task(_ensure_pipeline())
    yield


app = FastAPI(title="FLUX text-to-image (ai-gateway)", lifespan=lifespan)
add_idle_middleware(app)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _pipeline is not None,
            "model": os.environ.get("FLUX_MODEL", "black-forest-labs/FLUX.1-schnell")}


class ImageGenRequest(BaseModel):
    prompt: str
    size: str = "1024x1024"           # WxH
    n: int = 1                         # batch
    response_format: str = "b64_json"  # b64_json | url
    num_inference_steps: Optional[int] = None
    guidance_scale: float = 0.0        # 0.0 default for schnell, 3.5 for dev
    seed: Optional[int] = None


@app.post("/v1/images/generations")
async def generate_image(req: ImageGenRequest):
    touch_activity()
    pipe = await _ensure_pipeline()

    try:
        w, h = [int(x) for x in req.size.lower().split("x")]
    except Exception:
        raise HTTPException(400, f"invalid size: {req.size}")

    # FLUX.1-schnell defaults: 4 steps. dev: 28-50 steps.
    steps = req.num_inference_steps
    if steps is None:
        repo = os.environ.get("FLUX_MODEL", "")
        steps = 4 if "schnell" in repo.lower() else 28

    generator = None
    if req.seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(req.seed)

    images = []
    for i in range(req.n):
        out = pipe(
            prompt=req.prompt,
            height=h, width=w,
            num_inference_steps=steps,
            guidance_scale=req.guidance_scale,
            generator=generator,
        ).images[0]

        buf = io.BytesIO()
        out.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        images.append({"b64_json": b64})

    return {"created": int(time.time()), "data": images,
            "model": os.environ.get("FLUX_MODEL"), "steps": steps, "size": f"{w}x{h}"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
