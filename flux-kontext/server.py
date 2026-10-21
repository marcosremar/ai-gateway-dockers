"""FLUX.1-Kontext-dev FastAPI server — character-consistent image gen.

Endpoints:
  GET  /health
  POST /v1/images/edits   multipart {prompt, image, size?, seed?, num_inference_steps?}
                          returns JSON {data: [{b64_json: ...}], elapsed_s}

Pattern: pass one reference image + new scene prompt; Kontext keeps the
identity / setting consistent across generations. Used by canal-dark to
hold "Zeno" identical across 15 narrative scenes.
"""
import base64
import io
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

MODEL_ID = os.environ.get("KONTEXT_MODEL", "black-forest-labs/FLUX.1-Kontext-dev")
DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16}[
    os.environ.get("KONTEXT_DTYPE", "bfloat16")
]
state = {"pipe": None, "loaded": False, "load_error": None}


def _load() -> None:
    if state["pipe"] is not None:
        return
    try:
        from diffusers import FluxKontextPipeline
        pipe = FluxKontextPipeline.from_pretrained(
            MODEL_ID, torch_dtype=DTYPE, cache_dir="/models",
        )
        if os.environ.get("KONTEXT_OFFLOAD") == "1":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe = pipe.to("cuda")
        state["pipe"] = pipe
        state["loaded"] = True
    except Exception as e:
        import traceback
        state["load_error"] = repr(e) + "\n" + traceback.format_exc()
        raise


@asynccontextmanager
async def lifespan(_app):
    try:
        _load()
    except Exception:
        import traceback
        traceback.print_exc()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "healthy" if state["loaded"] else "loading",
        "model": MODEL_ID,
        "model_loaded": state["loaded"],
        "load_error": state["load_error"],
    }


@app.post("/v1/images/edits")
async def edits(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    size: str = Form("1664x928"),
    seed: Optional[int] = Form(None),
    num_inference_steps: int = Form(28),
    guidance_scale: float = Form(2.5),
):
    if not state["loaded"]:
        try:
            _load()
        except Exception as e:
            raise HTTPException(503, f"model not ready: {e}")
    img_bytes = await image.read()
    ref = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = (int(x) for x in size.split("x"))
    gen = torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None
    t0 = time.time()
    out = state["pipe"](
        image=ref,
        prompt=prompt,
        width=w,
        height=h,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=gen,
    ).images[0]
    elapsed = round(time.time() - t0, 2)
    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return JSONResponse({
        "created": int(time.time()),
        "data": [{"b64_json": base64.b64encode(buf.getvalue()).decode()}],
        "elapsed_s": elapsed,
    })
