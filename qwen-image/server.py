"""Qwen-Image (2512) FastAPI server. OpenAI-compatible /v1/images/generations.

Apache 2.0 license, 20B MMDiT. Quant 4-bit fits L40S 48GB. ~12s/img steady-state.
"""
import base64, io, os, time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import torch

MODEL_ID = os.environ.get("QWEN_IMAGE_MODEL", "Qwen/Qwen-Image")
DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
    os.environ.get("QWEN_IMAGE_DTYPE", "bfloat16")
]
state = {"pipe": None, "loaded": False, "load_error": None}


def _load():
    if state["pipe"] is not None:
        return
    try:
        from diffusers import DiffusionPipeline
        kw = dict(torch_dtype=DTYPE, cache_dir="/models")
        if os.environ.get("QWEN_IMAGE_QUANT") == "4bit":
            # diffusers 0.35+ requires PipelineQuantizationConfig wrapper
            # for transformers BitsAndBytesConfig (raw BNB config rejected
            # since 0.34's native quant API). Quantize the 20B MMDiT
            # ("transformer") and the text encoder; keep the VAE in fp16.
            from diffusers.quantizers import PipelineQuantizationConfig
            from diffusers.quantizers.quantization_config import (
                BitsAndBytesConfig as DiffBnBConfig,
            )
            from transformers import BitsAndBytesConfig as TransBnBConfig
            # Granular per-component quant: diffusers' BnB config for the
            # MMDiT transformer, transformers' BnB config for the Qwen2.5
            # text encoder (quant config classes belong to whichever lib
            # owns the component — mixing them raises the obscure
            # `'dict' has no attribute 'to_dict'` you'd get with the
            # generic backend+kwargs form).
            bnb_kwargs = dict(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=DTYPE,
            )
            kw["quantization_config"] = PipelineQuantizationConfig(
                quant_mapping={
                    "transformer": DiffBnBConfig(**bnb_kwargs),
                    "text_encoder": TransBnBConfig(**bnb_kwargs),
                },
            )
        pipe = DiffusionPipeline.from_pretrained(MODEL_ID, **kw)
        # L40S 48GB cannot host the full bf16 pipeline + activations at
        # 1664x928. enable_model_cpu_offload keeps the active stage
        # resident which still OOMs on 1024+ generations. Sequential
        # offload swaps every submodule individually — slower but never
        # OOMs. Slicing + tiling shaves activation memory further.
        try:
            pipe.enable_sequential_cpu_offload()
        except Exception:
            try:
                pipe.enable_model_cpu_offload()
            except Exception:
                pipe = pipe.to("cuda")
        for fn in ("enable_attention_slicing", "enable_vae_slicing", "enable_vae_tiling"):
            getattr(pipe, fn, lambda: None)()
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


class GenReq(BaseModel):
    prompt: str
    size: Optional[str] = "1024x1024"
    n: Optional[int] = 1
    num_inference_steps: Optional[int] = 30
    guidance_scale: Optional[float] = 4.0
    seed: Optional[int] = None
    response_format: Optional[str] = "b64_json"


@app.post("/v1/images/generations")
def generate(body: GenReq):
    if not state["loaded"]:
        try:
            _load()
        except Exception as e:
            raise HTTPException(503, f"model not ready: {e}")
    if state["pipe"] is None:
        raise HTTPException(503, "pipe missing")
    w, h = (int(x) for x in (body.size or "1024x1024").split("x"))
    gen = torch.Generator(device="cuda").manual_seed(body.seed) if body.seed is not None else None
    t0 = time.time()
    images = state["pipe"](
        prompt=body.prompt,
        width=w,
        height=h,
        num_inference_steps=body.num_inference_steps or 30,
        guidance_scale=body.guidance_scale or 4.0,
        num_images_per_prompt=body.n or 1,
        generator=gen,
    ).images
    elapsed = round(time.time() - t0, 2)
    data = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        data.append({"b64_json": base64.b64encode(buf.getvalue()).decode()})
    return JSONResponse({"created": int(time.time()), "data": data, "elapsed_s": elapsed})
