"""
DiT360 Server — 360° Panoramic Image Generation via FLUX.1-dev + LoRA

Generates high-quality equirectangular panoramic images (1024×2048) from text
prompts. Built on FLUX.1-dev (12B) with DiT360 LoRA fine-tune.
Supports 4x upscale via Real-ESRGAN for 8K output (4096×8192).

VRAM: ~37GB without offload, ~22GB with enable_model_cpu_offload().
CPU_OFFLOAD env var: "auto" (default) = offload if <40GB VRAM, "on", "off".

Endpoints:
  GET  /health              — Service status + model readiness
  GET  /version             — Image version info
  POST /v1/images/generate  — Generate panoramic image (supports upscale param)
"""

import os
import sys
import io
import time
import base64
import logging
import asyncio
from contextlib import asynccontextmanager

# ── Cold-start optimizations (BEFORE torch import so cache env vars take effect) ──
# Catch ALL exceptions — never let an OPTIONAL optimization block startup.
sys.path.insert(0, "/app")
try:
    from coldstart import bootstrap, prefetch_safetensors
    bootstrap(torch_cache_dir=os.environ.get("TORCHINDUCTOR_CACHE_DIR", "/app/.torch-cache"))
except Exception as _coldstart_err:
    print(
        f"[server] coldstart unavailable ({type(_coldstart_err).__name__}: "
        f"{_coldstart_err}) — running without optimizations",
        file=sys.stderr,
    )
    prefetch_safetensors = lambda *a, **k: 0.0  # no-op stub

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("dit360")

# ── Config ───────────────────────────────────────────────────────────────────

MODEL_ID = os.environ.get("MODEL_ID", "black-forest-labs/FLUX.1-dev")
LORA_ID = os.environ.get("LORA_ID", "Insta360-Research/DiT360-Panorama-Image-Generation")
DEFAULT_STEPS = int(os.environ.get("DEFAULT_STEPS", "28"))
DEFAULT_GUIDANCE = float(os.environ.get("DEFAULT_GUIDANCE", "2.8"))
DEFAULT_WIDTH = int(os.environ.get("DEFAULT_WIDTH", "2048"))
DEFAULT_HEIGHT = int(os.environ.get("DEFAULT_HEIGHT", "1024"))
CPU_OFFLOAD = os.environ.get("CPU_OFFLOAD", "auto")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Global state ────────────────────────────────────────────────────────────

pipe = None
upscaler = None
boot_time = time.time()

service_status = {
    "pipeline": "pending",
    "upscaler": "pending",
}


def _should_offload() -> bool:
    """Decide whether to use CPU offloading based on available VRAM."""
    if CPU_OFFLOAD == "on":
        return True
    if CPU_OFFLOAD == "off":
        return False
    # auto: offload if VRAM < 40GB
    if DEVICE == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        log.info(f"VRAM: {vram_gb:.1f}GB — offload={'yes' if vram_gb < 40 else 'no'}")
        return vram_gb < 40
    return False


# ── Background model loading ────────────────────────────────────────────────

async def _load_pipeline():
    """Load FLUX.1-dev + DiT360 LoRA pipeline."""
    global pipe
    try:
        service_status["pipeline"] = "downloading"
        log.info(f"Loading DiT360 pipeline ({MODEL_ID})...")

        from diffusers import FluxPipeline

        # Pre-warm OS page cache via sequential read of all FLUX safetensors
        # shards (~24GB across ~15 files). Subsequent diffusers load becomes
        # RAM-speed access. Best-effort, no-op if helper missing.
        try:
            t_pre = await asyncio.to_thread(prefetch_safetensors, MODEL_ID)
            if t_pre > 0:
                log.info(f"Prefetched FLUX safetensors in {t_pre:.1f}s")
        except Exception as e:
            log.warning(f"Prefetch failed (non-fatal): {e}")

        def _load():
            p = FluxPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
            )
            log.info(f"Loading LoRA weights from {LORA_ID}...")
            p.load_lora_weights(LORA_ID)

            if _should_offload():
                log.info("Enabling CPU offload (saves VRAM, slower inference)")
                p.enable_model_cpu_offload()
            else:
                log.info("Loading full pipeline to GPU")
                p = p.to("cuda")

            return p

        service_status["pipeline"] = "loading"
        pipe = await asyncio.to_thread(_load)
        service_status["pipeline"] = "ready"
        log.info("DiT360 pipeline ready")

    except Exception as e:
        service_status["pipeline"] = f"error: {e}"
        log.error(f"Pipeline load failed: {e}", exc_info=True)


async def _load_upscaler():
    """Load Real-ESRGAN 4x upscaler model."""
    global upscaler
    try:
        service_status["upscaler"] = "loading"
        log.info("Loading Real-ESRGAN 4x upscaler...")

        def _load():
            # Patch basicsr's broken import before importing realesrgan
            import torchvision.transforms.functional as TF
            import sys
            sys.modules['torchvision.transforms.functional_tensor'] = TF

            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            return RealESRGANer(
                scale=4,
                model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                model=model,
                tile=400,  # Tile size for large images (prevents OOM)
                tile_pad=10,
                pre_pad=0,
                half=True,  # fp16 for speed
                device="cuda" if DEVICE == "cuda" else "cpu",
            )

        upscaler = await asyncio.to_thread(_load)
        service_status["upscaler"] = "ready"
        log.info("Real-ESRGAN 4x upscaler ready")
    except Exception as e:
        service_status["upscaler"] = f"error: {e}"
        log.error(f"Upscaler load failed: {e}", exc_info=True)


# ── App ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(_load_pipeline())
    asyncio.create_task(_load_upscaler())
    try:
        from idle_watchdog import start_watchdog
        asyncio.create_task(start_watchdog())
    except Exception as e:
        log.warning(f"Idle watchdog not started: {e}")
    yield

app = FastAPI(title="DiT360", lifespan=lifespan)

try:
    from idle_watchdog import add_idle_middleware
    add_idle_middleware(app)
except Exception as e:
    log.warning(f"Idle middleware not installed: {e}")


@app.get("/health")
async def health():
    return {
        "status": "ok" if service_status["pipeline"] == "ready" else "loading",
        "uptime": round(time.time() - boot_time, 1),
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1) if DEVICE == "cuda" else None,
        "services": service_status,
        "model_warmth": {
            "pipeline": {"warm": service_status["pipeline"] == "ready"},
            "upscaler": {"warm": service_status["upscaler"] == "ready"},
        },
    }


@app.get("/version")
async def version():
    cap = torch.cuda.get_device_capability() if DEVICE == "cuda" else (0, 0)
    return {
        "version": "dit360-1.1",
        "type": "image-generation",
        "cuda_arch": f"sm_{cap[0] * 10 + cap[1]}",
        "models": {"base": MODEL_ID, "lora": LORA_ID},
        "defaults": {
            "steps": DEFAULT_STEPS,
            "guidance_scale": DEFAULT_GUIDANCE,
            "width": DEFAULT_WIDTH,
            "height": DEFAULT_HEIGHT,
        },
        "cpu_offload": _should_offload(),
    }


# ── Debug Endpoint ──────────────────────────────────────────────────────────

LOG_FILE = os.environ.get("LOG_FILE", "/tmp/container.log")

@app.get("/debug/logs")
async def debug_logs(lines: int = 200):
    """Return recent container logs for post-mortem debugging.
    Logs are captured by start.sh via tee to LOG_FILE."""
    try:
        with open(LOG_FILE) as f:
            all_lines = f.readlines()
        tail = all_lines[-lines:]
        return {
            "total_lines": len(all_lines),
            "returned_lines": len(tail),
            "uptime": round(time.time() - boot_time, 1),
            "services": service_status,
            "logs": "".join(tail),
        }
    except FileNotFoundError:
        return {"error": f"Log file {LOG_FILE} not found", "services": service_status}


# ── Image Generation Endpoint ──────────────────────────────────────────────

@app.post("/v1/images/generate")
async def generate_image(request: Request):
    """Generate a 360° panoramic image from a text prompt.

    Request body (JSON):
        prompt (str): Text description of the panoramic scene (required)
        width (int): Image width, default 2048
        height (int): Image height, default 1024
        num_inference_steps (int): Diffusion steps, default 28
        guidance_scale (float): Classifier-free guidance, default 2.8
        seed (int|null): Random seed for reproducibility, default null (random)
        upscale (int): Upscale factor (1=none, 2=2x, 4=4x via Real-ESRGAN). Default 1.
                       4x turns 2048x1024 into 8192x4096 (8K).
        response_format (str): "b64_json" (default) or "raw" (returns PNG directly)

    Returns:
        JSON with base64-encoded PNG image, or raw PNG bytes.
    """
    if pipe is None or service_status["pipeline"] != "ready":
        return JSONResponse(status_code=503, content={"error": "Pipeline not ready yet", "status": service_status})

    body = await request.json()
    prompt = body.get("prompt", "")
    if not prompt:
        return JSONResponse(status_code=400, content={"error": "prompt is required"})

    width = body.get("width", DEFAULT_WIDTH)
    height = body.get("height", DEFAULT_HEIGHT)
    steps = body.get("num_inference_steps", DEFAULT_STEPS)
    guidance = body.get("guidance_scale", DEFAULT_GUIDANCE)
    seed = body.get("seed")
    upscale_factor = body.get("upscale", 1)
    response_format = body.get("response_format", "b64_json")

    if upscale_factor not in (1, 2, 4):
        return JSONResponse(status_code=400, content={"error": "upscale must be 1, 2, or 4"})
    if upscale_factor > 1 and (upscaler is None or service_status["upscaler"] != "ready"):
        return JSONResponse(status_code=503, content={"error": "Upscaler not ready yet", "status": service_status})

    # Prepend panorama hint if not already present
    full_prompt = prompt if "panorama" in prompt.lower() else f"This is a panorama. {prompt}"

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda" if DEVICE == "cuda" else "cpu").manual_seed(int(seed))

    log.info(f"Generating: {width}x{height}, {steps} steps, guidance={guidance}, seed={seed}")
    t0 = time.time()

    def _generate():
        result = pipe(
            full_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        return result.images[0]

    try:
        image = await asyncio.to_thread(_generate)
    except Exception as e:
        log.error(f"Generation failed: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Generation failed: {e}"})

    gen_ms = round((time.time() - t0) * 1000)
    log.info(f"Generated in {gen_ms}ms")

    # Upscale if requested
    upscale_ms = 0
    if upscale_factor > 1:
        import numpy as np
        t1 = time.time()
        log.info(f"Upscaling {upscale_factor}x with Real-ESRGAN...")

        def _upscale():
            img_np = np.array(image)
            output, _ = upscaler.enhance(img_np, outscale=upscale_factor)
            from PIL import Image as PILImage
            return PILImage.fromarray(output)

        try:
            image = await asyncio.to_thread(_upscale)
            upscale_ms = round((time.time() - t1) * 1000)
            log.info(f"Upscaled to {image.size[0]}x{image.size[1]} in {upscale_ms}ms")
        except Exception as e:
            log.error(f"Upscale failed: {e}", exc_info=True)
            return JSONResponse(status_code=500, content={"error": f"Upscale failed: {e}"})

    latency_ms = gen_ms + upscale_ms

    # Encode to PNG
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    if response_format == "raw":
        return Response(content=png_bytes, media_type="image/png", headers={
            "X-Latency-Ms": str(latency_ms),
            "X-Seed": str(seed),
        })

    # Final dimensions (after upscale)
    final_w, final_h = image.size

    # Default: b64_json (OpenAI-style)
    return {
        "data": [{
            "b64_json": base64.b64encode(png_bytes).decode("utf-8"),
        }],
        "model": "dit360",
        "prompt": full_prompt,
        "width": final_w,
        "height": final_h,
        "base_width": width,
        "base_height": height,
        "upscale": upscale_factor,
        "steps": steps,
        "guidance_scale": guidance,
        "seed": seed,
        "latency_ms": latency_ms,
        "generation_ms": gen_ms,
        "upscale_ms": upscale_ms,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    log.info(f"Starting DiT360 server on :{port}")
    if DEVICE == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        log.info(f"VRAM: {vram:.1f}GB, CPU offload: {CPU_OFFLOAD}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
