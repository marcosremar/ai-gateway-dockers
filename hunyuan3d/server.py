"""
Hunyuan3D 2.0 — FastAPI inference server for Image/Text-to-3D generation.

Endpoints:
    POST /generate          — Upload an image, get back a .GLB file (image-to-3D)
    POST /generate-from-url — JSON {image_url, seed?, with_texture?}, get .GLB
    POST /generate-from-text — JSON {prompt, seed?, with_texture?}, get .GLB
    GET  /health            — Health check (model loaded?)
    GET  /diag              — Full diagnostic dump
    GET  /logs              — Tail of /var/log/app.log

Environment variables:
    HUNYUAN3D_MODEL    — HuggingFace model ID (default: tencent/Hunyuan3D-2)
    HUNYUAN3D_STEPS    — Diffusion steps (default: 30)
    HUNYUAN3D_OCTREE   — Octree resolution (default: 256)
    WITH_TEXTURE       — Default texture stage on (1) or off (0). Default: 1.
    HOST               — Bind address (default: 0.0.0.0)
    PORT               — Bind port (default: 8000)
"""

import os
import sys
import io
import time
import logging
import tempfile
import traceback

import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hunyuan3d")

MODEL_ID = os.environ.get("HUNYUAN3D_MODEL", "tencent/Hunyuan3D-2")
STEPS = int(os.environ.get("HUNYUAN3D_STEPS", "30"))
OCTREE = int(os.environ.get("HUNYUAN3D_OCTREE", "256"))
WITH_TEXTURE_DEFAULT = os.environ.get("WITH_TEXTURE", "1") == "1"
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

app = FastAPI(title="Hunyuan3D 2.0 Generation Server", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

try:
    from idle_watchdog import add_idle_middleware
    add_idle_middleware(app)
except Exception:
    pass

# ── Models (lazy-loaded on first request) ──────────────────────────────────
# shape_pipe: DiT flow-matching for raw mesh geometry
# paint_pipe: texture synthesis on top of the mesh (optional, can be skipped
#             with with_texture=False to halve VRAM use and ~3x speed)
# rembg: background remover applied to input image before shape gen — Hunyuan3D
#        is sensitive to backgrounds; the upstream demo always strips them.
shape_pipe = None
paint_pipe = None
rembg_session = None
model_loading = False
model_error: str | None = None
model_error_traceback: str | None = None


def load_models(with_texture: bool = True):
    """Load shape (and optionally texture paint) pipelines on first call.

    Loading is split so a low-VRAM host can run shape-only by setting
    WITH_TEXTURE=0 — saves ~6GB and avoids touching the texture extensions
    that frequently fail to compile on heterogeneous GPU SKUs."""
    global shape_pipe, paint_pipe, rembg_session, model_loading, model_error, model_error_traceback
    if shape_pipe is not None and (paint_pipe is not None or not with_texture):
        return
    model_loading = True
    log.info(f"Loading Hunyuan3D models from {MODEL_ID} (with_texture={with_texture})...")
    try:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        if shape_pipe is None:
            shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(MODEL_ID)
            log.info("Shape pipeline loaded")
        if with_texture and paint_pipe is None:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            paint_pipe = Hunyuan3DPaintPipeline.from_pretrained(MODEL_ID)
            log.info("Paint pipeline loaded")
        if rembg_session is None:
            from hy3dgen.rembg import BackgroundRemover
            rembg_session = BackgroundRemover()
            log.info("Background remover ready")
        log.info(
            f"Models loaded on {torch.cuda.get_device_name(0)} "
            f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)"
        )
        model_error = None
        model_error_traceback = None
    except Exception as e:
        model_error = f"{type(e).__name__}: {e}"
        model_error_traceback = traceback.format_exc()
        log.error("Failed to load models — full traceback:\n%s", model_error_traceback)
        raise
    finally:
        model_loading = False


def _load_models_thread():
    try:
        load_models(with_texture=WITH_TEXTURE_DEFAULT)
    except Exception as e:
        tb = traceback.format_exc()
        log.error("Background model load failed:\n%s", tb)
        print(f"[BACKGROUND THREAD ERROR] {e}\n{tb}", file=sys.stderr, flush=True)


@app.on_event("startup")
async def startup():
    """Kick off model load in background so /health responds immediately
    while the ~10GB HF download runs."""
    import threading
    log.info("Starting background model load...")
    threading.Thread(target=_load_models_thread, daemon=True).start()
    try:
        from idle_watchdog import start_watchdog
        import asyncio
        asyncio.create_task(start_watchdog())
    except Exception:
        pass


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Never 5xx — always 200 with diagnostic body so the gateway can poll
    even when the model is broken."""
    try:
        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False
    gpu_name = "N/A"
    gpu_mem = "N/A"
    if cuda_ok:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        except Exception as e:
            gpu_name = f"error: {e}"

    if shape_pipe is not None:
        status = "ready"
    elif model_loading:
        status = "loading"
    elif model_error:
        status = "error"
    else:
        status = "starting"

    content = {
        "status": status,
        "model": MODEL_ID,
        "gpu": gpu_name,
        "gpu_memory": gpu_mem,
        "error": model_error,
        "error_traceback": model_error_traceback,
        "config": {
            "steps": STEPS,
            "octree": OCTREE,
            "with_texture_default": WITH_TEXTURE_DEFAULT,
            "shape_loaded": shape_pipe is not None,
            "paint_loaded": paint_pipe is not None,
        },
    }
    try:
        from idle_watchdog import get_health_metrics
        content.update(get_health_metrics())
    except Exception:
        pass
    return JSONResponse(status_code=200, content=content)


@app.get("/diag")
async def diag():
    info: dict = {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "cwd": os.getcwd(),
            "pid": os.getpid(),
        }
    }
    torch_info: dict = {"version": getattr(torch, "__version__", "unknown")}
    try:
        torch_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            torch_info["device_count"] = torch.cuda.device_count()
            torch_info["device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            torch_info["total_memory_gb"] = round(props.total_memory / 1e9, 2)
    except Exception as e:
        torch_info["error"] = f"{type(e).__name__}: {e}"
    info["torch"] = torch_info

    env_allow_prefixes = (
        "HUNYUAN3D_", "HOST", "PORT", "CUDA_", "NVIDIA_", "PYTORCH_",
        "HF_", "HUGGINGFACE_", "WITH_TEXTURE", "PYTHONPATH",
    )
    info["env"] = {
        k: v for k, v in os.environ.items() if k.startswith(env_allow_prefixes)
    }
    info["model"] = {
        "id": MODEL_ID,
        "shape_loaded": shape_pipe is not None,
        "paint_loaded": paint_pipe is not None,
        "loading": model_loading,
        "error": model_error,
        "error_traceback": model_error_traceback,
    }

    log_tail: list[str] = []
    log_path = "/var/log/app.log"
    try:
        if os.path.exists(log_path):
            with open(log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                read_bytes = min(size, 64 * 1024)
                f.seek(size - read_bytes)
                tail_text = f.read().decode("utf-8", errors="replace")
                log_tail = tail_text.splitlines()[-200:]
    except Exception as e:
        log_tail = [f"[diag: failed to read {log_path}: {e}]"]
    info["app_log_tail"] = log_tail

    files_to_check = [
        "/app/server.py", "/app/start.sh", "/app/Hunyuan3D-2",
        "/app/Hunyuan3D-2/hy3dgen", "/var/log/app.log",
    ]
    info["files"] = {p: os.path.exists(p) for p in files_to_check}
    return JSONResponse(status_code=200, content=info)


@app.get("/logs")
async def logs(lines: int = Query(500, ge=1, le=5000)):
    log_path = "/var/log/app.log"
    if not os.path.exists(log_path):
        return JSONResponse(status_code=200, content={"error": f"{log_path} missing"})
    try:
        max_bytes = max(64 * 1024, lines * 200)
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            read_bytes = min(size, max_bytes)
            f.seek(size - read_bytes)
            text = f.read().decode("utf-8", errors="replace")
        tail = "\n".join(text.splitlines()[-lines:])
        return PlainTextResponse(content=tail, status_code=200)
    except Exception as e:
        return JSONResponse(status_code=200, content={"error": f"failed: {e}"})


def _generate_glb_from_image(image: Image.Image, seed: int, with_texture: bool) -> tuple[str, dict]:
    if shape_pipe is None or (with_texture and paint_pipe is None):
        load_models(with_texture=with_texture)

    t0 = time.time()
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    # Strip background — Hunyuan3D shape gen breaks on cluttered backdrops.
    if rembg_session is not None and image.getextrema()[3][0] == 255:
        image = rembg_session(image)

    log.info(f"Shape gen (steps={STEPS}, octree={OCTREE}, seed={seed}) ...")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    mesh = shape_pipe(
        image=image,
        num_inference_steps=STEPS,
        octree_resolution=OCTREE,
        generator=generator,
    )[0]

    if with_texture and paint_pipe is not None:
        log.info("Paint gen ...")
        mesh = paint_pipe(mesh, image=image)

    tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
    mesh.export(tmp.name)
    tmp.close()

    elapsed = time.time() - t0
    stats = {
        "elapsed_sec": round(elapsed, 2),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "with_texture": with_texture,
        "seed": seed,
    }
    log.info(f"Generated: {stats['vertices']} verts, {stats['faces']} faces in {elapsed:.1f}s")
    return tmp.name, stats


def _generate_glb_from_text(prompt: str, seed: int, with_texture: bool) -> tuple[str, dict]:
    """Text-to-3D: render a reference image from prompt via Hunyuan-DiT, then
    feed it through the standard image-to-3D pipeline. Mirrors the upstream
    text-to-3D demo flow."""
    from hy3dgen.text2image import HunyuanDiTPipeline
    log.info(f"Text-to-image: '{prompt[:60]}'")
    t2i = HunyuanDiTPipeline("Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled")
    image = t2i(prompt)
    return _generate_glb_from_image(image, seed, with_texture)


@app.post("/generate")
async def generate(
    file: UploadFile = File(...),
    seed: int = Query(0),
    with_texture: bool = Query(WITH_TEXTURE_DEFAULT),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "file must be an image")
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    try:
        glb_path, stats = _generate_glb_from_image(image, seed, with_texture)
    except Exception as e:
        log.error(f"Generation failed: {e}")
        raise HTTPException(500, f"generation failed: {e}")
    return FileResponse(
        glb_path,
        media_type="model/gltf-binary",
        filename=f"hunyuan3d-{seed}.glb",
        headers={
            "X-Hunyuan3d-Elapsed": str(stats["elapsed_sec"]),
            "X-Hunyuan3d-Vertices": str(stats["vertices"]),
            "X-Hunyuan3d-Faces": str(stats["faces"]),
        },
    )


@app.post("/generate-from-url")
async def generate_from_url(body: dict):
    image_url = body.get("image_url")
    seed = int(body.get("seed", 0))
    with_texture = bool(body.get("with_texture", WITH_TEXTURE_DEFAULT))
    if not image_url:
        raise HTTPException(400, "image_url is required")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(image_url)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content))
    except Exception as e:
        raise HTTPException(400, f"failed to download image: {e}")
    try:
        glb_path, stats = _generate_glb_from_image(image, seed, with_texture)
    except Exception as e:
        log.error(f"Generation failed: {e}")
        raise HTTPException(500, f"generation failed: {e}")
    return FileResponse(
        glb_path,
        media_type="model/gltf-binary",
        filename=f"hunyuan3d-{seed}.glb",
        headers={
            "X-Hunyuan3d-Elapsed": str(stats["elapsed_sec"]),
            "X-Hunyuan3d-Vertices": str(stats["vertices"]),
            "X-Hunyuan3d-Faces": str(stats["faces"]),
        },
    )


@app.post("/generate-from-text")
async def generate_from_text(body: dict):
    prompt = body.get("prompt")
    seed = int(body.get("seed", 0))
    with_texture = bool(body.get("with_texture", WITH_TEXTURE_DEFAULT))
    if not prompt:
        raise HTTPException(400, "prompt is required")
    try:
        glb_path, stats = _generate_glb_from_text(prompt, seed, with_texture)
    except Exception as e:
        log.error(f"Text generation failed: {e}")
        raise HTTPException(500, f"generation failed: {e}")
    return FileResponse(
        glb_path,
        media_type="model/gltf-binary",
        filename=f"hunyuan3d-text-{seed}.glb",
        headers={
            "X-Hunyuan3d-Elapsed": str(stats["elapsed_sec"]),
            "X-Hunyuan3d-Vertices": str(stats["vertices"]),
            "X-Hunyuan3d-Faces": str(stats["faces"]),
        },
    )


if __name__ == "__main__":
    log.info(f"Starting uvicorn on {HOST}:{PORT}")
    log.info(f"Python {sys.version}")
    try:
        log.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log.info(
                f"GPU: {torch.cuda.get_device_name(0)} "
                f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)"
            )
    except Exception as e:
        log.warning(f"CUDA introspection failed: {e}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
