"""
TRELLIS.2 — FastAPI inference server for Image-to-3D generation.

Endpoints:
    POST /generate          — Upload an image, get back a .GLB file
    POST /generate-from-url — Provide an image URL, get back a .GLB file
    GET  /health            — Health check (model loaded?)

Environment variables:
    TRELLIS_MODEL     — HuggingFace model ID (default: microsoft/TRELLIS.2-4B)
    TRELLIS_STEPS     — Number of diffusion steps per stage (default: 12)
    TRELLIS_RESOLUTION — Voxel resolution: 512, 1024, or 1536 (default: 512)
    TRELLIS_TEXTURE_SIZE — Texture map size in pixels (default: 2048)
    TRELLIS_DECIMATION — Target face count for mesh decimation (default: 500000)
    HOST              — Bind address (default: 0.0.0.0)
    PORT              — Bind port (default: 8000)
"""

import os
import io
import time
import logging
import tempfile
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("trellis2")

# ── Config ──
MODEL_ID = os.environ.get("TRELLIS_MODEL", "microsoft/TRELLIS.2-4B")
STEPS = int(os.environ.get("TRELLIS_STEPS", "12"))
RESOLUTION = int(os.environ.get("TRELLIS_RESOLUTION", "512"))
TEXTURE_SIZE = int(os.environ.get("TRELLIS_TEXTURE_SIZE", "2048"))
DECIMATION = int(os.environ.get("TRELLIS_DECIMATION", "500000"))
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

# ── App ──
app = FastAPI(title="TRELLIS.2 3D Generation Server", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Model (lazy-loaded on first request) ──
pipeline = None
model_loading = False
model_error = None


def load_model():
    global pipeline, model_loading, model_error
    if pipeline is not None:
        return
    model_loading = True
    log.info(f"Loading TRELLIS.2 model: {MODEL_ID} ...")
    try:
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(MODEL_ID)
        pipeline.cuda()
        log.info(f"Model loaded on {torch.cuda.get_device_name(0)} "
                 f"({torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB)")
        model_loading = False
    except Exception as e:
        model_error = str(e)
        model_loading = False
        log.error(f"Failed to load model: {e}")
        raise


def _load_model_thread():
    """Background thread that loads the model without blocking startup."""
    try:
        load_model()
    except Exception as e:
        log.error(f"Background model load failed: {e}")


@app.on_event("startup")
async def startup():
    """Start the model load in a background thread so /health responds
    immediately. The gateway's health check uses /health which returns
    {status: 'loading'} until the model is ready. Without this, the 15GB
    HuggingFace download blocks the FastAPI bind() and the gateway times
    out before the port is even open."""
    import threading
    log.info("Starting background model load...")
    threading.Thread(target=_load_model_thread, daemon=True).start()


# ── Endpoints ──

@app.get("/health")
async def health():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    gpu_mem = f"{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB" if torch.cuda.is_available() else "N/A"
    return {
        "status": "ready" if pipeline is not None else "loading" if model_loading else "error",
        "model": MODEL_ID,
        "gpu": gpu_name,
        "gpu_memory": gpu_mem,
        "error": model_error,
        "config": {
            "steps": STEPS,
            "resolution": RESOLUTION,
            "texture_size": TEXTURE_SIZE,
            "decimation": DECIMATION,
        },
    }


def _generate_glb(image: Image.Image, seed: int = 0) -> tuple[str, dict]:
    """Run the TRELLIS.2 pipeline on a PIL image and return (.glb path, metadata)."""
    if pipeline is None:
        load_model()

    t0 = time.time()

    # Preprocess: ensure RGBA, remove background if needed
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Run the 3-stage pipeline
    log.info(f"Generating 3D mesh (resolution={RESOLUTION}, steps={STEPS}, seed={seed}) ...")
    outputs = pipeline.run(
        image,
        seed=seed,
        sparse_structure_sampler_params={"steps": STEPS},
        slat_sampler_params={"steps": STEPS},
    )

    # Export to GLB
    from o_voxel.postprocess import to_glb
    mesh = outputs[0]
    glb = to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        decimation_target=DECIMATION,
        texture_size=TEXTURE_SIZE,
        remesh=True,
    )

    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
    glb.export(tmp.name, extension_webp=True)
    tmp.close()

    elapsed = time.time() - t0
    stats = {
        "elapsed_sec": round(elapsed, 2),
        "vertices": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]),
        "resolution": RESOLUTION,
        "texture_size": TEXTURE_SIZE,
        "seed": seed,
    }
    log.info(f"Generated: {stats['vertices']} verts, {stats['faces']} faces in {elapsed:.1f}s")
    return tmp.name, stats


@app.post("/generate")
async def generate(
    file: UploadFile = File(...),
    seed: int = Query(0, description="Random seed for reproducibility"),
):
    """Upload an image file, receive a .GLB 3D model."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "file must be an image (png, jpg, webp)")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    try:
        glb_path, stats = _generate_glb(image, seed)
    except Exception as e:
        log.error(f"Generation failed: {e}")
        raise HTTPException(500, f"generation failed: {e}")

    return FileResponse(
        glb_path,
        media_type="model/gltf-binary",
        filename=f"trellis2-{stats['seed']}.glb",
        headers={
            "X-Trellis-Elapsed": str(stats["elapsed_sec"]),
            "X-Trellis-Vertices": str(stats["vertices"]),
            "X-Trellis-Faces": str(stats["faces"]),
        },
    )


@app.post("/generate-from-url")
async def generate_from_url(
    body: dict,
):
    """Provide an image URL in the body, receive a .GLB 3D model.
    Body: { "image_url": "https://...", "seed": 0 }
    """
    image_url = body.get("image_url")
    seed = body.get("seed", 0)
    if not image_url:
        raise HTTPException(400, "image_url is required")

    # Download the image
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(image_url)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content))
    except Exception as e:
        raise HTTPException(400, f"failed to download image: {e}")

    try:
        glb_path, stats = _generate_glb(image, seed)
    except Exception as e:
        log.error(f"Generation failed: {e}")
        raise HTTPException(500, f"generation failed: {e}")

    return FileResponse(
        glb_path,
        media_type="model/gltf-binary",
        filename=f"trellis2-{stats['seed']}.glb",
        headers={
            "X-Trellis-Elapsed": str(stats["elapsed_sec"]),
            "X-Trellis-Vertices": str(stats["vertices"]),
            "X-Trellis-Faces": str(stats["faces"]),
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
