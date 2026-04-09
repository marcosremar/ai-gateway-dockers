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
import sys
import io
import time
import logging
import tempfile
import traceback
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
                 f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
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
    """Health check endpoint.

    IMPORTANT: This endpoint MUST NEVER return a 5xx. It is what the
    gateway / deployment pipeline polls to know if the container is up,
    so even when torch / cuda / the model are broken, we still want to
    return 200 with diagnostic info so the caller can see *why* the
    container is not ready. Any failure here is caught and embedded into
    the response body instead of bubbling up as an exception."""
    try:
        cuda_ok = torch.cuda.is_available()
    except Exception as e:
        cuda_ok = False
        log.warning(f"/health: torch.cuda.is_available() raised: {e}")

    gpu_name = "N/A"
    gpu_mem = "N/A"
    if cuda_ok:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception as e:
            gpu_name = f"error: {e}"
        try:
            gpu_mem = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        except Exception as e:
            gpu_mem = f"error: {e}"

    try:
        if pipeline is not None:
            status = "ready"
        elif model_loading:
            status = "loading"
        elif model_error:
            status = "error"
        else:
            status = "starting"
    except Exception as e:
        status = f"unknown: {e}"

    try:
        return JSONResponse(
            status_code=200,
            content={
                "status": status,
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
            },
        )
    except Exception as e:
        # Absolute last-resort fallback so /health literally never 5xxes.
        return JSONResponse(
            status_code=200,
            content={"status": "degraded", "error": f"/health crashed: {e}"},
        )


@app.get("/diag")
async def diag():
    """Full diagnostic dump for debugging broken deployments.

    Returns GPU info, environment variables (filtered), model status,
    and file existence checks. Used when /health says the server is up
    but generation still fails — gives the operator everything they need
    without having to SSH in."""
    info: dict = {}

    # ── Python / process ─────────────────────────────────────────────
    info["python"] = {
        "version": sys.version,
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "pid": os.getpid(),
    }

    # ── Torch / CUDA ─────────────────────────────────────────────────
    torch_info: dict = {"version": getattr(torch, "__version__", "unknown")}
    try:
        torch_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            torch_info["device_count"] = torch.cuda.device_count()
            torch_info["device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            torch_info["total_memory_gb"] = round(props.total_memory / 1e9, 2)
            torch_info["major"] = props.major
            torch_info["minor"] = props.minor
    except Exception as e:
        torch_info["error"] = f"{type(e).__name__}: {e}"
    info["torch"] = torch_info

    # ── Environment (curated, no secrets) ────────────────────────────
    env_allow_prefixes = (
        "TRELLIS_", "HOST", "PORT", "CUDA_", "NVIDIA_", "PYTORCH_",
        "HF_", "HUGGINGFACE_", "ATTN_", "LD_LIBRARY_PATH", "PATH", "PYTHONPATH",
    )
    info["env"] = {
        k: v for k, v in os.environ.items()
        if k.startswith(env_allow_prefixes)
    }

    # ── Model status ─────────────────────────────────────────────────
    info["model"] = {
        "id": MODEL_ID,
        "loaded": pipeline is not None,
        "loading": model_loading,
        "error": model_error,
    }

    # ── File existence checks ────────────────────────────────────────
    files_to_check = [
        "/app/server.py",
        "/app/start.sh",
        "/app/trellis2",
        "/app/trellis2/o-voxel",
        "/var/log/app.log",
    ]
    info["files"] = {p: os.path.exists(p) for p in files_to_check}

    # ── HF cache size (so we know if the model actually downloaded) ─
    hf_cache_candidates = [
        os.environ.get("HF_HOME"),
        os.path.expanduser("~/.cache/huggingface"),
        "/root/.cache/huggingface",
    ]
    hf_cache_info: dict = {}
    for c in hf_cache_candidates:
        if c and os.path.exists(c):
            try:
                total = 0
                for root, _dirs, files in os.walk(c):
                    for f in files:
                        try:
                            total += os.path.getsize(os.path.join(root, f))
                        except OSError:
                            pass
                hf_cache_info[c] = {"exists": True, "size_gb": round(total / 1e9, 2)}
            except Exception as e:
                hf_cache_info[c] = {"exists": True, "error": str(e)}
    info["hf_cache"] = hf_cache_info

    return JSONResponse(status_code=200, content=info)


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
