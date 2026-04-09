"""
LAM — Large Avatar Model  (SIGGRAPH 2025)
FastAPI server wrapping the LAM inference pipeline.

Endpoints:
  GET  /health                   — service status + model readiness
  GET  /version                  — model info
  POST /v1/avatar/generate       — photo → animated MP4 (multipart upload)
  POST /v1/avatar/generate-b64   — photo (base64) → frames (base64 list)

Input (multipart/form-data):
  file         — image file (jpg/png, frontal face)
  motion       — motion preset: "idle" | "talking" | "nod" | "shake" (default: "idle")
  num_frames   — number of frames to render (default: 30, max: 300)
  fps          — output FPS (default: 25)
  width        — render width  (default: 512)
  height       — render height (default: 512)

Output:
  video/mp4  — animated head video
"""

import os
import sys
import io
import time
import base64
import logging
import asyncio
import tempfile
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import torch
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

# ── LAM repo on path ─────────────────────────────────────────────────────────
LAM_DIR = Path("/app/lam")
sys.path.insert(0, str(LAM_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("lam")

# ── Config ───────────────────────────────────────────────────────────────────

MODEL_ID   = os.environ.get("MODEL_ID",   "3DAIGC/LAM-20K")
ASSETS_DIR = Path(os.environ.get("ASSETS_DIR", "/app/lam/assets"))
CKPT_DIR   = Path(os.environ.get("CKPT_DIR",   "/app/lam/ckpts/lam-20k"))
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_FPS        = int(os.environ.get("DEFAULT_FPS",        "25"))
DEFAULT_NUM_FRAMES = int(os.environ.get("DEFAULT_NUM_FRAMES", "30"))
DEFAULT_WIDTH      = int(os.environ.get("DEFAULT_WIDTH",      "512"))
DEFAULT_HEIGHT     = int(os.environ.get("DEFAULT_HEIGHT",     "512"))

# Motion preset → assets subfolder mapping
MOTION_PRESETS = {
    "idle":    "motions/idle",
    "talking": "motions/talking",
    "nod":     "motions/nod",
    "shake":   "motions/shake",
}

# ── Global state ─────────────────────────────────────────────────────────────

lam_model     = None
flame_tracker = None
boot_time     = time.time()
service_status = {"model": "pending", "tracker": "pending"}


# ── Startup ──────────────────────────────────────────────────────────────────

def download_model_if_needed():
    """Download LAM-20K checkpoint from HuggingFace if not already present."""
    if CKPT_DIR.exists() and any(CKPT_DIR.iterdir()):
        log.info("[lam] checkpoint already at %s", CKPT_DIR)
        return
    log.info("[lam] downloading checkpoint %s → %s", MODEL_ID, CKPT_DIR)
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_ID, local_dir=str(CKPT_DIR))
    log.info("[lam] checkpoint download complete")


def load_model():
    global lam_model, flame_tracker, service_status

    try:
        download_model_if_needed()

        from omegaconf import OmegaConf
        from src.models.lam.lam import ModelLAM
        from src.flame_tracking.flame_tracking import FlameTrackingSingleImage
        from safetensors.torch import load_file

        # Find config and checkpoint
        cfg_path  = CKPT_DIR / "config.yaml"
        ckpt_path = next(CKPT_DIR.glob("*.safetensors"), None)
        if ckpt_path is None:
            raise FileNotFoundError(f"No .safetensors found in {CKPT_DIR}")

        log.info("[lam] loading config: %s", cfg_path)
        cfg = OmegaConf.load(cfg_path)

        log.info("[lam] loading model from: %s", ckpt_path)
        lam_model = ModelLAM(**cfg.model)
        ckpt = load_file(str(ckpt_path), device="cpu")
        lam_model.load_state_dict(ckpt, strict=False)
        lam_model.to(DEVICE).eval()
        service_status["model"] = "ready"
        log.info("[lam] model ready on %s", DEVICE)

        log.info("[lam] loading FLAME tracker...")
        flame_assets = ASSETS_DIR / "flame"
        flame_tracker = FlameTrackingSingleImage(str(flame_assets))
        service_status["tracker"] = "ready"
        log.info("[lam] tracker ready")

    except Exception as e:
        log.error("[lam] load failed: %s", e, exc_info=True)
        service_status["model"]   = f"error: {e}"
        service_status["tracker"] = f"error: {e}"
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, load_model)
    except Exception:
        pass   # /health will report the error
    yield


app = FastAPI(title="LAM — Large Avatar Model", lifespan=lifespan)


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    ready = all(v == "ready" for v in service_status.values())
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status":  "ok" if ready else "loading",
            "service": "lam",
            "device":  DEVICE,
            "uptime_s": round(time.time() - boot_time, 1),
            "components": service_status,
        },
    )


@app.get("/version")
async def version():
    return {
        "service":   "lam",
        "model_id":  MODEL_ID,
        "device":    DEVICE,
        "torch":     torch.__version__,
        "cuda":      torch.version.cuda if torch.cuda.is_available() else None,
    }


# ── Core inference ───────────────────────────────────────────────────────────

def run_inference(
    image_path: str,
    motion_preset: str,
    num_frames: int,
    fps: int,
    width: int,
    height: int,
) -> bytes:
    """
    Run LAM inference: image → animated MP4 (bytes).
    Runs synchronously (call from thread pool via run_in_executor).
    """
    from src.utils.train_util import preprocess_image
    from src.utils.camera_util import prepare_motion_seqs
    import cv2

    # ── 1. FLAME tracking (preprocess face image) ────────────────────────────
    log.info("[lam] FLAME tracking: %s", image_path)
    processed_img, processed_mask, flame_params = flame_tracker.process(image_path)

    # ── 2. Preprocess image tensor ────────────────────────────────────────────
    source_size = lam_model.cfg.source_size if hasattr(lam_model, "cfg") else 512
    image_tensor = preprocess_image(processed_img, processed_mask, source_size)
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)  # [1, C, H, W]

    # ── 3. Motion sequence ────────────────────────────────────────────────────
    motion_dir = ASSETS_DIR / MOTION_PRESETS.get(motion_preset, "motions/idle")
    if not motion_dir.exists():
        # Fallback: use first available motion
        motions_root = ASSETS_DIR / "motions"
        if motions_root.exists():
            available = sorted(motions_root.iterdir())
            motion_dir = available[0] if available else None
        if motion_dir is None or not motion_dir.exists():
            raise FileNotFoundError(f"No motion sequences found in {ASSETS_DIR}")

    render_c2ws, render_intrs, render_bg_colors, flame_motion = prepare_motion_seqs(
        str(motion_dir), num_frames=num_frames, device=DEVICE
    )

    # ── 4. Inference ──────────────────────────────────────────────────────────
    log.info("[lam] running inference (%d frames, %dx%d)...", num_frames, width, height)
    t0 = time.time()
    with torch.no_grad():
        res = lam_model.infer_single_view(
            image_tensor,
            render_c2ws, render_intrs, render_bg_colors,
            flame_params={
                "betas":      flame_params.get("betas"),
                "expression": flame_motion.get("expression"),
                "jaw_pose":   flame_motion.get("jaw_pose"),
                "global_rot": flame_motion.get("global_rot"),
            },
        )
    log.info("[lam] inference done in %.2fs", time.time() - t0)

    # ── 5. Render frames ──────────────────────────────────────────────────────
    frames_rgb  = res["comp_rgb"]   # [Nv, H, W, 3], float 0-1
    frames_mask = res["comp_mask"]  # [Nv, H, W, 3]

    frames_uint8 = []
    for i in range(frames_rgb.shape[0]):
        rgb  = (frames_rgb[i].cpu().numpy()  * 255).clip(0, 255).astype(np.uint8)
        mask = (frames_mask[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        # Composite on white background
        alpha = mask[..., :1] / 255.0
        frame = (rgb * alpha + 255 * (1 - alpha)).astype(np.uint8)
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        frames_uint8.append(frame)

    # ── 6. Encode MP4 ────────────────────────────────────────────────────────
    tmp = tempfile.mkdtemp()
    out_path = os.path.join(tmp, "avatar.mp4")
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        for frame in frames_uint8:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        with open(out_path, "rb") as f:
            video_bytes = f.read()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    log.info("[lam] encoded %d frames → %d KB", len(frames_uint8), len(video_bytes) // 1024)
    return video_bytes


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/v1/avatar/generate")
async def generate(
    file:        UploadFile  = File(...),
    motion:      str         = Form("idle"),
    num_frames:  int         = Form(DEFAULT_NUM_FRAMES),
    fps:         int         = Form(DEFAULT_FPS),
    width:       int         = Form(DEFAULT_WIDTH),
    height:      int         = Form(DEFAULT_HEIGHT),
):
    """Upload a face photo → receive animated MP4."""
    if not all(v == "ready" for v in service_status.values()):
        raise HTTPException(503, "Model not ready yet")

    if motion not in MOTION_PRESETS and motion != "auto":
        raise HTTPException(400, f"motion must be one of {list(MOTION_PRESETS)}")

    num_frames = min(max(num_frames, 1), 300)

    # Save upload to temp file
    tmp = tempfile.mkdtemp()
    ext = Path(file.filename).suffix or ".jpg"
    img_path = os.path.join(tmp, f"input{ext}")
    try:
        data = await file.read()
        with open(img_path, "wb") as f:
            f.write(data)

        loop = asyncio.get_event_loop()
        video_bytes = await loop.run_in_executor(
            None, run_inference, img_path, motion, num_frames, fps, width, height
        )
    except Exception as e:
        log.error("[lam] inference error: %s", e, exc_info=True)
        raise HTTPException(500, str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return Response(
        content=video_bytes,
        media_type="video/mp4",
        headers={
            "Content-Disposition": "attachment; filename=avatar.mp4",
            "X-Frames": str(num_frames),
            "X-FPS":    str(fps),
        },
    )


class GenerateB64Request(BaseModel):
    image_b64:  str
    motion:     str = "idle"
    num_frames: int = DEFAULT_NUM_FRAMES
    fps:        int = DEFAULT_FPS
    width:      int = DEFAULT_WIDTH
    height:     int = DEFAULT_HEIGHT


@app.post("/v1/avatar/generate-b64")
async def generate_b64(req: GenerateB64Request):
    """Base64 image input → base64-encoded MP4 output (JSON)."""
    if not all(v == "ready" for v in service_status.values()):
        raise HTTPException(503, "Model not ready yet")

    tmp = tempfile.mkdtemp()
    img_path = os.path.join(tmp, "input.jpg")
    try:
        img_bytes = base64.b64decode(req.image_b64)
        with open(img_path, "wb") as f:
            f.write(img_bytes)

        loop = asyncio.get_event_loop()
        video_bytes = await loop.run_in_executor(
            None, run_inference,
            img_path, req.motion, req.num_frames, req.fps, req.width, req.height,
        )
    except Exception as e:
        log.error("[lam] inference error: %s", e, exc_info=True)
        raise HTTPException(500, str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return {
        "video_b64":   base64.b64encode(video_bytes).decode(),
        "num_frames":  req.num_frames,
        "fps":         req.fps,
        "size_bytes":  len(video_bytes),
    }


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, workers=1)
