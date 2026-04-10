"""
LAM — Large Avatar Model  (SIGGRAPH 2025)
FastAPI server wrapping the LAM inference pipeline.

Endpoints:
  GET  /health                   — service status + model readiness
  GET  /version                  — model info
  POST /v1/avatar/generate       — photo → animated MP4 (multipart upload)
  POST /v1/avatar/generate-b64   — base64 photo → base64 MP4 (JSON)

Input (multipart/form-data):
  file         — image file (jpg/png, frontal face)
  motion       — motion preset folder name inside assets/motions/ (default: first available)
  num_frames   — frames to render (default: 30, max: 300)
  fps          — output FPS (default: 25)
  width/height — render size (default: 512x512)
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

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("lam")

# ── Config ───────────────────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR   = LAM_DIR / "model_zoo/lam_models/releases/lam/lam-20k/step_045500"
CONFIG_PATH = LAM_DIR / "configs/inference/lam-20k-8gpu.yaml"

DEFAULT_FPS        = int(os.environ.get("DEFAULT_FPS",        "25"))
DEFAULT_NUM_FRAMES = int(os.environ.get("DEFAULT_NUM_FRAMES", "30"))
DEFAULT_WIDTH      = int(os.environ.get("DEFAULT_WIDTH",      "512"))
DEFAULT_HEIGHT     = int(os.environ.get("DEFAULT_HEIGHT",     "512"))

# ── Global state ─────────────────────────────────────────────────────────────
lam_model  = None
boot_time  = time.time()
service_status = {"model": "pending"}


def load_model():
    global lam_model, service_status
    try:
        from omegaconf import OmegaConf
        from src.models.lam.lam import ModelLAM
        from safetensors.torch import load_file

        log.info("[lam] loading config from %s", CONFIG_PATH)
        cfg = OmegaConf.load(str(CONFIG_PATH))

        ckpt_path = CKPT_DIR / "model.safetensors"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        log.info("[lam] loading model weights from %s", ckpt_path)
        lam_model = ModelLAM(**cfg.model)
        ckpt = load_file(str(ckpt_path), device="cpu")

        state_dict = lam_model.state_dict()
        for k, v in ckpt.items():
            if k in state_dict and state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
        lam_model.to(DEVICE).eval()

        service_status["model"] = "ready"
        log.info("[lam] model ready on %s", DEVICE)

    except Exception as e:
        log.error("[lam] load failed: %s", e, exc_info=True)
        service_status["model"] = f"error: {e}"
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, load_model)
    except Exception:
        pass
    yield


app = FastAPI(title="LAM — Large Avatar Model", lifespan=lifespan)


@app.get("/health")
async def health():
    ready = service_status.get("model") == "ready"
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status":    "ok" if ready else "loading",
            "service":   "lam",
            "device":    DEVICE,
            "uptime_s":  round(time.time() - boot_time, 1),
            "components": service_status,
        },
    )


@app.get("/version")
async def version():
    return {
        "service":  "lam",
        "ckpt_dir": str(CKPT_DIR),
        "device":   DEVICE,
        "torch":    torch.__version__,
        "cuda":     torch.version.cuda if torch.cuda.is_available() else None,
    }


def _find_motion_dir(motion: Optional[str]) -> Path:
    """Return a motion sequence dir from LAM assets."""
    # LAM assets are extracted to /app/lam — look for typical motion dirs
    candidates = [
        LAM_DIR / "assets" / "motion_seqs",
        LAM_DIR / "assets" / "motions",
        LAM_DIR / "data" / "motion_seqs",
    ]
    for base in candidates:
        if base.exists():
            entries = sorted(base.iterdir())
            if motion:
                match = next((e for e in entries if motion in e.name), None)
                if match:
                    return match
            if entries:
                return entries[0]

    # Fallback: scan entire assets tree for any dir with .npy files
    for p in sorted((LAM_DIR / "assets").rglob("*.npy")):
        return p.parent

    raise FileNotFoundError("No motion sequences found. Check LAM assets extraction.")


def run_inference(image_path: str, motion: str, num_frames: int,
                  fps: int, width: int, height: int) -> bytes:
    import cv2
    from omegaconf import OmegaConf
    from src.utils.train_util import preprocess_image
    from src.utils.camera_util import prepare_motion_seqs
    from src.flame_tracking.flame_tracking import FlameTrackingSingleImage

    cfg = OmegaConf.load(str(CONFIG_PATH))

    # FLAME tracking
    log.info("[lam] FLAME tracking: %s", image_path)
    human_model_path = str(LAM_DIR / "pretrained_models" / "human_model_files")
    tracker = FlameTrackingSingleImage(human_model_path)
    processed_img, processed_mask, flame_params = tracker.process(image_path)

    # Preprocess image
    source_size = getattr(cfg.model, "source_size", 512)
    image_tensor = preprocess_image(processed_img, processed_mask, source_size)
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    # Motion sequence
    motion_dir = _find_motion_dir(motion)
    log.info("[lam] using motion: %s", motion_dir)
    render_c2ws, render_intrs, render_bg_colors, flame_motion = prepare_motion_seqs(
        str(motion_dir), num_frames=num_frames, device=DEVICE
    )

    # Inference
    log.info("[lam] running inference (%d frames %dx%d)...", num_frames, width, height)
    t0 = time.time()
    with torch.no_grad():
        res = lam_model.infer_single_view(
            image_tensor, render_c2ws, render_intrs, render_bg_colors,
            flame_params={
                "betas":      flame_params.get("betas"),
                "expression": flame_motion.get("expression"),
                "jaw_pose":   flame_motion.get("jaw_pose"),
                "global_rot": flame_motion.get("global_rot"),
            },
        )
    log.info("[lam] inference done in %.2fs", time.time() - t0)

    frames_rgb  = res["comp_rgb"]
    frames_mask = res["comp_mask"]

    frames = []
    for i in range(frames_rgb.shape[0]):
        rgb   = (frames_rgb[i].cpu().numpy()  * 255).clip(0, 255).astype(np.uint8)
        mask  = (frames_mask[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        alpha = mask[..., :1] / 255.0
        frame = (rgb * alpha + 255 * (1 - alpha)).astype(np.uint8)
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        frames.append(frame)

    tmp = tempfile.mkdtemp()
    out = os.path.join(tmp, "avatar.mp4")
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out, fourcc, fps, (width, height))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        with open(out, "rb") as f:
            return f.read()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.post("/v1/avatar/generate")
async def generate(
    file:       UploadFile = File(...),
    motion:     str        = Form(""),
    num_frames: int        = Form(DEFAULT_NUM_FRAMES),
    fps:        int        = Form(DEFAULT_FPS),
    width:      int        = Form(DEFAULT_WIDTH),
    height:     int        = Form(DEFAULT_HEIGHT),
):
    if service_status.get("model") != "ready":
        raise HTTPException(503, "Model not ready yet")

    num_frames = min(max(num_frames, 1), 300)
    tmp = tempfile.mkdtemp()
    ext = Path(file.filename).suffix or ".jpg"
    img_path = os.path.join(tmp, f"input{ext}")
    try:
        with open(img_path, "wb") as f:
            f.write(await file.read())
        loop = asyncio.get_event_loop()
        video = await loop.run_in_executor(
            None, run_inference, img_path, motion, num_frames, fps, width, height
        )
    except Exception as e:
        log.error("[lam] error: %s", e, exc_info=True)
        raise HTTPException(500, str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return Response(content=video, media_type="video/mp4",
                    headers={"Content-Disposition": "attachment; filename=avatar.mp4"})


class GenerateB64Request(BaseModel):
    image_b64:  str
    motion:     str = ""
    num_frames: int = DEFAULT_NUM_FRAMES
    fps:        int = DEFAULT_FPS
    width:      int = DEFAULT_WIDTH
    height:     int = DEFAULT_HEIGHT


@app.post("/v1/avatar/generate-b64")
async def generate_b64(req: GenerateB64Request):
    if service_status.get("model") != "ready":
        raise HTTPException(503, "Model not ready yet")

    tmp = tempfile.mkdtemp()
    img_path = os.path.join(tmp, "input.jpg")
    try:
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(req.image_b64))
        loop = asyncio.get_event_loop()
        video = await loop.run_in_executor(
            None, run_inference, img_path, req.motion,
            req.num_frames, req.fps, req.width, req.height
        )
    except Exception as e:
        log.error("[lam] error: %s", e, exc_info=True)
        raise HTTPException(500, str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return {"video_b64": base64.b64encode(video).decode(), "size_bytes": len(video)}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, workers=1)
