"""
LAM — Large Avatar Model  (SIGGRAPH 2025)
FastAPI server wrapping the LAM inference pipeline.

Endpoints:
  GET  /health                   — service status + model readiness
  GET  /version                  — model info
  POST /v1/avatar/generate       — photo → animated MP4 (multipart upload)
  POST /v1/avatar/generate-b64   — base64 photo → base64 MP4 (JSON)

Input (multipart/form-data):
  file       — image file (jpg/png, frontal face)
  motion     — motion preset name inside assets/sample_motion/export/ (default: first available)
  fps        — output FPS (default: 30)
  width/height — render size (default: 512x512)
"""

import os
import sys
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
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

# ── LAM repo on path ─────────────────────────────────────────────────────────
LAM_DIR = Path("/app/lam")
# chdir so LAM's internal relative paths (vhap/, external/, model_zoo/) resolve
os.chdir(str(LAM_DIR))
sys.path.insert(0, str(LAM_DIR))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("lam")

# ── Config ───────────────────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR   = LAM_DIR / "model_zoo/lam_models/releases/lam/lam-20k/step_045500"
CONFIG_PATH = LAM_DIR / "configs/inference/lam-20k-8gpu.yaml"
MOTIONS_DIR = LAM_DIR / "assets/sample_motion/export"
TRACKING_MODELS_DIR = LAM_DIR / "model_zoo/flame_tracking_models"

DEFAULT_FPS = int(os.environ.get("DEFAULT_FPS", "30"))

# ── Global state ─────────────────────────────────────────────────────────────
lam_model     = None
flame_tracker = None
source_size   = 512
render_size   = 512
boot_time     = time.time()
service_status = {"model": "pending"}


def load_model():
    global lam_model, flame_tracker, source_size, render_size, service_status
    try:
        from omegaconf import OmegaConf
        from lam.models import ModelLAM
        from safetensors.torch import load_file

        log.info("[lam] loading config from %s", CONFIG_PATH)
        cfg = OmegaConf.load(str(CONFIG_PATH))

        source_size = cfg.dataset.source_image_res         # 512
        render_size = cfg.dataset.render_image.high        # 512

        ckpt_path = CKPT_DIR / "model.safetensors"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        log.info("[lam] loading model weights from %s", ckpt_path)
        model = ModelLAM(**cfg.model)
        ckpt = load_file(str(ckpt_path), device="cpu")

        state_dict = model.state_dict()
        for k, v in ckpt.items():
            if k in state_dict and state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
        model.to(DEVICE).eval()
        lam_model = model
        log.info("[lam] model ready on %s", DEVICE)

        # ── FLAME tracker ────────────────────────────────────────────────────
        from tools.flame_tracking_single_image import FlameTrackingSingleImage
        log.info("[lam] loading FLAME tracker...")
        flame_tracker = FlameTrackingSingleImage(
            output_dir=str(LAM_DIR / "output/tracking"),
            alignment_model_path=str(TRACKING_MODELS_DIR / "68_keypoints_model.pkl"),
            vgghead_model_path=str(TRACKING_MODELS_DIR / "vgghead/vgg_heads_l.trcd"),
            human_matting_path=str(TRACKING_MODELS_DIR / "matting/stylematte_synth.pt"),
            facebox_model_path=str(TRACKING_MODELS_DIR / "FaceBoxesV2.pth"),
            detect_iris_landmarks=False,
        )
        log.info("[lam] FLAME tracker ready")

        service_status["model"] = "ready"
        log.info("[lam] all components ready")

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
    model = service_status.get("model", "pending")
    if model == "ready":
        status_str  = "ok"
        status_code = 200
        error_msg   = None
    elif isinstance(model, str) and model.startswith("error:"):
        # Permanent load failure — HTTP 200 so the gateway can read JSON
        # and fail-fast with the actual error message (not a generic timeout).
        status_str  = "error"
        status_code = 200
        error_msg   = model
    else:
        # Still loading — HTTP 200 keeps the gateway in polling mode
        # (avoids the 5-min boot timeout that triggers on repeated 503s).
        status_str  = "loading"
        status_code = 200
        error_msg   = None

    return JSONResponse(
        status_code=status_code,
        content={
            "status":     status_str,
            "service":    "lam",
            "device":     DEVICE,
            "uptime_s":   round(time.time() - boot_time, 1),
            "error":      error_msg,
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
    """Return the flame_param subdirectory of a motion sequence."""
    if not MOTIONS_DIR.exists():
        raise FileNotFoundError(f"Motion sequences not found at {MOTIONS_DIR}")

    entries = sorted(p for p in MOTIONS_DIR.iterdir() if p.is_dir())
    if not entries:
        raise FileNotFoundError("No motion sequences found in assets/sample_motion/export/")

    if motion:
        match = next((e for e in entries if motion in e.name), None)
        if match:
            flame_dir = match / "flame_param"
            if flame_dir.exists():
                return flame_dir
            return match

    # Default: first entry
    first = entries[0]
    flame_dir = first / "flame_param"
    return flame_dir if flame_dir.exists() else first


def run_inference(image_path: str, motion: str, fps: int, width: int, height: int) -> bytes:
    import cv2
    from lam.runners.infer.head_utils import preprocess_image, prepare_motion_seqs

    tmp = tempfile.mkdtemp()
    try:
        # ── 1. FLAME tracking ─────────────────────────────────────────────
        log.info("[lam] FLAME tracking: %s", image_path)
        rc = flame_tracker.preprocess(image_path)
        if rc != 0:
            raise RuntimeError(f"FLAME preprocess failed (code {rc})")
        rc = flame_tracker.optimize()
        if rc != 0:
            raise RuntimeError(f"FLAME optimize failed (code {rc})")
        rc, tracked_dir = flame_tracker.export()
        if rc != 0:
            raise RuntimeError(f"FLAME export failed (code {rc})")

        tracked_img  = os.path.join(tracked_dir, "images/00000_00.png")
        tracked_mask = os.path.join(tracked_dir, "fg_masks/00000_00.png")
        log.info("[lam] tracked image: %s", tracked_img)

        # ── 2. Preprocess source image ────────────────────────────────────
        image, _, _, shape_param = preprocess_image(
            tracked_img, mask_path=tracked_mask,
            intr=None, pad_ratio=0, bg_color=1.,
            max_tgt_size=None, aspect_standard=1.0, enlarge_ratio=[1.0, 1.0],
            render_tgt_size=source_size, multiply=14,
            need_mask=True, get_shape_param=True,
        )

        # ── 3. Load motion sequence ───────────────────────────────────────
        motion_dir = _find_motion_dir(motion)
        log.info("[lam] using motion: %s", motion_dir)
        motion_seq = prepare_motion_seqs(
            str(motion_dir), image_folder=None, save_root=tmp, fps=fps,
            bg_color=1., aspect_standard=1.0, enlarge_ratio=[1.0, 1.0],
            render_image_res=render_size, multiply=16, need_mask=False,
            vis_motion=False, shape_param=shape_param,
            test_sample=True,   # subsample to ~50 frames
        )
        motion_seq["flame_params"]["betas"] = shape_param.unsqueeze(0)

        # ── 4. Run inference ──────────────────────────────────────────────
        log.info("[lam] running inference...")
        t0 = time.time()
        with torch.no_grad():
            res = lam_model.infer_single_view(
                image.unsqueeze(0).to(DEVICE),
                None, None,
                render_c2ws    = motion_seq["render_c2ws"].to(DEVICE),
                render_intrs   = motion_seq["render_intrs"].to(DEVICE),
                render_bg_colors = motion_seq["render_bg_colors"].to(DEVICE),
                flame_params   = {k: v.to(DEVICE) for k, v in motion_seq["flame_params"].items()},
            )
        log.info("[lam] inference done in %.2fs", time.time() - t0)

        # ── 5. Composite frames ───────────────────────────────────────────
        rgb  = res["comp_rgb"].detach().cpu().numpy()   # [Nv, H, W, 3], 0-1
        mask = res["comp_mask"].detach().cpu().numpy()  # [Nv, H, W, 1/3], 0-1
        if mask.shape[-1] != rgb.shape[-1]:
            mask = np.repeat(mask, rgb.shape[-1], axis=-1)
        mask[mask < 0.5] = 0.0
        rgb = rgb * mask + (1 - mask) * 1.0
        frames = (np.clip(rgb, 0, 1.0) * 255).astype(np.uint8)

        # Resize if needed
        if width != frames.shape[2] or height != frames.shape[1]:
            frames = np.stack([cv2.resize(f, (width, height)) for f in frames])

        # ── 6. Encode to MP4 ──────────────────────────────────────────────
        out = os.path.join(tmp, "avatar.mp4")
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
    file:   UploadFile = File(...),
    motion: str        = Form(""),
    fps:    int        = Form(DEFAULT_FPS),
    width:  int        = Form(512),
    height: int        = Form(512),
):
    if service_status.get("model") != "ready":
        raise HTTPException(503, "Model not ready yet")

    tmp = tempfile.mkdtemp()
    ext = Path(file.filename).suffix or ".jpg"
    img_path = os.path.join(tmp, f"input{ext}")
    try:
        with open(img_path, "wb") as f:
            f.write(await file.read())
        loop = asyncio.get_event_loop()
        video = await loop.run_in_executor(
            None, run_inference, img_path, motion, fps, width, height
        )
    except Exception as e:
        log.error("[lam] error: %s", e, exc_info=True)
        raise HTTPException(500, str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return Response(content=video, media_type="video/mp4",
                    headers={"Content-Disposition": "attachment; filename=avatar.mp4"})


class GenerateB64Request(BaseModel):
    image_b64: str
    motion:    str = ""
    fps:       int = DEFAULT_FPS
    width:     int = 512
    height:    int = 512


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
            None, run_inference, img_path, req.motion, req.fps, req.width, req.height
        )
    except Exception as e:
        log.error("[lam] error: %s", e, exc_info=True)
        raise HTTPException(500, str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return {"video_b64": base64.b64encode(video).decode(), "size_bytes": len(video)}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, workers=1)
