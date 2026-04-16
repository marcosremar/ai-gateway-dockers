"""
WHAM FastAPI server — temporal video-based SMPL estimation.

Processes video clips (not single frames) for temporal coherence.
Output: SMPL 24-joint rotations + global trajectory per frame.
Compatible with video2anim pipeline JSON format.

WHAM uses SLAM-based motion integration for world-grounded estimation.
57mm MPJPE on 3DPW — strong temporal coherence, minimal jitter.

Deploy via ai-gateway:
  bunx ai-gateway gpu deploy --image marcosremar/wham:latest
"""

import base64
import io
import logging
import os
import sys
import tempfile
import time
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from scipy.spatial.transform import Rotation

logging.basicConfig(level=logging.INFO, format="[wham] %(message)s")
log = logging.getLogger(__name__)

model = None
detector = None
device = None


def download_checkpoint():
    ckpt = os.environ.get("WHAM_CHECKPOINT", "/app/WHAM/checkpoints/wham_vit_w_3dpw.pth.tar")
    if os.path.exists(ckpt):
        return ckpt
    log.info("Downloading WHAM checkpoint...")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    try:
        import gdown
        gdown.download("1Erjkho7O0bnZFawarntICRUCroaKabRE", ckpt, quiet=False)
        log.info("Checkpoint downloaded")
    except Exception as e:
        log.error(f"Download failed: {e}")
    return ckpt


def load_model():
    global model, detector, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Loading WHAM on {device}...")

    download_checkpoint()
    sys.path.insert(0, "/app/WHAM")

    try:
        from lib.models import build_network
        from lib.utils.config import get_cfg
        cfg = get_cfg()
        cfg.merge_from_file(os.environ.get("WHAM_CONFIG", "/app/WHAM/configs/yamls/demo.yaml"))
        model = build_network(cfg, device)
        model.eval()
        log.info("WHAM model loaded")
    except Exception as e:
        log.error(f"Failed to load WHAM: {e}")
        log.info("Will use fallback inference path")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    try:
        from idle_watchdog import start_watchdog
        asyncio.create_task(start_watchdog())
    except ImportError:
        pass
    yield


app = FastAPI(title="WHAM", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Idle tracking middleware (must be added before app starts)
try:
    from idle_watchdog import add_idle_middleware
    add_idle_middleware(app)
except ImportError:
    pass


class PredictRequest(BaseModel):
    image_base64: str
    include_vertices: bool = False
    flip_test: bool = False


class PredictVideoRequest(BaseModel):
    """Process a batch of frames for temporal coherence."""
    frames_base64: list[str]
    fps: float = 30.0


def aa_to_quat_wxyz(aa: np.ndarray) -> list:
    """Axis-angle (N, 3) → quaternion [w,x,y,z] list."""
    quats = []
    for v in aa:
        r = Rotation.from_rotvec(v)
        q = r.as_quat()  # scipy: [x,y,z,w]
        quats.append([float(q[3]), float(q[0]), float(q[1]), float(q[2])])
    return quats


def smpl24_to_smplx55(body_quats: list, global_quat: list) -> list:
    """Expand SMPL 24-joint to SMPL-X 55-joint (pad hands/face with identity)."""
    # SMPL-X: [global(1), body(21), jaw(1), eyes(2), left_hand(15), right_hand(15)] = 55
    # SMPL:   [global(1), body(23)] — body has 2 extra (hands=wrist rotation)
    identity = [1, 0, 0, 0]
    out = [global_quat]          # 0: pelvis
    out.extend(body_quats[:21])  # 1-21: body
    out.append(identity)         # 22: jaw
    out.extend([identity] * 2)   # 23-24: eyes
    out.extend([identity] * 15)  # 25-39: left hand
    out.extend([identity] * 15)  # 40-54: right hand
    return out



def _get_idle_metrics() -> dict:
    """Get idle tracking metrics for ai-gateway health polling."""
    try:
        from idle_watchdog import get_health_metrics
        return get_health_metrics()
    except ImportError:
        return {}


@app.get("/health")
async def health():
    return {
        "status": "ok" if model is not None else "loading",
        "device": str(device) if device else "unknown",
        "model": "WHAM",
        **_get_idle_metrics(),
        "capabilities": ["body", "temporal", "trajectory"],
    }


@app.post("/predict")
async def predict(req: PredictRequest):
    """Single-frame prediction (falls back to per-frame, less accurate)."""
    if model is None:
        return {"error": "model not loaded"}, 503

    t0 = time.time()
    raw = base64.b64decode(req.image_base64)
    img = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    with torch.no_grad():
        # WHAM expects video sequences, but can process single frames
        # with reduced temporal benefit
        results = model.inference_single(img_bgr)

    # Parse SMPL output
    body_pose = results.get("body_pose", np.zeros((23, 3)))
    global_orient = results.get("global_orient", np.zeros(3))
    betas = results.get("betas", np.zeros(10))
    joints_3d = results.get("joints_3d")
    transl = results.get("transl", np.zeros(3))

    body_pose = np.array(body_pose).reshape(-1, 3)
    global_orient = np.array(global_orient).reshape(-1, 3)

    body_quats = aa_to_quat_wxyz(body_pose)
    global_quat = aa_to_quat_wxyz(global_orient)[0]
    theta_quat = smpl24_to_smplx55(body_quats, global_quat)

    j3d = joints_3d.tolist() if joints_3d is not None else None
    latency_ms = (time.time() - t0) * 1000

    return {
        "theta_quat": theta_quat,
        "joints_3d": j3d,
        "betas": np.array(betas).flatten().tolist(),
        "transl": np.array(transl).flatten().tolist(),
        "latency_ms": round(latency_ms, 1),
        "model": "wham",
        "has_hands": False,
        "n_joints": len(theta_quat),
    }


@app.post("/predict_video")
async def predict_video(req: PredictVideoRequest):
    """Process full video sequence for temporal coherence (WHAM's strength)."""
    if model is None:
        return {"error": "model not loaded"}, 503

    t0 = time.time()

    # Decode all frames
    frames = []
    for b64 in req.frames_base64:
        raw = base64.b64decode(b64)
        img = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
        frames.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    with torch.no_grad():
        results = model.inference_sequence(frames, fps=req.fps)

    # results contains per-frame SMPL params with temporal smoothing baked in
    output_frames = []
    for i in range(len(results.get("body_pose", []))):
        bp = np.array(results["body_pose"][i]).reshape(-1, 3)
        go = np.array(results["global_orient"][i]).reshape(-1, 3)
        body_quats = aa_to_quat_wxyz(bp)
        global_quat = aa_to_quat_wxyz(go)[0]
        theta_quat = smpl24_to_smplx55(body_quats, global_quat)

        j3d = results["joints_3d"][i].tolist() if "joints_3d" in results else None
        tr = results["transl"][i].tolist() if "transl" in results else None

        output_frames.append({
            "theta_quat": theta_quat,
            "joints_3d": j3d,
            "transl": tr,
        })

    latency_ms = (time.time() - t0) * 1000

    return {
        "frames": output_frames,
        "betas": np.array(results.get("betas", np.zeros(10))).flatten().tolist(),
        "fps": req.fps,
        "latency_ms": round(latency_ms, 1),
        "model": "wham",
        "frame_count": len(output_frames),
    }




@app.get("/debug")
async def debug():
    """Diagnostic endpoint — shows checkpoint status, import errors, model state."""
    import traceback
    info = {"device": str(device), "model_loaded": model is not None}
    
    # Check checkpoint
    import glob
    ckpt_dirs = ["/app/checkpoints", "/app/WHAM/checkpoints", "/app/GVHMR/inputs/checkpoints",
                 "/app/SMPLest-X/pretrained_models", "/app/TokenHMR/data", "/app/TRAM/data"]
    for d in ckpt_dirs:
        files = glob.glob(f"{d}/**/*", recursive=True)
        if files:
            info[f"files_in_{d}"] = [f"{f} ({os.path.getsize(f)//1024}KB)" for f in files[:20] if os.path.isfile(f)]
    
    # Try imports
    test_imports = ["scipy", "scipy.spatial.transform", "smplx", "cv2", "torch"]
    for mod in test_imports:
        try:
            __import__(mod)
            info[f"import_{mod}"] = "ok"
        except Exception as e:
            info[f"import_{mod}"] = str(e)[:100]
    
    # Try model-specific import
    try:
        import numpy
        info["numpy_version"] = numpy.__version__
    except: pass
    
    # Capture load_model error
    if model is None:
        try:
            load_model()
            info["reload_result"] = "model loaded!" if model is not None else "still None"
        except Exception as e:
            info["reload_error"] = traceback.format_exc()[-500:]
    
    return info


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
