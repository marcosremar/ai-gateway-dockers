"""
TokenHMR FastAPI server — occlusion-robust pose estimation.

Uses tokenized pose representation for robust handling of partial occlusion.
Key advantage: handles occluded body parts better than regression models.

Output: SMPL body rotations per frame.
Compatible with video2anim pipeline JSON format.

Deploy via ai-gateway:
  bunx ai-gateway gpu deploy --image marcosremar/tokenhmr:latest
"""

import base64
import io
import logging
import os
import sys
import time
import asyncio
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from scipy.spatial.transform import Rotation

logging.basicConfig(level=logging.INFO, format="[tokenhmr] %(message)s")
log = logging.getLogger(__name__)

model = None
device = None


def load_model():
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Loading TokenHMR on {device}...")

    sys.path.insert(0, "/app/TokenHMR")

    try:
        from tokenhmr.models import load_tokenhmr
        checkpoint = os.environ.get("TOKENHMR_CHECKPOINT", "/app/checkpoints/tokenhmr.pth")
        model = load_tokenhmr(checkpoint, device=device)
        model.eval()
        log.info("TokenHMR model loaded")
    except Exception as e:
        log.error(f"Failed to load TokenHMR: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    try:
        from idle_watchdog import start_watchdog
        asyncio.create_task(start_watchdog())
    except ImportError:
        pass
    yield


app = FastAPI(title="TokenHMR", lifespan=lifespan)
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


def rotmat_to_quat_wxyz(rotmats: np.ndarray) -> list:
    quats = []
    for R in rotmats:
        r = Rotation.from_matrix(R)
        q = r.as_quat()
        quats.append([float(q[3]), float(q[0]), float(q[1]), float(q[2])])
    return quats


def aa_to_quat_wxyz(aa: np.ndarray) -> list:
    quats = []
    for v in aa:
        r = Rotation.from_rotvec(v)
        q = r.as_quat()
        quats.append([float(q[3]), float(q[0]), float(q[1]), float(q[2])])
    return quats


def smpl24_to_smplx55(body_quats: list, global_quat: list) -> list:
    identity = [1, 0, 0, 0]
    out = [global_quat]
    out.extend(body_quats[:21])
    out.append(identity)         # jaw
    out.extend([identity] * 2)   # eyes
    out.extend([identity] * 15)  # left hand
    out.extend([identity] * 15)  # right hand
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
        "model": "TokenHMR",
        **_get_idle_metrics(),
        "capabilities": ["body", "occlusion-robust"],
    }


@app.post("/predict")
async def predict(req: PredictRequest):
    if model is None:
        return {"error": "model not loaded"}, 503

    t0 = time.time()
    raw = base64.b64decode(req.image_base64)
    img = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    with torch.no_grad():
        results = model(img_bgr)

    # TokenHMR outputs rotation matrices or axis-angle
    body_pose = results.get("body_pose")
    global_orient = results.get("global_orient")
    betas = results.get("betas", np.zeros(10))
    joints_3d = results.get("joints_3d")
    transl = results.get("transl", np.zeros(3))

    if body_pose is not None:
        bp = np.array(body_pose)
        if bp.ndim == 3 and bp.shape[-1] == 3 and bp.shape[-2] == 3:
            bp = bp.reshape(-1, 3, 3)
            body_quats = rotmat_to_quat_wxyz(bp)
        else:
            bp = bp.reshape(-1, 3)
            body_quats = aa_to_quat_wxyz(bp)
    else:
        body_quats = [[1, 0, 0, 0]] * 23

    if global_orient is not None:
        go = np.array(global_orient)
        if go.ndim >= 2 and go.shape[-1] == 3 and go.shape[-2] == 3:
            global_quat = rotmat_to_quat_wxyz(go.reshape(-1, 3, 3))[0]
        else:
            global_quat = aa_to_quat_wxyz(go.reshape(-1, 3))[0]
    else:
        global_quat = [1, 0, 0, 0]

    theta_quat = smpl24_to_smplx55(body_quats, global_quat)
    latency_ms = (time.time() - t0) * 1000

    return {
        "theta_quat": theta_quat,
        "joints_3d": joints_3d.tolist() if joints_3d is not None else None,
        "betas": np.array(betas).flatten().tolist(),
        "transl": np.array(transl).flatten().tolist(),
        "latency_ms": round(latency_ms, 1),
        "model": "tokenhmr",
        "has_hands": False,
        "n_joints": len(theta_quat),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
