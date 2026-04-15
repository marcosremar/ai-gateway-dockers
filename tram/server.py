"""
TRAM FastAPI server — global trajectory + multi-person tracking.

Recovers world-space 3D human motion with global trajectory from
monocular video. Supports multi-person scenarios.

Output: SMPL body rotations + global trajectory per frame.
Compatible with video2anim pipeline JSON format.

Deploy via ai-gateway:
  bunx ai-gateway gpu deploy --image marcosremar/tram:latest
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

logging.basicConfig(level=logging.INFO, format="[tram] %(message)s")
log = logging.getLogger(__name__)

model = None
device = None


def load_model():
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Loading TRAM on {device}...")

    sys.path.insert(0, "/app/TRAM")

    try:
        from lib.pipeline import TRAMPipeline
        checkpoint = os.environ.get("TRAM_CHECKPOINT", "/app/checkpoints/tram.pth")
        model = TRAMPipeline(checkpoint, device=device)
        log.info("TRAM pipeline loaded")
    except Exception as e:
        log.error(f"Failed to load TRAM: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    try:
        from idle_watchdog import add_idle_middleware, start_watchdog
        add_idle_middleware(app)
        asyncio.create_task(start_watchdog())
    except ImportError:
        pass
    yield


app = FastAPI(title="TRAM", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    image_base64: str
    include_vertices: bool = False
    flip_test: bool = False


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
    out.append(identity)
    out.extend([identity] * 2)
    out.extend([identity] * 15)
    out.extend([identity] * 15)
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
        "model": "TRAM",
        **_get_idle_metrics(),
        "capabilities": ["body", "trajectory", "multi-person"],
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
        results = model.predict_single(img_bgr)

    body_pose = np.array(results.get("body_pose", np.zeros((23, 3)))).reshape(-1, 3)
    global_orient = np.array(results.get("global_orient", np.zeros(3))).reshape(-1, 3)
    betas = results.get("betas", np.zeros(10))
    joints_3d = results.get("joints_3d")
    transl = results.get("transl", np.zeros(3))

    body_quats = aa_to_quat_wxyz(body_pose)
    global_quat = aa_to_quat_wxyz(global_orient)[0]
    theta_quat = smpl24_to_smplx55(body_quats, global_quat)

    latency_ms = (time.time() - t0) * 1000

    return {
        "theta_quat": theta_quat,
        "joints_3d": joints_3d.tolist() if joints_3d is not None else None,
        "betas": np.array(betas).flatten().tolist(),
        "transl": np.array(transl).flatten().tolist(),
        "latency_ms": round(latency_ms, 1),
        "model": "tram",
        "has_hands": False,
        "n_joints": len(theta_quat),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
