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
        from idle_watchdog import start_watchdog
        asyncio.create_task(start_watchdog())
    except ImportError:
        pass
    yield


app = FastAPI(title="TRAM", lifespan=lifespan)
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
