"""
SMPLest-X FastAPI server — whole-body SMPL-X estimation with hand tracking.

Exposes /health and /predict endpoints compatible with the video2anim pipeline.
Output: SMPL-X 55-joint quaternions + 3D positions (body + hands + face).

Key advantage over HybrIK: full hand articulation (15 joints per hand),
critical for sign language (LIBRAS) applications.

Deploy via ai-gateway:
  bunx ai-gateway gpu deploy --image marcosremar/smplest-x:latest
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

logging.basicConfig(level=logging.INFO, format="[smplest-x] %(message)s")
log = logging.getLogger(__name__)

# ─── Global model state ─────────────────────────────────────────────────────
model = None
device = None


def download_checkpoint():
    ckpt = os.environ.get("SMPLESTX_CHECKPOINT",
        "/app/SMPLest-X/pretrained_models/smplest_x_h/smplest_x_h.pth.tar")
    if os.path.exists(ckpt):
        return ckpt
    log.info("Downloading SMPLest-X checkpoint (8.2GB)...")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download("waanqii/SMPLest-X", "smplest_x_h.pth.tar",
            local_dir=os.path.dirname(ckpt))
        log.info("Checkpoint downloaded")
    except Exception as e:
        log.error(f"Download failed: {e}")
    return ckpt


def load_model():
    """Load SMPLest-X model with mmpose backend."""
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Loading SMPLest-X on {device}...")

    download_checkpoint()

    # SMPLest-X uses mmpose's top-down approach
    # Config and checkpoint paths (baked into Docker image)
    config_path = os.environ.get(
        "SMPLESTX_CONFIG",
        "/app/SMPLest-X/main/config/config_smpler_x_b32.py",
    )
    checkpoint_path = os.environ.get(
        "SMPLESTX_CHECKPOINT",
        "/app/checkpoints/smplest_x_b32.pth",
    )

    try:
        from mmpose.apis import init_model as init_pose_model
        model = init_pose_model(config_path, checkpoint_path, device=str(device))
        model.eval()
        log.info("SMPLest-X model loaded successfully")
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        # Try alternative loading path
        sys.path.insert(0, "/app/SMPLest-X/main")
        from utils.inference_utils import load_model as load_smplestx
        model = load_smplestx(config_path, checkpoint_path, device)
        log.info("SMPLest-X loaded via alternative path")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    try:
        from idle_watchdog import start_watchdog
        asyncio.create_task(start_watchdog())
    except ImportError:
        pass
    yield


app = FastAPI(title="SMPLest-X", lifespan=lifespan)
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


def decode_image(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(img)


def rotmat_to_quat_wxyz(rotmats: np.ndarray) -> list:
    """Convert rotation matrices (N,3,3) to quaternions [w,x,y,z] list."""
    quats = []
    for R in rotmats:
        r = Rotation.from_matrix(R)
        q = r.as_quat()  # [x,y,z,w] scipy convention
        quats.append([float(q[3]), float(q[0]), float(q[1]), float(q[2])])  # → [w,x,y,z]
    return quats


def axis_angle_to_quat_wxyz(aa: np.ndarray) -> list:
    """Convert axis-angle (N,3) to quaternions [w,x,y,z]."""
    quats = []
    for v in aa:
        r = Rotation.from_rotvec(v)
        q = r.as_quat()  # [x,y,z,w]
        quats.append([float(q[3]), float(q[0]), float(q[1]), float(q[2])])
    return quats



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
        "model": "SMPLest-X",
        **_get_idle_metrics(),
        "capabilities": ["body", "hands", "face"],
    }


@app.post("/predict")
async def predict(req: PredictRequest):
    if model is None:
        return {"error": "model not loaded"}, 503

    t0 = time.time()
    img = decode_image(req.image_base64)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    with torch.no_grad():
        # SMPLest-X inference — returns SMPL-X parameters
        # The exact API depends on the model variant, but the output is:
        #   body_pose: (1, 21, 3, 3) rotation matrices or (1, 63) axis-angle
        #   left_hand_pose: (1, 15, 3) axis-angle
        #   right_hand_pose: (1, 15, 3) axis-angle
        #   global_orient: (1, 3) axis-angle
        #   betas: (1, 10)
        #   expression: (1, 10)
        #   jaw_pose: (1, 3)
        #   joints_3d: (1, N, 3)
        results = model(img_bgr)

    # Extract SMPL-X parameters from results
    # Handle both mmpose dict output and direct tensor output
    if isinstance(results, dict):
        pred = results
    elif hasattr(results, "pred_instances"):
        inst = results.pred_instances
        pred = {
            "body_pose": inst.body_pose if hasattr(inst, "body_pose") else None,
            "left_hand_pose": inst.left_hand_pose if hasattr(inst, "left_hand_pose") else None,
            "right_hand_pose": inst.right_hand_pose if hasattr(inst, "right_hand_pose") else None,
            "global_orient": inst.global_orient if hasattr(inst, "global_orient") else None,
            "betas": inst.betas if hasattr(inst, "betas") else None,
            "expression": inst.expression if hasattr(inst, "expression") else None,
            "jaw_pose": inst.jaw_pose if hasattr(inst, "jaw_pose") else None,
            "keypoints_3d": inst.keypoints_3d if hasattr(inst, "keypoints_3d") else None,
        }
    else:
        pred = {}

    # Build unified 55-joint quaternion array (same as HybrIK output format)
    # SMPL-X order: [global, body(21), jaw(1), eyes(2), left_hand(15), right_hand(15)] = 55
    theta_quat = []

    # Joint 0: Global orient (pelvis)
    if pred.get("global_orient") is not None:
        go = np.array(pred["global_orient"]).reshape(-1, 3)
        theta_quat.extend(axis_angle_to_quat_wxyz(go))
    else:
        theta_quat.append([1, 0, 0, 0])

    # Joints 1-21: Body pose
    if pred.get("body_pose") is not None:
        bp = np.array(pred["body_pose"])
        if bp.ndim == 3 and bp.shape[-1] == 3 and bp.shape[-2] == 3:
            # Rotation matrices (N, 3, 3)
            bp = bp.reshape(-1, 3, 3)
            theta_quat.extend(rotmat_to_quat_wxyz(bp))
        else:
            # Axis-angle (N, 3) or (63,)
            bp = bp.reshape(-1, 3)
            theta_quat.extend(axis_angle_to_quat_wxyz(bp))
    else:
        theta_quat.extend([[1, 0, 0, 0]] * 21)

    # Joint 22: Jaw
    if pred.get("jaw_pose") is not None:
        jp = np.array(pred["jaw_pose"]).reshape(-1, 3)
        theta_quat.extend(axis_angle_to_quat_wxyz(jp))
    else:
        theta_quat.append([1, 0, 0, 0])

    # Joints 23-24: Eyes (usually identity)
    theta_quat.extend([[1, 0, 0, 0]] * 2)

    # Joints 25-39: Left hand (15 joints)
    if pred.get("left_hand_pose") is not None:
        lh = np.array(pred["left_hand_pose"]).reshape(-1, 3)
        theta_quat.extend(axis_angle_to_quat_wxyz(lh))
    else:
        theta_quat.extend([[1, 0, 0, 0]] * 15)

    # Joints 40-54: Right hand (15 joints)
    if pred.get("right_hand_pose") is not None:
        rh = np.array(pred["right_hand_pose"]).reshape(-1, 3)
        theta_quat.extend(axis_angle_to_quat_wxyz(rh))
    else:
        theta_quat.extend([[1, 0, 0, 0]] * 15)

    # 3D joints
    joints_3d = None
    if pred.get("keypoints_3d") is not None:
        j3d = np.array(pred["keypoints_3d"])
        if j3d.ndim == 3:
            j3d = j3d[0]  # remove batch dim
        joints_3d = j3d.tolist()

    # Betas
    betas = None
    if pred.get("betas") is not None:
        betas = np.array(pred["betas"]).flatten().tolist()

    # Expression
    expression = None
    if pred.get("expression") is not None:
        expression = np.array(pred["expression"]).flatten().tolist()

    # Translation
    transl = None
    if pred.get("transl") is not None:
        transl = np.array(pred["transl"]).flatten().tolist()
    elif joints_3d and len(joints_3d) > 0:
        transl = joints_3d[0]  # pelvis as root translation

    latency_ms = (time.time() - t0) * 1000

    return {
        "theta_quat": theta_quat,
        "joints_3d": joints_3d,
        "betas": betas,
        "expression": expression,
        "transl": transl,
        "latency_ms": round(latency_ms, 1),
        "model": "smplest-x",
        "has_hands": pred.get("left_hand_pose") is not None,
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
