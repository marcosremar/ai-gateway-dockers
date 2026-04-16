"""
SMPLest-X FastAPI server — whole-body SMPL-X estimation with hand tracking.

Exposes /health, /predict, /debug endpoints compatible with the video2anim pipeline.
Output: SMPL-X 55-joint quaternions + 3D positions (body + hands + face).

Key advantage over HybrIK: full hand articulation (15 joints per hand),
critical for sign language (LIBRAS) applications.

Loading strategy: background thread so /health responds immediately.
All weights baked into Docker image — zero runtime downloads.

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
import threading
import traceback
from typing import Optional

# Set working directory to SMPLest-X root for relative paths in config
SMPLESTX_ROOT = os.environ.get("SMPLESTX_ROOT", "/app/SMPLest-X")
os.chdir(SMPLESTX_ROOT)
sys.path.insert(0, SMPLESTX_ROOT)

# FastAPI imports first — server MUST start even if ML deps fail
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="[smplest-x] %(message)s")
log = logging.getLogger(__name__)

# ─── ML imports (lazy, with error capture) ──────────────────────────────────
_import_error = None
try:
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from scipy.spatial.transform import Rotation
    log.info(f"ML imports OK — torch={torch.__version__} cuda={torch.cuda.is_available()}")
except Exception as e:
    _import_error = traceback.format_exc()
    log.error(f"ML import failed: {e}")
    np = None
    torch = None
    cv2 = None

# ─── Global model state ────────────────────────────────────────────────────
_model_ready = False
_load_error = None
_load_progress = "not started"
_demoer = None
_detector = None
_cfg = None
_device = None


def _load_model_sync():
    """Load SMPLest-X model. Runs in background thread so /health responds immediately."""
    global _model_ready, _load_error, _load_progress
    global _demoer, _detector, _cfg, _device

    try:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Device: {_device}")

        # ── Step 1: Load config ──────────────────────────────────────────
        _load_progress = "loading config"
        log.info("Loading config...")

        from main.config import Config

        config_path = os.path.join(SMPLESTX_ROOT,
                                   "pretrained_models/smplest_x_h/config_base.py")
        log.info(f"Config: {config_path} (exists={os.path.exists(config_path)})")
        _cfg = Config.load_config(config_path)

        # Override paths to point to baked-in files
        _cfg.model.pretrained_model_path = os.path.join(
            SMPLESTX_ROOT, "pretrained_models/smplest_x_h/smplest_x_h.pth.tar")
        _cfg.model.human_model_path = os.path.join(
            SMPLESTX_ROOT, "human_models/human_model_files")
        _cfg.inference.detection.model_path = os.path.join(
            SMPLESTX_ROOT, "pretrained_models/yolov8x.pt")

        # Set log dirs (Tester parent class creates these)
        for key in ["output_dir", "model_dir", "log_dir", "result_dir"]:
            _cfg.log[key] = "/tmp/smplest_x_logs"
        os.makedirs("/tmp/smplest_x_logs", exist_ok=True)

        log.info(f"Checkpoint: {_cfg.model.pretrained_model_path} "
                 f"(exists={os.path.exists(_cfg.model.pretrained_model_path)})")
        log.info(f"Human models: {_cfg.model.human_model_path} "
                 f"(exists={os.path.isdir(_cfg.model.human_model_path)})")

        # ── Step 2: Build model (loads 8.2GB checkpoint) ─────────────────
        _load_progress = "building model (loading checkpoint)"
        log.info("Building model via Tester._make_model()...")

        from main.base import Tester
        _demoer = Tester(_cfg)
        _demoer._make_model()
        log.info("Model built successfully")

        # ── Step 3: Load YOLO detector ───────────────────────────────────
        _load_progress = "loading YOLO detector"
        log.info("Loading YOLO detector...")

        from ultralytics import YOLO
        yolo_path = _cfg.inference.detection.model_path
        if not os.path.exists(yolo_path):
            log.info("YOLOv8x not found, will auto-download...")
            yolo_path = "yolov8x.pt"
        _detector = YOLO(yolo_path)
        log.info("YOLO loaded")

        # ── Done ─────────────────────────────────────────────────────────
        _model_ready = True
        _load_error = None
        _load_progress = "ready"
        log.info("SMPLest-X fully loaded and ready!")

    except Exception as e:
        _load_error = traceback.format_exc()
        _load_progress = f"failed: {str(e)[:200]}"
        log.error(f"Failed to load model: {e}")
        log.error(_load_error)


# ─── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(title="SMPLest-X")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Idle tracking middleware (must be added before app starts)
try:
    from idle_watchdog import add_idle_middleware, get_health_metrics
    add_idle_middleware(app)
except ImportError:
    def get_health_metrics():
        return {}


@app.on_event("startup")
async def startup():
    if _import_error:
        log.error("Skipping model load — ML import failed")
        return
    # Load model in background thread
    thread = threading.Thread(target=_load_model_sync, daemon=True)
    thread.start()
    log.info("Model loading started in background thread")
    # Start idle watchdog
    try:
        from idle_watchdog import start_watchdog
        asyncio.create_task(start_watchdog())
    except ImportError:
        pass


# ─── Request/Response models ───────────────────────────────────────────────
class PredictRequest(BaseModel):
    image_base64: str
    include_vertices: bool = False


# ─── Rotation utilities ────────────────────────────────────────────────────
def axis_angle_to_quat_wxyz(aa: "np.ndarray") -> list:
    """Convert axis-angle (N,3) to quaternions [w,x,y,z]."""
    quats = []
    for v in np.array(aa).reshape(-1, 3):
        r = Rotation.from_rotvec(v)
        q = r.as_quat()  # scipy: [x,y,z,w]
        quats.append([float(q[3]), float(q[0]), float(q[1]), float(q[2])])
    return quats


def rotmat_to_quat_wxyz(rotmats: "np.ndarray") -> list:
    """Convert rotation matrices (N,3,3) to quaternions [w,x,y,z]."""
    quats = []
    for R in np.array(rotmats).reshape(-1, 3, 3):
        r = Rotation.from_matrix(R)
        q = r.as_quat()
        quats.append([float(q[3]), float(q[0]), float(q[1]), float(q[2])])
    return quats


def _to_quat(tensor, n_joints: int) -> list:
    """Auto-detect rotation format (axis-angle or rotmat) and convert to quats."""
    arr = tensor.detach().cpu().numpy()
    # Remove batch dim
    if arr.ndim == 4:
        arr = arr[0]
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    # Detect format
    if arr.ndim == 3 and arr.shape[-1] == 3 and arr.shape[-2] == 3:
        # Rotation matrices (N, 3, 3)
        return rotmat_to_quat_wxyz(arr[:n_joints])
    else:
        # Axis-angle (N*3,) or (N, 3)
        return axis_angle_to_quat_wxyz(arr.reshape(-1, 3)[:n_joints])


# ─── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    resp = {
        "status": "error" if _import_error else ("ok" if _model_ready else "loading"),
        "model": "SMPLest-X",
        "model_ready": _model_ready,
        "load_progress": _load_progress,
        "device": str(_device) if _device else "unknown",
        **get_health_metrics(),
        "capabilities": ["body", "hands", "face"],
    }
    if _import_error:
        resp["import_error"] = _import_error[-500:]
    if _load_error:
        resp["load_error"] = _load_error[-1000:]
    return resp


@app.post("/predict")
async def predict(req: PredictRequest):
    if not _model_ready:
        return {"error": "model not loaded", "load_progress": _load_progress}

    t0 = time.time()

    # Decode image
    raw = base64.b64decode(req.image_base64)
    img = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    original_img = img_bgr.copy()

    # ── Person detection with YOLO ──────────────────────────────────────
    det_conf = 0.5
    if _cfg and hasattr(_cfg, "inference"):
        det_conf = _cfg.inference.detection.get("conf", 0.5)

    det_results = _detector.predict(
        img_bgr, device=str(_device), classes=0,
        conf=det_conf, verbose=False)

    if not det_results or len(det_results[0].boxes) == 0:
        return {
            "error": "no person detected",
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }

    # Pick highest-confidence detection
    boxes = det_results[0].boxes
    best_idx = boxes.conf.argmax().item()
    bbox_xyxy = boxes.xyxy[best_idx].cpu().numpy()  # [x1, y1, x2, y2]

    # ── Preprocess: crop + resize ───────────────────────────────────────
    x1, y1, x2, y2 = bbox_xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    bbox_ratio = 1.2
    if _cfg and hasattr(_cfg, "data"):
        bbox_ratio = _cfg.data.get("bbox_ratio", 1.2)
    w *= bbox_ratio
    h *= bbox_ratio

    input_img_shape = (512, 384)
    if _cfg and hasattr(_cfg, "model"):
        input_img_shape = tuple(_cfg.model.get("input_img_shape", (512, 384)))

    # Try SMPLest-X preprocessing (generate_patch_image)
    try:
        from common.utils.preprocessing import generate_patch_image
        bbox_xywh = np.array([cx - w / 2, cy - h / 2, w, h])
        patch_img, _, _ = generate_patch_image(
            original_img, bbox_xywh, False, 1.0, 0.0, input_img_shape)
    except Exception:
        # Fallback: simple crop + resize
        x1c = max(0, int(cx - w / 2))
        y1c = max(0, int(cy - h / 2))
        x2c = min(img_bgr.shape[1], int(cx + w / 2))
        y2c = min(img_bgr.shape[0], int(cy + h / 2))
        crop = img_bgr[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            crop = img_bgr
        patch_img = cv2.resize(crop, (input_img_shape[1], input_img_shape[0]))

    # Transform: normalize to ImageNet stats
    import torchvision.transforms as T
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(patch_img.astype(np.float32) / 255.0)

    # ── Model inference ─────────────────────────────────────────────────
    inputs = {"img": img_tensor.unsqueeze(0).to(_device)}
    targets = {}
    meta_info = {}

    with torch.no_grad():
        out = _demoer.model(inputs, targets, meta_info, "test")

    # ── Extract SMPL-X parameters ───────────────────────────────────────
    # Build unified 55-joint quaternion array matching HybrIK output format:
    # [global(1), body(21), jaw(1), eyes(2), left_hand(15), right_hand(15)]
    theta_quat = []
    out_keys = list(out.keys()) if isinstance(out, dict) else []

    # Joint 0: Global orient / root pose
    for key in ["smplx_root_pose", "global_orient", "root_pose"]:
        if key in out:
            theta_quat.extend(_to_quat(out[key], 1))
            break
    else:
        theta_quat.append([1, 0, 0, 0])

    # Joints 1-21: Body pose
    for key in ["smplx_body_pose", "body_pose"]:
        if key in out:
            theta_quat.extend(_to_quat(out[key], 21))
            break
    else:
        theta_quat.extend([[1, 0, 0, 0]] * 21)

    # Joint 22: Jaw
    for key in ["smplx_jaw_pose", "jaw_pose"]:
        if key in out:
            theta_quat.extend(_to_quat(out[key], 1))
            break
    else:
        theta_quat.append([1, 0, 0, 0])

    # Joints 23-24: Eyes (usually identity)
    theta_quat.extend([[1, 0, 0, 0]] * 2)

    # Joints 25-39: Left hand (15 joints) ★
    has_hands = False
    for key in ["smplx_lhand_pose", "left_hand_pose", "lhand_pose"]:
        if key in out:
            theta_quat.extend(_to_quat(out[key], 15))
            has_hands = True
            break
    else:
        theta_quat.extend([[1, 0, 0, 0]] * 15)

    # Joints 40-54: Right hand (15 joints) ★
    for key in ["smplx_rhand_pose", "right_hand_pose", "rhand_pose"]:
        if key in out:
            theta_quat.extend(_to_quat(out[key], 15))
            has_hands = True
            break
    else:
        theta_quat.extend([[1, 0, 0, 0]] * 15)

    # 3D joints
    joints_3d = None
    for key in ["smplx_joint_cam", "joint_cam", "joints_3d", "keypoints_3d"]:
        if key in out:
            j3d = out[key].detach().cpu().numpy()
            if j3d.ndim == 3:
                j3d = j3d[0]
            joints_3d = j3d.tolist()
            break

    # Shape (betas)
    betas = None
    for key in ["smplx_shape", "betas", "shape"]:
        if key in out:
            betas = out[key].detach().cpu().numpy().flatten().tolist()
            break

    # Expression
    expression = None
    for key in ["smplx_expr", "expression", "expr"]:
        if key in out:
            expression = out[key].detach().cpu().numpy().flatten().tolist()
            break

    # Translation (use pelvis joint if available)
    transl = None
    if joints_3d and len(joints_3d) > 0:
        transl = joints_3d[0]

    latency_ms = (time.time() - t0) * 1000

    return {
        "theta_quat": theta_quat,
        "joints_3d": joints_3d,
        "betas": betas,
        "expression": expression,
        "transl": transl,
        "latency_ms": round(latency_ms, 1),
        "model": "smplest-x",
        "has_hands": has_hands,
        "n_joints": len(theta_quat),
        "bbox": bbox_xyxy.tolist(),
        "output_keys": out_keys,
    }


@app.get("/debug")
async def debug():
    """Diagnostic endpoint — shows file status, imports, model state."""
    info = {
        "device": str(_device) if _device else "unknown",
        "model_ready": _model_ready,
        "load_progress": _load_progress,
        "cwd": os.getcwd(),
    }

    # Check body model files
    body_dir = os.path.join(SMPLESTX_ROOT, "human_models/human_model_files")
    for subdir in ["smplx", "smpl", "mano"]:
        d = os.path.join(body_dir, subdir)
        if os.path.isdir(d):
            files = os.listdir(d)
            info[f"body_{subdir}"] = files
        else:
            info[f"body_{subdir}"] = "MISSING"

    # Check checkpoint
    ckpt = os.path.join(SMPLESTX_ROOT,
                        "pretrained_models/smplest_x_h/smplest_x_h.pth.tar")
    info["checkpoint_exists"] = os.path.exists(ckpt)
    if os.path.exists(ckpt):
        info["checkpoint_size_gb"] = round(os.path.getsize(ckpt) / (1024 ** 3), 2)

    # Check config
    cfg_path = os.path.join(SMPLESTX_ROOT,
                            "pretrained_models/smplest_x_h/config_base.py")
    info["config_exists"] = os.path.exists(cfg_path)

    # YOLO
    yolo_path = os.path.join(SMPLESTX_ROOT, "pretrained_models/yolov8x.pt")
    info["yolo_exists"] = os.path.exists(yolo_path)

    # Test imports
    test_mods = ["scipy", "smplx", "cv2", "torch", "mmcv", "mmpose",
                 "mmdet", "mmengine", "ultralytics"]
    for mod in test_mods:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "ok")
            info[f"import_{mod}"] = ver
        except Exception as e:
            info[f"import_{mod}"] = f"FAIL: {str(e)[:80]}"

    if np is not None:
        info["numpy_version"] = np.__version__
    if torch is not None:
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_mem / (1024 ** 3), 1)

    if _load_error:
        info["load_error"] = _load_error[-2000:]

    # Try to reload model if it failed (for debugging)
    if not _model_ready and _load_error and "reload" not in info:
        info["hint"] = "Model failed to load. Check load_error for details."

    return info


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
