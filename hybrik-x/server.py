"""
HybrIK-X inference server.

HTTP:
    POST /predict  { "image_base64": "...", "bbox": [x1,y1,x2,y2] (optional) }
    GET  /health   → {"status": "loading"} while models load, {"status": "ok"} when ready

WebSocket:
    WS /ws/predict  — stream frames, get pose results in real time

    Client → Server (each message, JSON):
        { "image_base64": "...", "bbox": [x1,y1,x2,y2], "include_vertices": false, "flip_test": true }

    Server → Client:
        { "status": "ok", "theta_quat": [...], "betas": [...], ... }   # same as /predict response
        { "status": "error", "detail": "..." }                         # inference error
        { "status": "loading", "detail": "..." }                       # models not ready yet
"""
import asyncio
import base64
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, "/app/HybrIK")

# ---------------------------------------------------------------------------
# Config / paths
# ---------------------------------------------------------------------------
CFG_FILE = "/app/HybrIK/configs/smplx/256x192_hrnet_rle_smplx_kid.yaml"
CKPT = "/app/HybrIK/pretrained_models/hybrikx_rle_hrnet.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model state — populated during lifespan startup so /health responds
# immediately with "loading" while models are warming up.
_models_ready = False
_det_model = None
_hybrik_model = None
_transformation = None
_det_transform = None
_get_one_box = None  # lazily imported from hybrik


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _models_ready, _det_model, _hybrik_model, _transformation, _det_transform, _get_one_box

    print(f"[hybrik-x] Loading models on {device}...", flush=True)

    from hybrik.models import builder
    from hybrik.utils.config import update_config
    from hybrik.utils.presets import SimpleTransform3DSMPLX
    from hybrik.utils.vis import get_one_box
    from torchvision import transforms as T
    from torchvision.models.detection import fasterrcnn_resnet50_fpn

    cfg = update_config(CFG_FILE)
    cfg["MODEL"]["EXTRA"]["USE_KID"] = cfg["DATASET"].get("USE_KID", False)
    cfg["LOSS"]["ELEMENTS"]["USE_KID"] = cfg["DATASET"].get("USE_KID", False)

    bbox_3d_shape = getattr(cfg.MODEL, "BBOX_3D_SHAPE", (2000, 2000, 2000))
    bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]

    dummy_set = edict(
        joint_pairs_17=None,
        joint_pairs_24=None,
        joint_pairs_29=None,
        bbox_3d_shape=bbox_3d_shape,
    )

    _transformation = SimpleTransform3DSMPLX(
        dummy_set,
        scale_factor=cfg.DATASET.SCALE_FACTOR,
        color_factor=cfg.DATASET.COLOR_FACTOR,
        occlusion=cfg.DATASET.OCCLUSION,
        input_size=cfg.MODEL.IMAGE_SIZE,
        output_size=cfg.MODEL.HEATMAP_SIZE,
        depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
        bbox_3d_shape=bbox_3d_shape,
        rot=cfg.DATASET.ROT_FACTOR,
        sigma=cfg.MODEL.EXTRA.SIGMA,
        train=False,
        add_dpg=False,
        loss_type=cfg.LOSS["TYPE"],
    )

    _det_transform = T.Compose([T.ToTensor()])
    _det_model = fasterrcnn_resnet50_fpn(pretrained=True).to(device).eval()

    _hybrik_model = builder.build_sppe(cfg.MODEL)
    save_dict = torch.load(CKPT, map_location="cpu")
    if isinstance(save_dict, dict) and "model" in save_dict:
        _hybrik_model.load_state_dict(save_dict["model"])
    else:
        _hybrik_model.load_state_dict(save_dict)
    _hybrik_model.to(device).eval()

    _get_one_box = get_one_box
    _models_ready = True
    print("Application startup complete.", flush=True)

    yield  # server runs here

    print("[hybrik-x] Shutting down.", flush=True)


app = FastAPI(title="HybrIK-X", description="Whole-body SMPL-X mesh recovery", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    image_base64: str
    bbox: Optional[list] = None  # [x1, y1, x2, y2] — skip detection
    include_vertices: bool = False
    flip_test: bool = True


class PredictResponse(BaseModel):
    # 55 joint rotations as quaternions [w,x,y,z] — shape [55][4]
    theta_quat: list
    # 55 joint rotation matrices — shape [55][3][3]
    rot_mats: list
    # shape betas (10) + kid (1) — [11]
    betas: list
    # expression params — [10]
    expression: list
    # global translation in meters — [3]
    transl: list
    # camera scale — [1]
    cam_scale: list
    # 3D joints root-relative in meters — [71][3]
    joints_3d: list
    # bounding box used — [x1,y1,x2,y2]
    bbox: list
    # full SMPL-X mesh vertices (only if include_vertices=True) — [10475][3]
    vertices: Optional[list] = None
    # inference latency in ms
    latency_ms: Optional[float] = None


# ---------------------------------------------------------------------------
# Core inference function (shared by HTTP and WebSocket)
# ---------------------------------------------------------------------------

def _run_inference(image_base64: str, bbox_hint: Optional[list], flip_test: bool, include_vertices: bool) -> dict:
    """Decode image, run detection + HybrIK-X, return result dict."""
    try:
        img_bytes = base64.b64decode(image_base64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2 could not decode the image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise ValueError(f"Invalid image: {e}")

    t0 = time.perf_counter()
    with torch.no_grad():
        if bbox_hint is not None:
            tight_bbox = bbox_hint
        else:
            det_input = _det_transform(img).to(device)
            det_output = _det_model([det_input])[0]
            tight_bbox = _get_one_box(det_output)
            if tight_bbox is None:
                raise ValueError("No person detected in image")

        pose_input, bbox, img_center = _transformation.test_transform(img.copy(), tight_bbox)
        pose_input = pose_input.to(device).unsqueeze(0)

        out = _hybrik_model(
            pose_input,
            flip_test=flip_test,
            bboxes=torch.tensor(bbox, device=device).unsqueeze(0).float(),
            img_center=torch.from_numpy(img_center).to(device).unsqueeze(0).float(),
        )

    latency_ms = (time.perf_counter() - t0) * 1000
    b = 0
    result = {
        "theta_quat": out.pred_theta_quat[b].cpu().numpy().reshape(55, 4).tolist(),
        "rot_mats": out.pred_theta_mat[b].cpu().numpy().reshape(55, 3, 3).tolist(),
        "betas": out.pred_beta[b].cpu().numpy().tolist(),
        "expression": out.pred_expression[b].cpu().numpy().tolist(),
        "transl": out.transl[b].cpu().numpy().tolist(),
        "cam_scale": out.cam_scale[b].cpu().numpy().tolist(),
        "joints_3d": out.pred_xyz_hybrik[b].cpu().numpy().reshape(-1, 3).tolist(),
        "bbox": list(bbox),
        "latency_ms": round(latency_ms, 1),
    }
    if include_vertices:
        result["vertices"] = out.pred_vertices[b].cpu().numpy().tolist()
    return result


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    if not _models_ready:
        return {"status": "loading", "device": str(device)}
    return {"status": "ok", "device": str(device)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not _models_ready:
        raise HTTPException(status_code=503, detail="Models still loading")
    try:
        result = _run_inference(req.image_base64, req.bbox, req.flip_test, req.include_vertices)
        return result
    except ValueError as e:
        status = 422 if "No person" in str(e) else 400
        raise HTTPException(status_code=status, detail=str(e))


# ---------------------------------------------------------------------------
# WebSocket endpoint — stream frames, get pose in real time
# ---------------------------------------------------------------------------

@app.websocket("/ws/predict")
async def ws_predict(websocket: WebSocket):
    """
    Real-time pose estimation over WebSocket.

    Send JSON frames:
        { "image_base64": "...", "bbox": [...], "flip_test": true, "include_vertices": false }

    Receive JSON results:
        { "status": "ok", "theta_quat": [...], "latency_ms": 42.1, ... }
        { "status": "error", "detail": "No person detected" }
        { "status": "loading" }  — if models not ready, retry in 1s
    """
    await websocket.accept()
    loop = asyncio.get_event_loop()

    try:
        while True:
            raw = await websocket.receive_text()

            if not _models_ready:
                await websocket.send_json({"status": "loading"})
                continue

            try:
                import json
                msg = json.loads(raw)
            except Exception:
                await websocket.send_json({"status": "error", "detail": "Invalid JSON"})
                continue

            image_b64 = msg.get("image_base64", "")
            if not image_b64:
                await websocket.send_json({"status": "error", "detail": "image_base64 required"})
                continue

            bbox_hint = msg.get("bbox")
            flip_test = bool(msg.get("flip_test", True))
            include_vertices = bool(msg.get("include_vertices", False))

            # Run inference in executor so we don't block the event loop
            try:
                result = await loop.run_in_executor(
                    None,
                    _run_inference,
                    image_b64,
                    bbox_hint,
                    flip_test,
                    include_vertices,
                )
                await websocket.send_json({"status": "ok", **result})
            except ValueError as e:
                await websocket.send_json({"status": "error", "detail": str(e)})
            except Exception as e:
                await websocket.send_json({"status": "error", "detail": f"Inference failed: {e}"})

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    os.makedirs("/var/log/portal", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
