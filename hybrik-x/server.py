"""
HybrIK-X inference server.

Receives a base64-encoded image, returns SMPL-X body parameters:
- 55 joint rotations (quaternions + rotation matrices)
- shape betas, expression params
- 3D joint positions (meters, root-relative)
- global translation

Usage:
    POST /predict  { "image_base64": "...", "bbox": [x1,y1,x2,y2] (optional) }
    GET  /health
"""
import base64
import os
import sys

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

sys.path.insert(0, "/app/HybrIK")

from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPLX
from hybrik.utils.vis import get_one_box
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

app = FastAPI(title="HybrIK-X", description="Whole-body SMPL-X mesh recovery")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CFG_FILE = "/app/HybrIK/configs/smplx/256x192_hrnet_rle_smplx_kid.yaml"
CKPT = "/app/HybrIK/pretrained_models/hybrikx_rle_hrnet.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
det_transform = T.Compose([T.ToTensor()])

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

transformation = SimpleTransform3DSMPLX(
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

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
det_model = fasterrcnn_resnet50_fpn(pretrained=True).to(device).eval()

hybrik_model = builder.build_sppe(cfg.MODEL)
save_dict = torch.load(CKPT, map_location="cpu")
if isinstance(save_dict, dict) and "model" in save_dict:
    hybrik_model.load_state_dict(save_dict["model"])
else:
    hybrik_model.load_state_dict(save_dict)
hybrik_model.to(device).eval()

print("HybrIK-X models loaded.", flush=True)

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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Decode image
    try:
        img_bytes = base64.b64decode(req.image_base64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    with torch.no_grad():
        # Detect person
        if req.bbox is not None:
            tight_bbox = req.bbox
        else:
            det_input = det_transform(img).to(device)
            det_output = det_model([det_input])[0]
            tight_bbox = get_one_box(det_output)
            if tight_bbox is None:
                raise HTTPException(status_code=422, detail="No person detected")

        # Preprocess
        pose_input, bbox, img_center = transformation.test_transform(img.copy(), tight_bbox)
        pose_input = pose_input.to(device).unsqueeze(0)

        # Inference
        out = hybrik_model(
            pose_input,
            flip_test=req.flip_test,
            bboxes=torch.tensor(bbox, device=device).unsqueeze(0).float(),
            img_center=torch.from_numpy(img_center).to(device).unsqueeze(0).float(),
        )

    b = 0
    resp = PredictResponse(
        theta_quat=out.pred_theta_quat[b].cpu().numpy().reshape(55, 4).tolist(),
        rot_mats=out.pred_theta_mat[b].cpu().numpy().reshape(55, 3, 3).tolist(),
        betas=out.pred_beta[b].cpu().numpy().tolist(),
        expression=out.pred_expression[b].cpu().numpy().tolist(),
        transl=out.transl[b].cpu().numpy().tolist(),
        cam_scale=out.cam_scale[b].cpu().numpy().tolist(),
        joints_3d=out.pred_xyz_hybrik[b].cpu().numpy().reshape(-1, 3).tolist(),
        bbox=list(bbox),
    )

    if req.include_vertices:
        resp.vertices = out.pred_vertices[b].cpu().numpy().tolist()

    return resp


if __name__ == "__main__":
    os.makedirs("/var/log/portal", exist_ok=True)
    print("Application startup complete.", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
