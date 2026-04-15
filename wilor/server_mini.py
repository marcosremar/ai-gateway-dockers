"""
WiLoR-mini FastAPI server — uses the pip-installable wilor_mini package.

Usage:
  /tmp/wilor-env-312/bin/python services/wilor/server_mini.py

Returns MANO joint rotations (15 axis-angle vectors) per detected hand.
"""
import base64
import io
import time

import numpy as np
import torch

# Patch torch.load for compatibility with ultralytics 8.1 + torch 2.6+
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import uvicorn
import json as json_mod
import asyncio

# ── WiLoR-mini pipeline ──
_pipeline = None
_device = None


def _rotmat_to_axis_angle(R):
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if angle < 1e-6:
        return np.zeros(3)
    axis = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    axis = axis / (2 * np.sin(angle) + 1e-8)
    return axis * angle


def _rotmats_to_axis_angle(rotmats):
    return np.array([_rotmat_to_axis_angle(R) for R in rotmats])


MANO_ROTATION_NAMES = [
    "index_mcp", "index_pip", "index_dip",
    "middle_mcp", "middle_pip", "middle_dip",
    "pinky_mcp", "pinky_pip", "pinky_dip",
    "ring_mcp", "ring_pip", "ring_dip",
    "thumb_cmc", "thumb_mcp", "thumb_ip",
]

JOINT_NAMES = [
    "wrist",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
]


app = FastAPI(title="WiLoR-mini Hand Pose API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

try:
    from idle_watchdog import add_idle_middleware
    add_idle_middleware(app)
except Exception:
    pass


@app.on_event("startup")
async def startup():
    global _pipeline, _device
    if torch.cuda.is_available():
        _device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        _device = torch.device("mps")
    else:
        _device = torch.device("cpu")

    # FP16 on CUDA halves inference time (~12ms vs 25ms on 4090)
    _dtype = torch.float16 if _device.type == "cuda" else torch.float32
    print(f"[WiLoR-mini] Loading on {_device} dtype={_dtype}...")
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    _pipeline = WiLorHandPose3dEstimationPipeline(device=_device, dtype=_dtype)
    print("[WiLoR-mini] Ready.")
    try:
        from idle_watchdog import start_watchdog
        import asyncio
        asyncio.create_task(start_watchdog())
    except Exception:
        pass


class PredictRequest(BaseModel):
    image_base64: str


@app.get("/health")
async def health():
    dtype = str(getattr(_pipeline, 'dtype', 'float32')) if _pipeline else 'unknown'
    resp = {"status": "ok" if _pipeline else "loading", "device": str(_device), "dtype": dtype}
    try:
        from idle_watchdog import get_health_metrics
        resp.update(get_health_metrics())
    except Exception:
        pass
    return resp


@app.post("/predict")
async def predict(req: PredictRequest):
    if not _pipeline:
        return JSONResponse({"error": "Model not loaded"}, status_code=503)

    t0 = time.perf_counter()

    try:
        img_bytes = base64.b64decode(req.image_base64)
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img_pil)
    except Exception as e:
        return JSONResponse({"error": f"Image decode: {e}"}, status_code=400)

    try:
        with torch.no_grad():
            result = _pipeline.predict(img_np)
            if isinstance(result, tuple):
                outputs = result[0]
            else:
                outputs = result
    except Exception as e:
        return JSONResponse({"error": f"Inference: {e}"}, status_code=500)

    hands = []
    # Debug: print all keys available in the output
    if outputs:
        sample = outputs[0] if isinstance(outputs, list) else outputs
        if isinstance(sample, dict):
            print(f"[WiLoR-debug] Output keys: {list(sample.keys())}")
            for k, v in sample.items():
                if hasattr(v, 'shape'):
                    print(f"  {k}: tensor {v.shape} {v.dtype}")
                elif isinstance(v, (int, float, bool, str)):
                    print(f"  {k}: {v}")
                elif isinstance(v, dict):
                    print(f"  {k}: dict keys={list(v.keys())}")
                    for kk, vv in v.items():
                        if hasattr(vv, 'shape'):
                            print(f"    {kk}: tensor {vv.shape}")
                else:
                    print(f"  {k}: {type(v).__name__}")
    for out in outputs:
        is_right = bool(out.get("is_right", 1))
        hand = {"side": "right" if is_right else "left"}

        # 3D joints — wilor_mini stores in wilor_preds
        wp = out.get("wilor_preds", {})
        joints = wp.get("pred_keypoints_3d") if isinstance(wp, dict) else None
        if joints is None:
            joints = out.get("pred_joints_img") or out.get("pred_keypoints_3d")
        if joints is not None:
            j = joints.detach().cpu().numpy() if hasattr(joints, 'detach') else np.array(joints)
            if j.ndim > 2:
                j = j[0]
            if j.shape[0] == 21:
                hand["joints"] = {name: j[i].tolist() for i, name in enumerate(JOINT_NAMES)}

        # MANO rotations — wilor_mini stores them under 'wilor_preds'
        mano = out.get("wilor_preds", out.get("pred_mano_params", {}))
        if isinstance(mano, dict):
            hp = mano.get("hand_pose")
            if hp is not None:
                hp_np = hp.detach().cpu().numpy() if hasattr(hp, 'detach') else np.array(hp)
                hp_np = hp_np.reshape(-1, 3)  # flatten batch dim
                if hp_np.shape[0] == 15:
                    # Already axis-angle (15, 3) from wilor_mini
                    hand["rotations"] = hp_np.tolist()
                    hand["named_rotations"] = {
                        name: hp_np[i].tolist() for i, name in enumerate(MANO_ROTATION_NAMES)
                    }
                elif hp_np.shape[0] == 45:
                    # Flat axis-angle (45,) → reshape to (15, 3)
                    hp_15 = hp_np.reshape(15, 3)
                    hand["rotations"] = hp_15.tolist()
                    hand["named_rotations"] = {
                        name: hp_15[i].tolist() for i, name in enumerate(MANO_ROTATION_NAMES)
                    }

            go = mano.get("global_orient")
            if go is not None:
                go_np = go.detach().cpu().numpy() if hasattr(go, 'detach') else np.array(go)
                go_flat = go_np.flatten()
                if go_flat.shape[0] == 3:
                    hand["wrist_rotation"] = go_flat.tolist()
                elif go_flat.shape[0] == 9:
                    hand["wrist_rotation"] = _rotmat_to_axis_angle(go_flat.reshape(3, 3)).tolist()

            bt = mano.get("betas")
            if bt is not None:
                bt_np = bt.detach().cpu().numpy() if hasattr(bt, 'detach') else np.array(bt)
                hand["betas"] = bt_np.flatten().tolist()

        hands.append(hand)

    ms = round((time.perf_counter() - t0) * 1000, 1)
    return {"hands": hands, "ms": ms}


# ── Fast crop endpoint: skip YOLO, run ViT directly on pre-cropped hand ──
@app.post("/predict_crop")
async def predict_crop(req: PredictRequest):
    """Accept a pre-cropped 256x256 hand image, skip YOLO detection.
    The browser crops the hand using MediaPipe's bbox, so we only need
    the ViT + MANO head (~20ms on 4090 vs ~60ms with YOLO)."""
    if not _pipeline:
        return JSONResponse({"error": "Model not loaded"}, status_code=503)

    t0 = time.perf_counter()
    try:
        img_bytes = base64.b64decode(req.image_base64)
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        # Resize to 256x256 (WiLoR's expected input)
        img_pil = img_pil.resize((256, 256))
        img_np = np.array(img_pil)
    except Exception as e:
        return JSONResponse({"error": f"Image decode: {e}"}, status_code=400)

    try:
        with torch.no_grad():
            # Run full pipeline on pre-cropped image (YOLO on 256x256 is fast)
            result = _pipeline.predict(img_np)
            outputs = result[0] if isinstance(result, tuple) else result
    except Exception as e:
        return JSONResponse({"error": f"Inference: {e}"}, status_code=500)

    hands = []
    for out in outputs:
        hand = {"side": "right"}  # caller specifies; default right
        mano = out.get("wilor_preds", {})
        if isinstance(mano, dict):
            hp = mano.get("hand_pose")
            if hp is not None:
                hp_np = hp.detach().cpu().numpy() if hasattr(hp, 'detach') else np.array(hp)
                hp_np = hp_np.reshape(-1, 3)
                if hp_np.shape[0] == 15:
                    hand["rotations"] = hp_np.tolist()
                    hand["named_rotations"] = {
                        name: hp_np[i].tolist() for i, name in enumerate(MANO_ROTATION_NAMES)
                    }
            go = mano.get("global_orient")
            if go is not None:
                go_np = go.detach().cpu().numpy() if hasattr(go, 'detach') else np.array(go)
                go_flat = go_np.flatten()
                if go_flat.shape[0] == 3:
                    hand["wrist_rotation"] = go_flat.tolist()
        hands.append(hand)

    if not hands:
        hands = [{"side": "right"}]

    ms = round((time.perf_counter() - t0) * 1000, 1)
    return {"hands": hands, "ms": ms}


# ── WebSocket endpoint (persistent connection, no TCP handshake per frame) ──
@app.websocket("/ws")
async def websocket_predict(ws: WebSocket):
    await ws.accept()
    print("[WiLoR-ws] Client connected")
    try:
        while True:
            data = await ws.receive_text()
            msg = json_mod.loads(data)
            b64 = msg.get("image_base64", "")
            if not b64 or not _pipeline:
                await ws.send_json({"hands": [], "ms": 0})
                continue

            t0 = time.perf_counter()
            img_bytes = base64.b64decode(b64)
            img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_np = np.array(img_pil)

            # "crop" mode: pre-cropped hand → skip YOLO, run ViT directly
            mode = msg.get("mode", "full")  # "full" or "crop"
            is_right_hint = msg.get("is_right", 1)

            if mode == "crop":
                # Pre-cropped hand image — run full pipeline on small image
                # (YOLO on 256x256 is fast; avoids wilor_model preprocessing issues)
                img_cropped = np.array(img_pil.resize((256, 256)))
                with torch.no_grad():
                    result = _pipeline.predict(img_cropped)
                    outputs_crop = result[0] if isinstance(result, tuple) else result

                hands = []
                if outputs_crop:
                    for out in outputs_crop:
                        hand = {"side": "right" if is_right_hint else "left"}
                        mano = out.get("wilor_preds", {})
                        if isinstance(mano, dict):
                            hp = mano.get("hand_pose")
                            if hp is not None:
                                hp_np = hp.detach().cpu().numpy() if hasattr(hp, 'detach') else np.array(hp)
                                hp_np = hp_np.reshape(-1, 3)
                                if hp_np.shape[0] == 15:
                                    hand["rotations"] = hp_np.tolist()
                                    hand["named_rotations"] = {
                                        n: hp_np[i].tolist() for i, n in enumerate(MANO_ROTATION_NAMES)}
                            go = mano.get("global_orient")
                            if go is not None:
                                go_np = go.detach().cpu().numpy() if hasattr(go, 'detach') else np.array(go)
                                go_flat = go_np.flatten()
                                if go_flat.shape[0] == 3:
                                    hand["wrist_rotation"] = go_flat.tolist()
                        hands.append(hand)
                if not hands:
                    hands = [{"side": "right" if is_right_hint else "left"}]
            else:
                # Full pipeline: YOLO detection + ViT + MANO
                with torch.no_grad():
                    result = _pipeline.predict(img_np)
                    outputs = result[0] if isinstance(result, tuple) else result

                hands = []
                for out in outputs:
                    is_right = bool(out.get("is_right", 1))
                    hand = {"side": "right" if is_right else "left"}
                    mano = out.get("wilor_preds", {})
                    if isinstance(mano, dict):
                        hp = mano.get("hand_pose")
                        if hp is not None:
                            hp_np = hp.detach().cpu().numpy() if hasattr(hp, 'detach') else np.array(hp)
                            hp_np = hp_np.reshape(-1, 3)
                            if hp_np.shape[0] == 15:
                                hand["rotations"] = hp_np.tolist()
                                hand["named_rotations"] = {
                                    n: hp_np[i].tolist() for i, n in enumerate(MANO_ROTATION_NAMES)}
                        go = mano.get("global_orient")
                        if go is not None:
                            go_np = go.detach().cpu().numpy() if hasattr(go, 'detach') else np.array(go)
                            go_flat = go_np.flatten()
                            if go_flat.shape[0] == 3:
                                hand["wrist_rotation"] = go_flat.tolist()
                hands.append(hand)

            ms = round((time.perf_counter() - t0) * 1000, 1)
            await ws.send_json({"hands": hands, "ms": ms})
    except WebSocketDisconnect:
        print("[WiLoR-ws] Client disconnected")
    except Exception as e:
        print(f"[WiLoR-ws] Error: {e}")


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
