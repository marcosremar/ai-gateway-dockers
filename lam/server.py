"""
LAM — Large Avatar Model  (SIGGRAPH 2025)
FastAPI server wrapping the LAM inference pipeline.

Endpoints:
  GET  /health                   — service status + model readiness
  GET  /version                  — model info
  POST /v1/avatar/generate       — photo → animated MP4 (multipart upload)
  POST /v1/avatar/generate-b64   — base64 photo → base64 MP4 (JSON)
  POST /v1/avatar/generate-glb   — photo → rigged GLB (FLAME mesh + skeleton + morph targets)

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
import json
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


def _build_rigged_glb(runtime_dir: str, max_expr: int = 50) -> bytes:
    """
    Assemble a rigged glTF-2.0 GLB from LAM's runtime_data directory.

    Reads:
      nature.obj             — shaped FLAME mesh (~20 K verts)
      bone_tree.json         — skeleton hierarchy (5 joints)
      lbs_weight_20k.json    — per-vertex LBS weights  [V, 5]
      bs/expr*.obj           — expression morph targets (up to max_expr)

    Returns raw GLB bytes (rigged + morph-target capable, no animations).
    """
    import numpy as np
    import trimesh
    import pygltflib

    rd = Path(runtime_dir)

    # ── 1. Mesh ────────────────────────────────────────────────────────────────
    mesh_raw = trimesh.load(str(rd / "nature.obj"), process=False)
    verts   = np.array(mesh_raw.vertices, dtype=np.float32)   # [V, 3]
    faces   = np.array(mesh_raw.faces,    dtype=np.uint32)    # [F, 3]
    mesh_sm = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    normals = np.array(mesh_sm.vertex_normals, dtype=np.float32)  # [V, 3]
    V = len(verts)

    # ── 2. Skeleton ────────────────────────────────────────────────────────────
    bt = json.loads((rd / "bone_tree.json").read_text())
    joints: list[dict] = []

    def _parse(node: dict, parent: int = -1) -> None:
        i = len(joints)
        joints.append({
            "name":   node["name"],
            "wpos":   np.array(node["position"], np.float32),
            "parent": parent,
        })
        for c in node.get("children", []):
            _parse(c, i)

    _parse(bt["bones"][0])
    NJ = len(joints)  # 5 for FLAME

    # ── 3. LBS weights → top-4 joints per vertex ──────────────────────────────
    lbs_full = np.array(
        json.loads((rd / "lbs_weight_20k.json").read_text()), dtype=np.float32
    )  # [V, NJ]
    top4_i = np.argsort(-lbs_full, axis=1)[:, :4].astype(np.uint8)   # [V, 4]
    top4_w = np.take_along_axis(lbs_full, top4_i.astype(np.int32), axis=1).astype(np.float32)
    row_s  = top4_w.sum(axis=1, keepdims=True); row_s[row_s == 0] = 1.0
    top4_w /= row_s

    # ── 4. Expression morph targets ────────────────────────────────────────────
    bs_dir   = rd / "bs"
    expr_files = sorted(bs_dir.glob("expr*.obj"))[:max_expr]
    morph_deltas: list[np.ndarray] = []
    morph_names: list[str] = []
    for ef in expr_files:
        em = trimesh.load(str(ef), process=False)
        delta = np.array(em.vertices, dtype=np.float32) - verts
        if np.abs(delta).max() > 1e-6:          # skip zero-deformation shapes
            morph_deltas.append(delta)
            morph_names.append(ef.stem)

    # ── 5. Inverse bind matrices (column-major 4×4) ────────────────────────────
    ibm_list = []
    for j in joints:
        m = np.eye(4, dtype=np.float32)
        m[0, 3] = -j["wpos"][0]
        m[1, 3] = -j["wpos"][1]
        m[2, 3] = -j["wpos"][2]
        ibm_list.append(m.T.flatten())           # glTF is column-major
    ibm_arr = np.stack(ibm_list)                 # [NJ, 16]

    # ── 6. Pack binary buffer ──────────────────────────────────────────────────
    ARRAY_BUFFER         = pygltflib.ARRAY_BUFFER
    ELEMENT_ARRAY_BUFFER = pygltflib.ELEMENT_ARRAY_BUFFER

    bin_chunks: list[bytes] = []
    bv_list:    list[pygltflib.BufferView] = []
    ac_list:    list[pygltflib.Accessor]   = []

    def _add(data: np.ndarray, target=None) -> int:
        """Append data to buffer, register BufferView + Accessor; return accessor index."""
        raw = data.tobytes()
        # pad to 4-byte boundary
        pad = (4 - len(raw) % 4) % 4
        bin_chunks.append(raw + b"\x00" * pad)

        bv_idx = len(bv_list)
        bv_list.append(pygltflib.BufferView(
            buffer=0,
            byteOffset=sum(len(c) for c in bin_chunks[:-1]),
            byteLength=len(raw),
            target=target,
        ))

        dtype_map = {
            np.float32:  pygltflib.FLOAT,
            np.uint32:   pygltflib.UNSIGNED_INT,
            np.uint16:   pygltflib.UNSIGNED_SHORT,
            np.uint8:    pygltflib.UNSIGNED_BYTE,
        }
        comp_type = dtype_map[data.dtype.type]

        shape1 = data.shape[1] if data.ndim > 1 else 1
        type_map = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4", 16: "MAT4"}
        acc_type = type_map[shape1]

        flat = data.reshape(len(data), -1) if data.ndim > 1 else data.reshape(-1, 1)
        # glTF spec: min/max not defined for MAT* types
        if acc_type.startswith("MAT"):
            min_val, max_val = None, None
        else:
            min_val = flat.min(axis=0).tolist()
            max_val = flat.max(axis=0).tolist()

        acc_idx = len(ac_list)
        ac_list.append(pygltflib.Accessor(
            bufferView=bv_idx,
            byteOffset=0,
            componentType=comp_type,
            type=acc_type,
            count=len(data),
            min=min_val,
            max=max_val,
        ))
        return acc_idx

    pos_acc     = _add(verts,              ARRAY_BUFFER)
    norm_acc    = _add(normals,            ARRAY_BUFFER)
    idx_acc     = _add(faces.flatten(),    ELEMENT_ARRAY_BUFFER)
    joints_acc  = _add(top4_i,             ARRAY_BUFFER)
    weights_acc = _add(top4_w,             ARRAY_BUFFER)
    ibm_acc     = _add(ibm_arr)

    morph_prim_targets = []
    for delta in morph_deltas:
        da = _add(delta, ARRAY_BUFFER)
        morph_prim_targets.append({"POSITION": da})

    # ── 7. Scene graph ─────────────────────────────────────────────────────────
    # Node layout:
    #   0           — mesh node
    #   1 … NJ      — joint nodes (indexed from 1)
    #   NJ+1        — scene root (contains nodes 0 and 1)

    def _joint_node_idx(ji: int) -> int:
        return ji + 1

    joint_nodes = []
    for i, j in enumerate(joints):
        children_ji = [k for k, jj in enumerate(joints) if jj["parent"] == i]
        if j["parent"] == -1:
            lpos = j["wpos"].tolist()
        else:
            lpos = (j["wpos"] - joints[j["parent"]]["wpos"]).tolist()

        joint_nodes.append(pygltflib.Node(
            name=j["name"],
            translation=lpos,
            children=[_joint_node_idx(c) for c in children_ji] or None,
        ))

    prim = pygltflib.Primitive(
        attributes=pygltflib.Attributes(
            POSITION=pos_acc,
            NORMAL=norm_acc,
            JOINTS_0=joints_acc,
            WEIGHTS_0=weights_acc,
        ),
        indices=idx_acc,
        targets=morph_prim_targets or None,
    )

    mesh_node = pygltflib.Node(
        name="head_mesh",
        mesh=0,
        skin=0,
    )

    scene_root = pygltflib.Node(
        name="Avatar",
        children=[0, _joint_node_idx(0)],   # mesh node + root joint node
    )

    nodes = [mesh_node] + joint_nodes + [scene_root]
    root_node_idx = len(nodes) - 1

    skin = pygltflib.Skin(
        name="FLAME",
        joints=[_joint_node_idx(i) for i in range(NJ)],
        inverseBindMatrices=ibm_acc,
        skeleton=_joint_node_idx(0),
    )

    gltf_mesh = pygltflib.Mesh(
        name="head",
        primitives=[prim],
        extras={"targetNames": morph_names} if morph_names else None,
    )

    bin_blob = b"".join(bin_chunks)

    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(name="Scene", nodes=[root_node_idx])],
        nodes=nodes,
        meshes=[gltf_mesh],
        skins=[skin],
        accessors=ac_list,
        bufferViews=bv_list,
        buffers=[pygltflib.Buffer(byteLength=len(bin_blob))],
    )
    gltf.set_binary_blob(bin_blob)
    return b"".join(gltf.save_to_bytes())


def run_glb_generation(image_path: str) -> bytes:
    """FLAME track → save_h5_info → build rigged GLB."""
    from lam.runners.infer.head_utils import preprocess_image

    tmp = tempfile.mkdtemp()
    runtime_dir = str(LAM_DIR / "runtime_data")
    try:
        # ── 1. FLAME tracking ─────────────────────────────────────────────────
        log.info("[lam/glb] FLAME tracking: %s", image_path)
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

        # ── 2. Shape parameters ───────────────────────────────────────────────
        _, _, _, shape_param = preprocess_image(
            tracked_img, mask_path=tracked_mask,
            intr=None, pad_ratio=0, bg_color=1.,
            max_tgt_size=None, aspect_standard=1.0, enlarge_ratio=[1.0, 1.0],
            render_tgt_size=source_size, multiply=14,
            need_mask=True, get_shape_param=True,
        )

        # ── 3. Generate runtime_data/ (nature.obj, skeleton, weights, bs/) ────
        log.info("[lam/glb] generating h5 info → %s", runtime_dir)
        os.makedirs(runtime_dir, exist_ok=True)
        lam_model.renderer.flame_model.save_h5_info(
            shape_param.unsqueeze(0).to(DEVICE), fd=runtime_dir
        )

        # ── 4. Build rigged GLB in pure Python ────────────────────────────────
        log.info("[lam/glb] assembling rigged GLB...")
        t0 = time.time()
        glb_bytes = _build_rigged_glb(runtime_dir)
        log.info("[lam/glb] GLB ready: %.1f KB in %.2fs",
                 len(glb_bytes) / 1024, time.time() - t0)
        return glb_bytes

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


_glb_lock = asyncio.Lock()


@app.post("/v1/avatar/generate-glb")
async def generate_glb(
    file: UploadFile = File(...),
):
    """Generate a rigged GLB avatar from a single face photo.

    Returns a glTF-2.0 binary (.glb) with:
    - FLAME head mesh (~20 K vertices)
    - 5-joint skeleton (root → neck → jaw / leftEye / rightEye)
    - Linear-blend skinning weights
    - Up to 50 expression morph targets (blend shapes)
    """
    if service_status.get("model") != "ready":
        raise HTTPException(503, "Model not ready yet")

    tmp = tempfile.mkdtemp()
    ext = Path(file.filename).suffix or ".jpg"
    img_path = os.path.join(tmp, f"input{ext}")
    try:
        with open(img_path, "wb") as f:
            f.write(await file.read())

        async with _glb_lock:
            loop = asyncio.get_event_loop()
            glb_bytes = await loop.run_in_executor(
                None, run_glb_generation, img_path
            )
    except Exception as e:
        log.error("[lam/glb] error: %s", e, exc_info=True)
        raise HTTPException(500, str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return Response(
        content=glb_bytes,
        media_type="model/gltf-binary",
        headers={"Content-Disposition": "attachment; filename=avatar.glb"},
    )


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, workers=1)
