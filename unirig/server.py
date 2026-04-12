"""
UniRig — FastAPI server for automatic 3D model rigging.

Endpoints:
    POST /rig          — Upload a GLB/OBJ/FBX, get back a rigged GLB
    GET  /health       — Service status
    GET  /diag         — Full diagnostic dump

Input:  GLB/OBJ/FBX file (multipart upload)
Output: Rigged GLB file with skeleton + skin weights

Pipeline: extract mesh → generate skeleton → predict skin weights → merge
"""

import os
import sys
import time
import shutil
import logging
import tempfile
import traceback
import subprocess
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("unirig")

UNIRIG_DIR = Path("/app/unirig")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

app = FastAPI(title="UniRig Auto-Rigging Server", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Model readiness
model_ready = False
model_error = None


@app.on_event("startup")
async def startup():
    global model_ready, model_error
    try:
        # Verify UniRig is importable
        sys.path.insert(0, str(UNIRIG_DIR))
        log.info("Checking UniRig installation...")

        # Verify checkpoints exist
        ckpt_dir = UNIRIG_DIR / "checkpoints"
        if not ckpt_dir.exists():
            log.info("Downloading UniRig checkpoints...")
            from huggingface_hub import snapshot_download
            snapshot_download("VAST-AI/UniRig", local_dir=str(ckpt_dir))

        log.info(f"UniRig ready on {torch.cuda.get_device_name(0)}")
        model_ready = True
    except Exception as e:
        model_error = f"{type(e).__name__}: {e}"
        log.error(f"Startup failed: {model_error}\n{traceback.format_exc()}")


@app.get("/health")
def health():
    return {
        "status": "ready" if model_ready else ("error" if model_error else "loading"),
        "model": "VAST-AI/UniRig",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else None,
        "error": model_error,
    }


def run_unirig_pipeline(input_path: str, output_path: str, timeout: int = 600):
    """Run the full UniRig pipeline: skeleton → skin → merge."""
    work_dir = tempfile.mkdtemp(prefix="unirig_")

    try:
        skeleton_out = os.path.join(work_dir, "skeleton.fbx")
        skin_out = os.path.join(work_dir, "skin.fbx")

        # Step 1: Generate skeleton
        log.info(f"Step 1/3: Generating skeleton for {input_path}")
        cmd_skeleton = [
            "bash", str(UNIRIG_DIR / "launch/inference/generate_skeleton.sh"),
            "--input", input_path,
            "--output", skeleton_out,
        ]
        result = subprocess.run(
            cmd_skeleton, cwd=str(UNIRIG_DIR),
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Skeleton generation failed: {result.stderr[-500:]}")
        log.info("Skeleton generated")

        # Step 2: Generate skin weights
        log.info("Step 2/3: Predicting skin weights...")
        cmd_skin = [
            "bash", str(UNIRIG_DIR / "launch/inference/generate_skin.sh"),
            "--input", skeleton_out,
            "--output", skin_out,
        ]
        result = subprocess.run(
            cmd_skin, cwd=str(UNIRIG_DIR),
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Skin generation failed: {result.stderr[-500:]}")
        log.info("Skin weights predicted")

        # Step 3: Merge skeleton+skin with original mesh
        # The merge uses bpy (Blender Python) which is only available in
        # Blender's bundled Python. We run the merge via Blender's Python.
        log.info("Step 3/3: Merging with original mesh...")
        blender_bin = "/opt/blender/blender"
        merge_script = str(UNIRIG_DIR / "src/inference/merge.py")

        # Try standard merge first (works if bpy is in system python)
        cmd_merge = [
            "bash", str(UNIRIG_DIR / "launch/inference/merge.sh"),
            "--source", skin_out,
            "--target", input_path,
            "--output", output_path,
        ]
        result = subprocess.run(
            cmd_merge, cwd=str(UNIRIG_DIR),
            capture_output=True, text=True, timeout=timeout,
        )

        # If merge failed due to bpy, retry with Blender's python
        if result.returncode != 0 and "bpy" in result.stderr:
            log.info("Retrying merge with Blender's Python...")
            cmd_blender = [
                blender_bin, "--background", "--python", merge_script,
                "--", "--source", skin_out, "--target", input_path,
                "--output", output_path,
            ]
            result = subprocess.run(
                cmd_blender, cwd=str(UNIRIG_DIR),
                capture_output=True, text=True, timeout=timeout,
            )

        if result.returncode != 0 and not os.path.exists(output_path):
            raise RuntimeError(f"Merge failed: {result.stderr[-500:]}")
        log.info(f"Rigged model saved to {output_path}")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/rig")
async def rig_model(file: UploadFile = File(...)):
    """Upload a 3D model, get back a rigged GLB."""
    if not model_ready:
        raise HTTPException(503, detail="Model not ready" + (f": {model_error}" if model_error else ""))

    # Validate file extension
    ext = Path(file.filename or "model.glb").suffix.lower()
    if ext not in (".glb", ".gltf", ".obj", ".fbx"):
        raise HTTPException(400, detail=f"Unsupported format: {ext}. Use .glb, .obj, or .fbx")

    tmp_dir = tempfile.mkdtemp(prefix="unirig_upload_")
    input_path = os.path.join(tmp_dir, f"input{ext}")
    output_path = os.path.join(tmp_dir, "rigged.glb")

    try:
        # Save uploaded file
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)
        log.info(f"Received {file.filename} ({len(content)} bytes)")

        # Run pipeline
        t0 = time.time()
        run_unirig_pipeline(input_path, output_path)
        duration = time.time() - t0
        log.info(f"Rigging completed in {duration:.1f}s")

        if not os.path.exists(output_path):
            raise HTTPException(500, detail="Pipeline completed but output file not found")

        return FileResponse(
            output_path,
            media_type="model/gltf-binary",
            filename="rigged.glb",
            headers={"X-Duration-Seconds": f"{duration:.1f}"},
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Rigging failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, detail=f"Rigging failed: {e}")
    finally:
        # Cleanup after response is sent (FileResponse streams the file)
        # We can't delete immediately — FastAPI needs the file for streaming
        pass


@app.get("/diag")
def diag():
    """Full diagnostic dump."""
    info = {
        "unirig_dir": str(UNIRIG_DIR),
        "unirig_exists": UNIRIG_DIR.exists(),
        "checkpoints": str(UNIRIG_DIR / "checkpoints"),
        "checkpoints_exists": (UNIRIG_DIR / "checkpoints").exists(),
        "model_ready": model_ready,
        "model_error": model_error,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "python_version": sys.version,
    }
    if (UNIRIG_DIR / "checkpoints").exists():
        info["checkpoint_files"] = [
            str(p.relative_to(UNIRIG_DIR / "checkpoints"))
            for p in (UNIRIG_DIR / "checkpoints").rglob("*")
            if p.is_file()
        ][:20]
    return info


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
