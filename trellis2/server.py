"""TRELLIS.2 image-to-3D FastAPI server. MIT license.

Endpoints:
  GET  /health
  POST /v1/3d/generate  multipart {image, pipeline_type=512|1024|1024_cascade|1536_cascade,
                                   seed?, decimation_target?}
                        returns GLB binary (model/gltf-binary).

Mirrors upstream `example.py`: pipeline.run(image) → mesh.simplify() →
o_voxel.postprocess.to_glb(...) → .glb on disk.
"""
import io
import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

state = {"pipe": None, "loaded": False, "load_error": None}


def _load() -> None:
    if state["pipe"] is not None:
        return
    try:
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
        pipe.cuda()
        state["pipe"] = pipe
        state["loaded"] = True
    except Exception as e:
        state["load_error"] = repr(e)
        raise


@asynccontextmanager
async def lifespan(_app):
    try:
        _load()
    except Exception:
        pass
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "healthy" if state["loaded"] else "loading",
        "model": "microsoft/TRELLIS.2-4B",
        "model_loaded": state["loaded"],
        "load_error": state["load_error"],
    }


_VALID_PIPELINE_TYPES = {"512", "1024", "1024_cascade", "1536_cascade"}


@app.post("/v1/3d/generate")
async def generate(
    image: UploadFile = File(...),
    pipeline_type: str = Form("1024"),
    seed: int = Form(42),
    decimation_target: int = Form(1000000),
    texture_size: int = Form(4096),
):
    if pipeline_type not in _VALID_PIPELINE_TYPES:
        raise HTTPException(
            400, f"pipeline_type must be one of {sorted(_VALID_PIPELINE_TYPES)}"
        )
    if not state["loaded"]:
        try:
            _load()
        except Exception as e:
            raise HTTPException(503, f"model not ready: {e}")

    img_bytes = await image.read()
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    import o_voxel  # type: ignore[import-not-found]

    t0 = time.time()
    meshes = state["pipe"].run(pil, seed=seed, pipeline_type=pipeline_type)
    if not meshes:
        raise HTTPException(500, "pipeline returned no meshes")
    mesh = meshes[0]
    mesh.simplify(16777216)  # nvdiffrast limit

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=False,
    )

    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        path = tmp.name
    try:
        glb.export(path, extension_webp=True)
        with open(path, "rb") as f:
            body = f.read()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    elapsed = round(time.time() - t0, 2)
    return Response(
        content=body,
        media_type="model/gltf-binary",
        headers={
            "X-Elapsed-Seconds": str(elapsed),
            "X-Pipeline-Type": pipeline_type,
        },
    )
