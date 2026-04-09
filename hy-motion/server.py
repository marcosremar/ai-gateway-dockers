"""
HY-Motion-1.0 inference server.

HTTP:
    POST /predict   { "text": "...", "duration": 5.0, "cfg_scale": 5.0,
                      "num_seeds": 1, "format": "bvh" | "gltf" }
                    → { "status": "ok", "format": "bvh", "data_base64": "...",
                        "duration": 5.0, "rewritten_text": "...", "latency_ms": 12345 }
    GET  /health    → {"status": "loading"} while models load,
                      {"status": "ok", "device": "cuda"} when ready

Notes:
    - Model loads on lifespan startup, so /health responds with "loading"
      until the LLM encoder + DiT + CLIP are all in VRAM. On a cold RTX 5090
      with weights pre-baked into the image, this takes ~2-4 minutes.
    - We default to --disable_rewrite --disable_duration_est because the
      Text2MotionPrompter LLM rewriter isn't pre-baked (it would add another
      ~16GB to the image). Callers can supply their own rewritten prompt
      via the `text` field if they want LLM polishing.
    - FBX export needs the proprietary Autodesk SDK (`fbxsdkpy`) which we
      don't ship. Output formats supported: BVH and GLTF.
"""
import asyncio
import base64
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

sys.path.insert(0, "/app/HY-Motion-1.0")

# ---------------------------------------------------------------------------
# Config / paths
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("HY_MOTION_MODEL_PATH", "/app/HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0")
OUTPUT_ROOT = os.environ.get("HY_MOTION_OUTPUT_ROOT", "/tmp/hy-motion-out")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Default request shape values, also documented in the README.
DEFAULT_DURATION = 5.0
DEFAULT_CFG_SCALE = 5.0
DEFAULT_NUM_SEEDS = 1     # 1 = lowest VRAM, fastest. Bump for diversity.
# T2MRuntime.generate_motion supports two output_format values:
#   "fbx"  → writes a real .fbx file (needs fbxsdkpy, which we ship)
#   "dict" → writes a JSON file with raw SMPL/SMPL-H pose params per frame
# We expose both so callers can pick: "fbx" for direct skeletal animation,
# "dict" for retargeting onto a different rig (e.g. Sofia / Ready Player Me).
DEFAULT_FORMAT = "fbx"
SUPPORTED_FORMATS = {"fbx", "dict"}

# ---------------------------------------------------------------------------
# Global model state — populated during lifespan startup so /health responds
# immediately with "loading" while the model is warming up.
# ---------------------------------------------------------------------------
_models_ready = False
_runtime = None
_device_str = "cpu"


def _ensure_dit_weights() -> str:
    """Make sure the HY-Motion DiT generator is on local disk.

    T2MRuntime needs file paths (`config_path`, `ckpt_name`), not HF repo IDs,
    so we snapshot_download the DiT generator into MODEL_PATH on first boot
    and reuse it on subsequent boots if the cache survived.

    The text encoders (CLIP-L and Qwen3-8B) are handled separately by
    transformers' own cache via USE_HF_MODELS=1 — see text_encoder.py in the
    HY-Motion repo. We don't need to materialize those into MODEL_PATH.
    """
    cfg = os.path.join(MODEL_PATH, "config.yml")
    ckpt = os.path.join(MODEL_PATH, "latest.ckpt")
    if os.path.exists(cfg) and os.path.exists(ckpt):
        print(f"[hy-motion] DiT weights already cached at {MODEL_PATH}", flush=True)
        return MODEL_PATH

    print(f"[hy-motion] Downloading HY-Motion DiT weights to {MODEL_PATH}...", flush=True)
    from huggingface_hub import snapshot_download
    # The repo lays the model out under HY-Motion-1.0/<files>; download into
    # the parent dir so the result lands at /app/HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0/
    parent = os.path.dirname(MODEL_PATH)
    os.makedirs(parent, exist_ok=True)
    snapshot_download(
        "tencent/HY-Motion-1.0",
        allow_patterns=["HY-Motion-1.0/*"],
        local_dir=parent,
    )
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Download finished but checkpoint missing at {ckpt}")
    print(f"[hy-motion] DiT weights ready at {MODEL_PATH}", flush=True)
    return MODEL_PATH


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _models_ready, _runtime, _device_str

    import torch
    _device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[hy-motion] Torch device: {_device_str}", flush=True)
    if _device_str == "cuda":
        try:
            print(f"[hy-motion] CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
            print(f"[hy-motion] CUDA capability: {torch.cuda.get_device_capability(0)}", flush=True)
        except Exception as e:
            print(f"[hy-motion] CUDA introspection failed: {e}", flush=True)

    # First boot: download DiT generator (~4 GB) into MODEL_PATH. CLIP-L and
    # Qwen3-8B (~18 GB combined) are downloaded automatically by transformers
    # the first time T2MRuntime instantiates them, into HF_HOME.
    _ensure_dit_weights()

    cfg = os.path.join(MODEL_PATH, "config.yml")
    ckpt = os.path.join(MODEL_PATH, "latest.ckpt")

    print(f"[hy-motion] Loading T2MRuntime from {MODEL_PATH}...", flush=True)
    print(f"[hy-motion] First load also downloads CLIP-L (~1.7 GB) and Qwen3-8B (~16 GB) to HF cache", flush=True)
    from hymotion.utils.t2m_runtime import T2MRuntime
    _runtime = T2MRuntime(
        config_path=cfg,
        ckpt_name=ckpt,
        device_ids=None,
        # disable the prompt-engineering LLM (Text2MotionPrompter) since we
        # don't ship its weights. Callers send pre-formatted prompts.
        disable_prompt_engineering=True,
        prompt_engineering_host=None,
        prompt_engineering_model_path=None,
    )

    _models_ready = True
    print("Application startup complete.", flush=True)

    yield  # server runs here

    print("[hy-motion] Shutting down.", flush=True)


app = FastAPI(title="HY-Motion-1.0", description="Tencent text-to-3D-motion generator", lifespan=lifespan)

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
    text: str = Field(..., description="English natural-language motion prompt, <60 words")
    duration: float = Field(DEFAULT_DURATION, ge=1.0, le=10.0, description="Output motion length in seconds")
    cfg_scale: float = Field(DEFAULT_CFG_SCALE, ge=1.0, le=15.0, description="Classifier-free guidance scale")
    num_seeds: int = Field(DEFAULT_NUM_SEEDS, ge=1, le=4, description="Random seed count (1 = least VRAM)")
    format: str = Field(DEFAULT_FORMAT, description="Output format: fbx or dict")
    seed: Optional[int] = Field(None, description="Override seed (else random)")


class PredictResponse(BaseModel):
    status: str
    format: str
    # base64-encoded motion file (BVH text or GLTF binary)
    data_base64: str
    duration: float
    rewritten_text: str
    latency_ms: float


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------

def _run_inference(req: PredictRequest) -> dict:
    """Run T2MRuntime and return the requested output format as base64."""
    import random
    if req.format not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format '{req.format}'. Use one of: {sorted(SUPPORTED_FORMATS)}")

    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(OUTPUT_ROOT, job_id)
    os.makedirs(job_dir, exist_ok=True)

    seeds = [req.seed if req.seed is not None else random.randint(0, 999999) for _ in range(req.num_seeds)]
    seeds_csv = ",".join(map(str, seeds))

    # T2MRuntime exposes `generate_motion` which writes one or more files to
    # disk and returns (html, fbx_files, _). We pass output_format = our
    # format string and read the resulting file off disk afterwards.
    t0 = time.perf_counter()
    _, files, _ = _runtime.generate_motion(
        text=req.text,
        seeds_csv=seeds_csv,
        duration=req.duration,
        cfg_scale=req.cfg_scale,
        output_format=req.format,
        original_text=req.text,
        output_dir=job_dir,
        output_filename=job_id,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    if not files:
        raise RuntimeError("HY-Motion produced no output files")

    # Pick the first generated file (we requested num_seeds=1 by default)
    out_path = files[0] if isinstance(files, list) else files
    if not os.path.exists(out_path):
        raise RuntimeError(f"Generated file not found at {out_path}")

    with open(out_path, "rb") as f:
        data = f.read()

    return {
        "status": "ok",
        "format": req.format,
        "data_base64": base64.b64encode(data).decode("ascii"),
        "duration": req.duration,
        "rewritten_text": req.text,  # we disabled rewrite, so it's identical
        "latency_ms": round(latency_ms, 1),
    }


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    if not _models_ready:
        return {"status": "loading", "device": _device_str}
    return {"status": "ok", "device": _device_str, "model_path": MODEL_PATH}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if not _models_ready:
        raise HTTPException(status_code=503, detail="Models still loading")
    try:
        loop = asyncio.get_event_loop()
        # Run the (long, GPU-bound) inference in a worker thread so we don't
        # block the event loop and so the /health endpoint stays responsive.
        result = await loop.run_in_executor(None, _run_inference, req)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


if __name__ == "__main__":
    os.makedirs("/var/log/portal", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
