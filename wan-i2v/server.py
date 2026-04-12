"""
Wan 2.1 Image-to-Video Service — FastAPI server para animar planos de fundo.

Modelo: Wan-AI/Wan2.1-I2V-14B-480P
  - Menor modelo I2V da família Wan 2.1
  - Requer: ~16GB VRAM com CPU offload, 24GB VRAM nativo
  - Nota: Wan 2.1 1.3B é apenas Text-to-Video; não existe 1.3B I2V oficial.
    Este é o menor I2V disponível.

Uso:
  python server.py [--port 8095] [--device cuda] [--cpu-offload]

Endpoints:
  GET  /health            — status + modelo carregado
  POST /generate          — imagem base64 → vídeo base64 (MP4)
  GET  /presets           — lista de prompts prontos para cenas
"""

import argparse
import base64
import io
import os
import sys
import time
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--port",        type=int,  default=int(os.environ.get("PORT", "8000")))
parser.add_argument("--device",      type=str,  default="cuda", choices=["cuda", "cpu"])
parser.add_argument("--cpu-offload", action="store_true", default=True,
                    help="Habilita CPU offload — permite rodar com menos VRAM (16GB+)")
parser.add_argument("--model-id",    type=str,
                    default=os.environ.get("MODEL_ID", "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"),
                    help="HuggingFace model ID (diffusers format)")
args, _ = parser.parse_known_args()

# ── Prompt presets para cenas panorâmicas ─────────────────────────────────────

SCENE_PRESETS = {
    "people_walking": (
        "subtle natural movement, people walking slowly and naturally, "
        "gentle ambient animation, camera static, photorealistic, "
        "no camera shake, cinematic, smooth motion"
    ),
    "ambient": (
        "gentle ambient movement, leaves rustling, subtle light changes, "
        "atmospheric, calm, camera static, photorealistic, cinematic"
    ),
    "crowd": (
        "crowd moving naturally, people talking and gesturing, "
        "urban life, camera static, photorealistic, cinematic, smooth"
    ),
    "nature": (
        "gentle wind through trees, flowing water, birds moving subtly, "
        "nature alive, camera static, photorealistic, cinematic"
    ),
    "night": (
        "city lights flickering gently, people moving slowly, "
        "night atmosphere, camera static, photorealistic, cinematic"
    ),
}

NEGATIVE_PROMPT = (
    "static, frozen, no movement, blurry, overexposed, "
    "camera shake, distorted, artifacts, low quality"
)

# ── Pipeline global ───────────────────────────────────────────────────────────

pipeline = None
model_loaded = False
load_error: Optional[str] = None
load_traceback: Optional[str] = None

def load_pipeline():
    global pipeline, model_loaded, load_error, load_traceback
    try:
        print(f"[wan-i2v] Carregando modelo: {args.model_id}", flush=True)
        print(f"[wan-i2v] Device: {args.device} | CPU offload: {args.cpu_offload}", flush=True)
        t0 = time.time()

        from diffusers import WanImageToVideoPipeline

        dtype = torch.bfloat16 if args.device == "cuda" else torch.float32

        pipe = WanImageToVideoPipeline.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
        )

        if args.cpu_offload and args.device == "cuda":
            # Mantém apenas o componente ativo na VRAM — usa ~16GB em vez de 24GB
            pipe.enable_model_cpu_offload()
            print("[wan-i2v] CPU offload ativado — VRAM reduzida para ~16GB", flush=True)
        else:
            pipe = pipe.to(args.device)

        pipeline = pipe
        model_loaded = True
        print(f"[wan-i2v] Modelo carregado em {time.time()-t0:.1f}s", flush=True)

    except Exception as e:
        import traceback as tb
        load_error = str(e)
        load_traceback = tb.format_exc()
        print(f"[wan-i2v] ERRO ao carregar modelo: {e}", flush=True, file=sys.stderr)
        print(load_traceback, flush=True, file=sys.stderr)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_pipeline()
    yield

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Wan 2.1 I2V Service",
    description="Image-to-Video para animação de planos de fundo panorâmicos",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response schemas ────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    image_b64: str = Field(..., description="Imagem base64 (JPEG/PNG) do frame panorâmico")
    prompt: str = Field(
        default=SCENE_PRESETS["people_walking"],
        description="Prompt descrevendo o movimento desejado",
    )
    negative_prompt: str = Field(default=NEGATIVE_PROMPT)
    preset: Optional[str] = Field(
        default=None,
        description=f"Preset de cena: {list(SCENE_PRESETS.keys())}. Sobrescreve 'prompt' se informado.",
    )
    num_frames: int = Field(default=81, ge=16, le=121,
                            description="Número de frames. 81 @ 16fps = ~5 segundos")
    fps: int = Field(default=16, ge=8, le=24)
    height: int = Field(default=480, description="Altura do vídeo em pixels")
    width: int = Field(default=832, description="Largura do vídeo em pixels")
    guidance_scale: float = Field(default=5.0, ge=1.0, le=10.0)
    num_inference_steps: int = Field(default=30, ge=10, le=50)

class GenerateResponse(BaseModel):
    video_b64: str
    content_type: str = "video/mp4"
    num_frames: int
    fps: int
    duration_seconds: float
    elapsed_ms: int

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    vram_gb = None
    if torch.cuda.is_available():
        try:
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        except Exception:
            pass
    return {
        "status": "ok" if model_loaded else ("error" if load_error else "loading"),
        "model": args.model_id,
        "device": args.device,
        "cpu_offload": args.cpu_offload,
        "model_loaded": model_loaded,
        "load_error": load_error,
        # Gateway reads "error" key for fail-fast detection
        "error": load_error if load_error else None,
        "error_traceback": load_traceback if load_traceback else None,
        "cuda_available": torch.cuda.is_available(),
        "vram_gb": vram_gb,
    }


@app.get("/presets")
def get_presets():
    return {
        "presets": {k: v for k, v in SCENE_PRESETS.items()},
        "negative_prompt": NEGATIVE_PROMPT,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if not model_loaded:
        raise HTTPException(503, detail=load_error or "Modelo ainda carregando, aguarde.")

    # ── Decode image ──
    try:
        img_bytes = base64.b64decode(req.image_b64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        # Resize to match requested resolution
        image = image.resize((req.width, req.height), Image.LANCZOS)
    except Exception as e:
        raise HTTPException(400, detail=f"Imagem inválida: {e}")

    # ── Resolve prompt ──
    prompt = req.prompt
    if req.preset and req.preset in SCENE_PRESETS:
        prompt = SCENE_PRESETS[req.preset]

    # ── Generate video ──
    t0 = time.time()
    try:
        output = pipeline(
            image=image,
            prompt=prompt,
            negative_prompt=req.negative_prompt,
            height=req.height,
            width=req.width,
            num_frames=req.num_frames,
            guidance_scale=req.guidance_scale,
            num_inference_steps=req.num_inference_steps,
        )
        frames = output.frames[0]  # list of PIL Images
    except Exception as e:
        raise HTTPException(500, detail=f"Erro na geração: {e}")

    elapsed_ms = int((time.time() - t0) * 1000)

    # ── Export to MP4 in memory ──
    try:
        import imageio
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        imageio.mimwrite(tmp_path, [np.array(f) for f in frames], fps=req.fps, quality=8)

        with open(tmp_path, "rb") as f:
            video_bytes = f.read()
        os.unlink(tmp_path)

        video_b64 = base64.b64encode(video_bytes).decode()
    except Exception as e:
        raise HTTPException(500, detail=f"Erro ao exportar vídeo: {e}")

    duration = req.num_frames / req.fps
    print(f"[wan-i2v] Gerado: {req.num_frames} frames @ {req.fps}fps = {duration:.1f}s em {elapsed_ms}ms", flush=True)

    return GenerateResponse(
        video_b64=video_b64,
        num_frames=req.num_frames,
        fps=req.fps,
        duration_seconds=duration,
        elapsed_ms=elapsed_ms,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
