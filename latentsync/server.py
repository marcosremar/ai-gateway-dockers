"""LatentSync Service — FastAPI wrapper pro ai-gateway.

Endpoints:
  GET  /health        — readiness probe (gateway autoscaler).
  GET  /v1/demo       — minimal HTML form to manually exercise /v1/lipsync.
  POST /v1/lipsync    — multipart `image` (image OR mp4) + `audio` (wav/mp3).
                        Returns MP4 of the input face/video lipsynced to the
                        audio. Compatible with `marcosremar/musetalk` so the
                        ai-gateway pipeline can switch engines without code
                        changes on the caller side.

LatentSync's upstream inference is a CLI script (`./inference.sh`) that wraps
`predict.py`. We invoke it as a subprocess per request — simpler than
re-implementing the engine in-process and matches how MuseTalk's docker is
already used by the gateway. Scaling concurrency is delegated to the gateway
race controller (one container = one job at a time today; ok because each
request is GPU-bound and saturates the device).

Env:
  IDLE_TIMEOUT_MIN          — auto-shutdown after no requests (default 15).
  LATENTSYNC_VARIANT        — `1.5` (8GB VRAM) or `1.6` (18GB). Default 1.5.
  LATENTSYNC_CHECKPOINT_DIR — overrides where weights are looked up.
"""

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
import uvicorn

from idle_watchdog import add_idle_middleware, start_watchdog, touch_activity

REPO_DIR = Path("/app/LatentSync")
CKPT_DIR = Path(os.environ.get("LATENTSYNC_CHECKPOINT_DIR", "/app/checkpoints"))
VARIANT = os.environ.get("LATENTSYNC_VARIANT", "1.5")

# Single global lock — one inference per container (GPU-bound).
inference_lock = asyncio.Lock()
boot_ts = time.time()
load_error: str | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not REPO_DIR.exists():
        global load_error
        load_error = f"LatentSync repo not found at {REPO_DIR}"
    asyncio.create_task(start_watchdog())
    yield


app = FastAPI(title="LatentSync (ai-gateway)", lifespan=lifespan)
add_idle_middleware(app)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "healthy" if load_error is None else "degraded",
        "model_loaded": load_error is None,
        "variant": VARIANT,
        "checkpoint_dir": str(CKPT_DIR),
        "uptime_s": round(time.time() - boot_ts, 1),
        "load_error": load_error,
    }


@app.get("/v1/demo", response_class=HTMLResponse)
async def demo() -> str:
    return """<!doctype html><html><body style="font-family:sans-serif;max-width:560px;margin:40px auto">
<h2>LatentSync — one-shot demo</h2>
<form enctype="multipart/form-data" method="post" action="/v1/lipsync">
  <label>Imagem ou vídeo: <input type="file" name="image" accept="image/*,video/mp4" required></label><br><br>
  <label>Áudio: <input type="file" name="audio" accept="audio/*" required></label><br><br>
  <button>Gerar</button>
</form>
</body></html>"""


def _resolve_checkpoint() -> Path:
    # LatentSync's snapshot from HF includes a top-level `latentsync_unet.pt`
    # (1.5) or `latentsync_unet_1.6.pt` (1.6). We check both shapes and fall
    # back to the variant-tagged marker the start script writes.
    candidates = [
        CKPT_DIR / "latentsync_unet.pt",
        CKPT_DIR / f"latentsync_unet_{VARIANT}.pt",
        CKPT_DIR / "checkpoints" / "latentsync_unet.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"No LatentSync checkpoint found under {CKPT_DIR} (looked for {[str(c) for c in candidates]})",
    )


def _resolve_config() -> Path:
    cfg = REPO_DIR / "configs" / f"unet" / f"stage2{'_512' if VARIANT.startswith('1.6') else ''}.yaml"
    if cfg.exists():
        return cfg
    fallback = REPO_DIR / "configs" / "unet" / "stage2.yaml"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No LatentSync stage2 config under {REPO_DIR / 'configs/unet'}")


@app.post("/v1/lipsync")
async def lipsync(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
) -> Response:
    touch_activity()

    if load_error is not None:
        raise HTTPException(503, f"Model not loaded: {load_error}")

    workdir = Path(tempfile.mkdtemp(prefix="latentsync_"))
    request_id = uuid.uuid4().hex[:8]
    in_video = workdir / "input.mp4"
    in_audio = workdir / "input.wav"
    out_video = workdir / f"out_{request_id}.mp4"

    try:
        # LatentSync expects an mp4 video as the source. If the caller posts a
        # still image, we synthesize a 1-second loop so the engine has frames
        # to inpaint over.
        suffix = (image.filename or "").lower().split(".")[-1]
        is_image = suffix in {"jpg", "jpeg", "png", "webp", "bmp"}
        raw_in = workdir / f"input_raw.{suffix or 'bin'}"
        raw_in.write_bytes(await image.read())
        if is_image:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loop", "1", "-t", "1", "-i", str(raw_in),
                    "-vf", "scale=512:512:force_original_aspect_ratio=increase,crop=512:512",
                    "-pix_fmt", "yuv420p", "-r", "25", str(in_video),
                ],
                check=True, capture_output=True,
            )
        else:
            shutil.move(str(raw_in), str(in_video))

        in_audio.write_bytes(await audio.read())

        ckpt = _resolve_checkpoint()
        cfg = _resolve_config()

        cmd = [
            "python3", "-m", "scripts.inference",
            "--unet_config_path", str(cfg),
            "--inference_ckpt_path", str(ckpt),
            "--video_path", str(in_video),
            "--audio_path", str(in_audio),
            "--video_out_path", str(out_video),
            "--seed", "1247",
        ]

        async with inference_lock:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(REPO_DIR),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            stdout, _ = await proc.communicate()

        if proc.returncode != 0 or not out_video.exists():
            tail = (stdout or b"").decode(errors="replace")[-2000:]
            raise HTTPException(500, f"LatentSync inference failed (exit {proc.returncode}):\n{tail}")

        return Response(content=out_video.read_bytes(), media_type="video/mp4")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        log_level="info",
    )
