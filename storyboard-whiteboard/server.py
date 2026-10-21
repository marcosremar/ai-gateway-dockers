"""Storyboard whiteboard animation server — FastAPI wrapper around the
OpenCV object-by-object hand-drawn engine bundled in `engine.py`.

Replaces our previous SVG-turbulence WhiteboardDraw.tsx (Pattern 9) with the
legit progressive-pixel-fill + real hand sprite traveling along contours.

Routes:
  GET  /health                         — readiness (no model — instant).
  POST /v1/whiteboard/animate          — multipart input, returns mp4 OR
                                          {job_id, video_url}.
  GET  /results/<job_id>/output.mp4    — serves rendered file.

Env:
  IDLE_TIMEOUT_MIN     — auto-shutdown after no requests (default 15).
  PORT                 — listen port (default 8000).
  WB_RESULTS_DIR       — where rendered mp4s land (default /app/results).
  WB_BUNDLED_HAND      — bundled hand sprite path (default /app/assets/...).
  WB_BUNDLED_HAND_MASK — bundled hand mask path.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
import uvicorn

from idle_watchdog import add_idle_middleware, start_watchdog, touch_activity
from engine import PRESETS, render_whiteboard


RESULTS_DIR = Path(os.environ.get("WB_RESULTS_DIR", "/app/results"))
JOBS_DIR = Path(os.environ.get("WB_JOBS_DIR", "/app/jobs"))
BUNDLED_HAND = Path(os.environ.get("WB_BUNDLED_HAND", "/app/assets/drawing-hand.png"))
BUNDLED_HAND_MASK = Path(os.environ.get("WB_BUNDLED_HAND_MASK", "/app/assets/hand-mask.png"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# Local-dev fallback: when running outside the container, the bundled assets
# live next to this file (./assets/...). Resolve them so smoke tests work.
if not BUNDLED_HAND.exists():
    BUNDLED_HAND = Path(__file__).parent / "assets" / "drawing-hand.png"
if not BUNDLED_HAND_MASK.exists():
    BUNDLED_HAND_MASK = Path(__file__).parent / "assets" / "hand-mask.png"

MAX_RENDER_SECONDS = int(os.environ.get("WB_MAX_RENDER_SECONDS", "300"))
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(start_watchdog())
    yield


app = FastAPI(title="storyboard-whiteboard (ai-gateway)", lifespan=lifespan)
add_idle_middleware(app)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": True,  # no weights — engine is always ready
        "engine": "storyboard-ai/opencv-whiteboard",
        "results_dir": str(RESULTS_DIR),
        "bundled_hand": str(BUNDLED_HAND),
        "bundled_hand_mask": str(BUNDLED_HAND_MASK),
        "presets": list(PRESETS.keys()),
    }


@app.get("/v1/whiteboard/presets")
async def list_presets():
    """Returns the registered presets and their canonical style fields. Useful
    for the orchestrator when the user picks "renovadamente"-style content
    pre-emptively."""
    out = {}
    for name, cfg in PRESETS.items():
        # Convert BGR tuples back to RGB hex for client-side display.
        bg = cfg["background_color"]
        out[name] = {
            "background_color_hex": "#{:02x}{:02x}{:02x}".format(bg[2], bg[1], bg[0]),
            "accent_palette_hex": [
                "#{:02x}{:02x}{:02x}".format(c[2], c[1], c[0])
                for c in cfg["accent_palette"]
            ],
            "stroke_speed": cfg["stroke_speed"],
            "end_duration_s": cfg["end_duration_s"],
        }
    return out


def _clamp(name: str, value: int, lo: int, hi: int) -> int:
    if value < lo or value > hi:
        raise HTTPException(400, f"{name} must be in [{lo}, {hi}], got {value}")
    return value


async def _save_upload(upload: UploadFile, dest: Path, max_mb: int = 50) -> None:
    data = await upload.read()
    if len(data) == 0:
        raise HTTPException(400, f"empty upload: {upload.filename}")
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(413, f"upload too large: {upload.filename} ({len(data)} bytes)")
    dest.write_bytes(data)


@app.post("/v1/whiteboard/animate")
async def animate(
    image: UploadFile = File(..., description="PNG line drawing"),
    mask: Optional[UploadFile] = File(None, description="LabelMe-style JSON segmentation"),
    hand: Optional[UploadFile] = File(None, description="custom hand PNG sprite"),
    hand_mask: Optional[UploadFile] = File(None, description="custom hand mask PNG"),
    frame_rate: int = Form(25),
    resize: int = Form(1080),
    split_len: int = Form(10),
    object_skip_rate: int = Form(8),
    bg_object_skip_rate: int = Form(14),
    end_duration_s: int = Form(3),
    # ── Style controls ──────────────────────────────────────────────
    preset: Optional[str] = Form(
        None,
        description="Named style preset (e.g. 'renovadamente'). See GET /v1/whiteboard/presets.",
    ),
    background_color: Optional[str] = Form(
        None,
        description="Hex like '#ffffff'. Overrides preset / default canvas.",
    ),
    accent_palette: Optional[str] = Form(
        None,
        description="Comma-separated hex colors '#e53e3e,#facc15,...' applied to largest connected components.",
    ),
    stroke_speed: Optional[float] = Form(
        None,
        description="Multiplier on stroke skip rates (1.0=default, 1.5=faster, 0.7=slower).",
    ),
    response_format: str = Form("url", description="url | file"),
):
    touch_activity()

    frame_rate = _clamp("frame_rate", frame_rate, 5, 60)
    resize = _clamp("resize", resize, 256, 2160)
    split_len = _clamp("split_len", split_len, 8, 20)
    object_skip_rate = _clamp("object_skip_rate", object_skip_rate, 4, 15)
    bg_object_skip_rate = _clamp("bg_object_skip_rate", bg_object_skip_rate, 10, 20)
    end_duration_s = _clamp("end_duration_s", end_duration_s, 0, 10)

    if preset is not None and preset not in PRESETS:
        raise HTTPException(400, f"unknown preset {preset!r}; have {list(PRESETS)}")
    if stroke_speed is not None and (stroke_speed < 0.25 or stroke_speed > 4.0):
        raise HTTPException(400, "stroke_speed must be in [0.25, 4.0]")
    palette_list = None
    if accent_palette:
        palette_list = [p.strip() for p in accent_palette.split(",") if p.strip()]
        if len(palette_list) > 12:
            raise HTTPException(400, "accent_palette max 12 colors")

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    image_path = job_dir / "input.png"
    await _save_upload(image, image_path)

    mask_path: Optional[Path] = None
    if mask is not None and mask.filename:
        mask_path = job_dir / "mask.json"
        await _save_upload(mask, mask_path, max_mb=10)

    hand_path = BUNDLED_HAND
    if hand is not None and hand.filename:
        hand_path = job_dir / "hand.png"
        await _save_upload(hand, hand_path, max_mb=10)

    hand_mask_path = BUNDLED_HAND_MASK
    if hand_mask is not None and hand_mask.filename:
        hand_mask_path = job_dir / "hand_mask.png"
        await _save_upload(hand_mask, hand_mask_path, max_mb=10)

    output_dir = RESULTS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "output.mp4"

    t0 = time.time()
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                lambda: render_whiteboard(
                    image_path=str(image_path),
                    mask_path=str(mask_path) if mask_path else None,
                    hand_path=str(hand_path),
                    hand_mask_path=str(hand_mask_path),
                    output_path=str(output_path),
                    frame_rate=frame_rate,
                    resize=resize,
                    split_len=split_len,
                    object_skip_rate=object_skip_rate,
                    bg_object_skip_rate=bg_object_skip_rate,
                    end_duration_s=end_duration_s,
                    preset=preset,
                    background_color=background_color,
                    accent_palette=palette_list,
                    stroke_speed=stroke_speed,
                )
            ),
            timeout=MAX_RENDER_SECONDS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(504, f"whiteboard render exceeded {MAX_RENDER_SECONDS}s")
    except Exception as e:
        raise HTTPException(500, f"whiteboard render failed: {e}")

    wall_s = round(time.time() - t0, 3)

    if response_format == "file":
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=f"whiteboard-{job_id}.mp4",
        )

    return {
        "job_id": job_id,
        "video_url": f"/results/{job_id}/output.mp4",
        "duration_s": result["duration_s"],
        "frames_total": result["frames_total"],
        "render_time_s": result["render_time_s"],
        "wall_time_s": wall_s,
        "preset": result.get("preset", "default"),
    }


@app.get("/results/{job_id}/{filename}")
async def serve_result(job_id: str, filename: str):
    if not SAFE_ID_RE.match(job_id):
        raise HTTPException(400, "invalid job_id")
    if "/" in filename or ".." in filename or not SAFE_ID_RE.match(filename.split(".")[0]):
        raise HTTPException(400, "invalid filename")
    target = RESULTS_DIR / job_id / filename
    if not target.exists():
        raise HTTPException(404, "not found")
    media = "video/mp4" if filename.endswith(".mp4") else "application/octet-stream"
    return FileResponse(target, media_type=media)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
