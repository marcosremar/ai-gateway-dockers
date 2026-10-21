"""Manim Community diagram-render server — FastAPI wrapper para ai-gateway.

Exposes a single custom endpoint so the canal-dark / project-philosofi pipeline
can render Manim Scene classes from arbitrary Python source (AST-validated).

  GET  /health                      — readiness probe.
  POST /v1/diagrams/generate        — JSON {scene_class, scene_code, quality}.

Env:
  IDLE_TIMEOUT_MIN    — auto-shutdown after no requests (default 15).
  PORT                — listen port (default 8000).
  MANIM_RESULTS_DIR   — where rendered mp4s land (default /app/results).
"""

import ast
import asyncio
import base64
import os
import re
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

from idle_watchdog import add_idle_middleware, start_watchdog, touch_activity


RESULTS_DIR = Path(os.environ.get("MANIM_RESULTS_DIR", "/app/results"))
SCENES_DIR = Path("/app/scenes")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SCENES_DIR.mkdir(parents=True, exist_ok=True)

# ── Hard limits to keep the worker safe on shared CPU pods ───────────────────
MAX_CODE_SIZE = 64 * 1024            # 64KB of source per scene
MAX_RENDER_SECONDS = 180             # kill manim after 3 min
QUALITY_MAP = {"l": "low_quality", "m": "medium_quality", "h": "high_quality"}

# ── AST validation ───────────────────────────────────────────────────────────
FORBIDDEN_MODULES = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "pathlib",
    "urllib", "http", "ftplib", "asyncio", "threading", "multiprocessing",
    "ctypes", "importlib", "pickle", "marshal", "code", "codeop",
    "requests", "httpx", "aiohttp", "smtplib", "ssl",
})
FORBIDDEN_BUILTINS = frozenset({
    "eval", "exec", "compile", "open", "__import__", "input",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "memoryview", "breakpoint",
})
FORBIDDEN_ATTRS = frozenset({
    "__class__", "__bases__", "__subclasses__", "__mro__",
    "__globals__", "__builtins__", "__import__", "__loader__",
    "__dict__", "__getattribute__", "__reduce__", "__reduce_ex__",
})


def _validate_scene_code(code: str) -> None:
    """Walk the AST and reject anything that lets the scene escape the sandbox."""
    if len(code) > MAX_CODE_SIZE:
        raise HTTPException(400, f"scene_code too large ({len(code)} > {MAX_CODE_SIZE})")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise HTTPException(400, f"scene_code SyntaxError: {e.msg} (line {e.lineno})")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_MODULES:
                    raise HTTPException(400, f"forbidden import: {alias.name}")
                if root != "manim" and root != "numpy" and root != "math":
                    raise HTTPException(400, f"only `manim`, `numpy`, `math` imports allowed (got {alias.name})")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in FORBIDDEN_MODULES:
                raise HTTPException(400, f"forbidden from-import: {node.module}")
            if root not in {"manim", "numpy", "math"}:
                raise HTTPException(400, f"only `manim`, `numpy`, `math` from-imports allowed (got {node.module})")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_BUILTINS:
                raise HTTPException(400, f"forbidden builtin call: {func.id}")
            if isinstance(func, ast.Attribute) and func.attr in FORBIDDEN_BUILTINS:
                raise HTTPException(400, f"forbidden attr call: {func.attr}")
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRS:
                raise HTTPException(400, f"forbidden attribute access: {node.attr}")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise HTTPException(400, "global/nonlocal not allowed")


_SAFE_CLASS_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(start_watchdog())
    yield


app = FastAPI(title="Manim CE diagram render (ai-gateway)", lifespan=lifespan)
add_idle_middleware(app)


@app.get("/health")
async def health():
    return {"status": "ok", "engine": "manim-community", "results_dir": str(RESULTS_DIR)}


class DiagramRequest(BaseModel):
    scene_class: str = Field(..., description="Top-level Scene subclass name")
    scene_code: str = Field(..., description="Python source defining the Scene")
    quality: str = Field("h", description="l=480p, m=720p, h=1080p")
    response_format: str = Field("url", description="url | b64_json")


@app.post("/v1/diagrams/generate")
async def generate_diagram(req: DiagramRequest):
    touch_activity()

    if req.quality not in QUALITY_MAP:
        raise HTTPException(400, f"quality must be one of {list(QUALITY_MAP)}")
    if not _SAFE_CLASS_RE.match(req.scene_class):
        raise HTTPException(400, "invalid scene_class identifier")
    _validate_scene_code(req.scene_code)
    if req.scene_class not in req.scene_code:
        raise HTTPException(400, f"scene_code must define class {req.scene_class}")

    job_id = uuid.uuid4().hex[:12]
    scene_path = SCENES_DIR / f"scene_{job_id}.py"
    output_dir = RESULTS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_path.write_text(req.scene_code)

    cmd = [
        "manim",
        f"-q{req.quality}",
        "--media_dir", str(output_dir),
        "--disable_caching",
        "--output_file", "output",
        str(scene_path),
        req.scene_class,
    ]

    t0 = time.time()
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=MAX_RENDER_SECONDS)
        except asyncio.TimeoutError:
            proc.kill()
            raise HTTPException(504, f"manim render exceeded {MAX_RENDER_SECONDS}s")
    except FileNotFoundError:
        raise HTTPException(500, "manim binary not found in PATH")

    render_time = time.time() - t0

    if proc.returncode != 0:
        err = (stderr or b"").decode("utf-8", errors="replace")[-2000:]
        raise HTTPException(500, f"manim failed (exit {proc.returncode}): {err}")

    # Manim drops the mp4 at: <media_dir>/videos/scene_<id>/<quality>/output.mp4
    mp4_candidates = list(output_dir.rglob("output.mp4"))
    if not mp4_candidates:
        raise HTTPException(500, "manim produced no output.mp4")
    mp4 = mp4_candidates[0]

    # Probe duration via ffprobe (best-effort, fail-safe).
    duration_s = None
    try:
        probe = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(mp4),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, _ = await probe.communicate()
        duration_s = float(out.decode().strip())
    except Exception:
        pass

    if req.response_format == "b64_json":
        b64 = base64.b64encode(mp4.read_bytes()).decode()
        return {"video_b64": b64, "duration_s": duration_s,
                "render_time_s": round(render_time, 2), "job_id": job_id}

    # url mode: serve via /results/<job>/output.mp4
    return {"video_url": f"/results/{job_id}/output.mp4",
            "duration_s": duration_s,
            "render_time_s": round(render_time, 2),
            "job_id": job_id}


@app.get("/results/{job_id}/{filename}")
async def serve_result(job_id: str, filename: str):
    if not _SAFE_CLASS_RE.match(job_id) or "/" in filename or ".." in filename:
        raise HTTPException(400, "invalid path")
    candidates = list((RESULTS_DIR / job_id).rglob(filename))
    if not candidates:
        raise HTTPException(404, "not found")
    return FileResponse(candidates[0], media_type="video/mp4")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
