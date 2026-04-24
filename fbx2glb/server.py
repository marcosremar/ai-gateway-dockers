"""
fbx2glb — FastAPI wrapper around the autoresearch-winning Blender
headless FBX→GLB converter (convert.sh).

Endpoints:

  GET  /health        Quick liveness check (reports Blender version)
  GET  /              Minimal HTML help page
  POST /v1/convert    Multipart FBX upload → GLB binary response

`/v1/convert` accepts `file` as a multipart field. The handler writes
the upload to a scratch directory, invokes `convert.sh <in.fbx> <out.glb>`
(which calls Blender headless + runs the POSITION rebase post-process),
then streams the resulting `.glb` back with Content-Type
`model/gltf-binary`. Each request gets its own tempdir so concurrent
uploads don't collide.

Env:
  IDLE_TIMEOUT_MIN      minutes of inactivity before self-exit (default 15, 0=off)
  MAX_FBX_BYTES         upload size cap (default 256 MB)
  CONVERT_TIMEOUT_SEC   per-conversion wall-clock cap (default 240)
"""

import asyncio
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
import uvicorn

from idle_watchdog import add_idle_middleware, start_watchdog, touch_activity

log = logging.getLogger("fbx2glb")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

APP_ROOT = Path(__file__).parent.resolve()
CONVERT_SH = APP_ROOT / "convert.sh"

MAX_FBX_BYTES = int(os.environ.get("MAX_FBX_BYTES", str(256 * 1024 * 1024)))
CONVERT_TIMEOUT_SEC = int(os.environ.get("CONVERT_TIMEOUT_SEC", "240"))


def blender_version() -> str:
    """Run `blender --version` once at boot for /health to surface."""
    try:
        out = subprocess.run(
            ["blender", "--version"], capture_output=True, text=True, timeout=10
        )
        first = (out.stdout or out.stderr).splitlines()[0] if out.stdout or out.stderr else ""
        return first.strip()
    except Exception as e:
        return f"blender --version failed: {e}"


BLENDER_VERSION = blender_version()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("fbx2glb starting. %s", BLENDER_VERSION)
    asyncio.create_task(start_watchdog())
    yield


app = FastAPI(
    title="fbx2glb",
    description="Blender 4.5 headless FBX→GLB converter with POSITION rebase post-process.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
add_idle_middleware(app)


INDEX_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>fbx2glb</title>
<style>body{font:14px/1.6 -apple-system,system-ui,sans-serif;max-width:640px;margin:40px auto;padding:0 20px;color:#222}
code{background:#f3f3f3;padding:1px 5px;border-radius:3px}pre{background:#f3f3f3;padding:12px;border-radius:6px;overflow:auto}</style>
</head><body><h1>fbx2glb</h1>
<p>Blender 4.5 headless FBX→GLB converter with a POSITION rebase + inverse-bind-matrix
compensation post-process (from the autoresearch/fbx2glb research loop).</p>
<h3>Usage</h3>
<pre>curl -fsS -X POST -F file=@mesh.fbx http://localhost:8000/v1/convert --output mesh.glb</pre>
<h3>Endpoints</h3><ul>
<li><code>GET /health</code> &mdash; liveness + Blender version</li>
<li><code>POST /v1/convert</code> &mdash; multipart <code>file</code> (FBX), returns <code>model/gltf-binary</code></li>
</ul></body></html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root() -> str:
    return INDEX_HTML


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "blender": BLENDER_VERSION}


@app.post("/v1/convert")
async def convert(request: Request, file: UploadFile = File(...)) -> Response:
    if not file.filename or not file.filename.lower().endswith(".fbx"):
        raise HTTPException(status_code=400, detail="upload must be named *.fbx")

    # Reserve a scratch dir per request — concurrent uploads must not
    # collide on input.fbx / output.glb.
    tmp = Path(tempfile.mkdtemp(prefix="fbx2glb-"))
    in_path = tmp / "input.fbx"
    out_path = tmp / "output.glb"

    try:
        # Stream upload to disk with a hard size cap.
        written = 0
        with open(in_path, "wb") as fh:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_FBX_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"FBX exceeds MAX_FBX_BYTES={MAX_FBX_BYTES}",
                    )
                fh.write(chunk)

        if written == 0:
            raise HTTPException(status_code=400, detail="empty upload")

        touch_activity()
        t0 = time.time()
        proc = subprocess.run(
            [str(CONVERT_SH), str(in_path), str(out_path)],
            capture_output=True,
            text=True,
            timeout=CONVERT_TIMEOUT_SEC,
        )
        elapsed = time.time() - t0
        if proc.returncode != 0 or not out_path.exists():
            log.warning("convert.sh failed rc=%s stderr=%s", proc.returncode, proc.stderr[:800])
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "conversion failed",
                    "returncode": proc.returncode,
                    "stderr": proc.stderr[-2000:],
                },
            )

        glb_bytes = out_path.read_bytes()
        log.info(
            "converted %s (%d B) -> %d B in %.2fs",
            file.filename, written, len(glb_bytes), elapsed,
        )

        out_name = file.filename.rsplit(".", 1)[0] + ".glb"
        return Response(
            content=glb_bytes,
            media_type="model/gltf-binary",
            headers={
                "Content-Disposition": f'attachment; filename="{out_name}"',
                "X-Convert-Elapsed-Ms": f"{int(elapsed * 1000)}",
                "X-Blender-Version": BLENDER_VERSION,
            },
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail=f"conversion exceeded {CONVERT_TIMEOUT_SEC}s wall-clock budget",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
