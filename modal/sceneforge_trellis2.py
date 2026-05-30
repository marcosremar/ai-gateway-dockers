"""Modal deploy script for TRELLIS-2 image-to-3D inference.

Cheapest viable Modal GPU: A10G ($1.10/h while running). Modal bills
per-second; when no requests arrive for `scaledown_window` seconds the
container is killed and billing stops. Idle cost = $0.

The marcosremar/trellis2:latest registry image already ships a FastAPI
server on port 8000 (`/generate` accepts multipart `file=` and returns a
binary GLB). We wrap it with `@modal.web_server` so Modal's HTTPS proxy
forwards traffic to whatever process binds port 8000 inside the
container.

Usage:
    modal deploy dockers/modal/trellis2.py

Env knobs (override at deploy time):
    MODAL_TRELLIS2_GPU       — Modal GPU type (default "a10g")
    MODAL_TRELLIS2_TIMEOUT   — per-request seconds (default 1800)
    MODAL_TRELLIS2_SCALEDOWN — idle-shutdown seconds (default 120)
    TRELLIS2_START_CMD       — server start command run inside the
                               container (default tries common patterns)
"""

import os
import subprocess
import time

import modal

# NOTE: app id is intentionally NOT "trellis2" — the marcosremar/trellis2
# container ships a Python package named `trellis2` at the top of sys.path.
# When Modal's runtime imports the deploy module by its filename
# (`trellis2.py` → module name `trellis2`), Python finds the in-image
# package first and returns it, which obviously has no `Trellis2` class.
# Symptom: AttributeError: module 'trellis2' has no attribute 'Trellis2'.
# Using a unique app id keeps the module file separate from anything the
# container ships and Modal's import resolver picks the right one.
app = modal.App("sceneforge-trellis2")

# `add_python="3.11"` was tried earlier but produced
# `ModuleNotFoundError: No module named 'torch'` in container logs —
# Modal overlays a fresh Python that can't see the image's torch
# install. The image already ships its own Python with torch + model
# weights, so let it through unchanged.
# Pin to digest so Modal pulls the latest CI build instead of reusing
# the cached layer from an earlier `:latest`. Bump this when ai-gateway-
# dockers ships a new image — `docker pull marcosremar/trellis2:latest
# && docker inspect marcosremar/trellis2:latest --format '{{ .RepoDigests }}'`
# or
#   curl -s 'https://hub.docker.com/v2/repositories/marcosremar/trellis2/tags/latest' | jq '.digest'
image = modal.Image.from_registry(
    "marcosremar/trellis2@sha256:e31938bda7ebb783e1a7f64937621a0ca18b23b4e86b003841dab53e2e5e68cc",
)

GPU = os.environ.get("MODAL_TRELLIS2_GPU", "a10g")
TIMEOUT = int(os.environ.get("MODAL_TRELLIS2_TIMEOUT", "1800"))
SCALEDOWN = int(os.environ.get("MODAL_TRELLIS2_SCALEDOWN", "120"))


# Class-based pattern mirrors `babelcast.py` — Modal's runtime resolves
# `app.Trellis2.web` reliably. The plain `@app.function + @modal.web_server`
# form deployed but failed at runtime with
#   AttributeError: module 'trellis2' has no attribute 'serve'
# during container startup, presumably because the function-form wrapper
# isn't surfaced where Modal's `import_single_function_service` looks.
# HF_TOKEN is required because the trellis2 server downloads
# `facebook/dinov3-vitl16-pretrain-lvd1689m` at boot, which is a `gated:
# manual` Hugging Face repo. Without it, /generate returns
#   {"detail":"generation failed: You are trying to access a gated repo..."}
# Operator must run once: `modal secret create huggingface HF_TOKEN=...`.
_secrets = []
try:
    _secrets.append(modal.Secret.from_name("huggingface"))
except Exception:
    # Surface the missing-secret case at deploy time with a clear message
    # instead of failing on the first /generate call from the container.
    print("[sceneforge-trellis2] WARN: Modal secret 'huggingface' not found. "
          "Create it once: modal secret create huggingface HF_TOKEN=hf_xxx")


@app.cls(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT,
    scaledown_window=SCALEDOWN,
    # min_containers=0 (default) — pay nothing when idle.
    secrets=_secrets,
)
class Trellis2:
    @modal.enter()
    def boot(self) -> None:
        """Start the trellis2 FastAPI server in the container.

        We try the marcosremar/trellis2 image's expected entry points in
        order. Override TRELLIS2_START_CMD at deploy time if the image
        ships its server somewhere else.
        """
        candidates = [
            os.environ.get("TRELLIS2_START_CMD"),
            "/app/start.sh",
            "/start.sh",
            "python /app/server.py",
            "python -m uvicorn server:app --host 0.0.0.0 --port 8000",
            "python -m uvicorn app.server:app --host 0.0.0.0 --port 8000",
        ]
        cmd = next((c for c in candidates if c), candidates[-1])
        print(f"[trellis2] starting server: {cmd}")
        subprocess.Popen(cmd, shell=True)
        # Brief grace period so the @web_server health check sees the port
        # already bound on the first probe instead of racing the boot.
        time.sleep(2)

    @modal.web_server(port=8000, startup_timeout=300)
    def web(self) -> None:
        """Modal HTTPS proxy forwards external requests to port 8000 inside
        the container. The actual server is started by `boot` above; this
        method only declares the port to expose."""
