"""Modal deploy: FLUX.1-schnell (Apache 2.0) for canal-dark image gen.

Mirrors `qwen3_tts_clone.py` shape — local source mount, no babelcast-docker
clone, idle watchdog included. Endpoint:

  POST /v1/images/generations  JSON {prompt, size?, response_format=b64_json|url}

Cost on Modal L40S: ~$0.04-0.08 per 1024x1024 image (schnell 4-step).

Deploy:
  modal deploy modal/flux_canal_dark.py

Hits the same OpenAI-compatible shape canal-dark already calls via
`pipeline/services/image_flux.py`, so no client changes — point
FLUX_BASE_URL at the Modal endpoint and the existing `--render` flow
fills empty scene photo_urls.
"""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent / "flux"

app = modal.App("canal-dark-flux-schnell")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .run_commands(
        "pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 "
        "--index-url https://download.pytorch.org/whl/cu124",
        "pip install --no-cache-dir diffusers==0.32.2 transformers==4.47.1 "
        "accelerate==1.2.1 sentencepiece protobuf",
        "pip install --no-cache-dir 'fastapi>=0.115.0' "
        "'uvicorn[standard]>=0.32.0' python-multipart httpx pydantic Pillow",
    )
    # Pre-cache FLUX.1-schnell weights (~10GB) so cold-start is fast.
    # FLUX.1-schnell is gated even though Apache 2.0 — needs HF token
    # with access (visit huggingface.co/black-forest-labs/FLUX.1-schnell
    # and click "Agree and access").
    .run_commands(
        "python3 -c \""
        "import os; "
        "from huggingface_hub import snapshot_download; "
        "snapshot_download('black-forest-labs/FLUX.1-schnell', "
        "  cache_dir='/models', token=os.environ['HF_TOKEN']); "
        "print('cached')\"",
        secrets=[modal.Secret.from_name("huggingface-canal-dark")],
    )
    .add_local_file(str(ROOT / "server.py"), "/app/server.py")
    .add_local_file(str(ROOT / "idle_watchdog.py"), "/app/idle_watchdog.py")
)

_func_kwargs = dict(
    image=image,
    gpu="L40S",  # 48GB — schnell 4-step ~3s/img, fits bf16 with margin
    timeout=900,
    scaledown_window=60,    # die 60s after last request
    min_containers=0,       # pay-per-use only
    secrets=[modal.Secret.from_name("huggingface-canal-dark")],
)


@app.function(**_func_kwargs)
@modal.concurrent(max_inputs=2)
@modal.web_server(port=8000, startup_timeout=600)
def serve():
    import os
    import subprocess

    os.environ.setdefault("HF_HOME", "/models")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/models")
    os.environ.setdefault("FLUX_OFFLOAD", "0")  # 10GB fits L40S 48GB no offload
    os.chdir("/app")
    subprocess.Popen(
        ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
    )
