"""Modal deploy: FLUX.1-Kontext-dev — character-consistent image gen.

Black Forest Labs 2025 release. Takes a reference image + prompt,
preserves identity/setting across generations. Used by canal-dark to
hold "Zeno of Citium" identical across 15 narrative scenes.

OpenAI-ish endpoint:
  POST /v1/images/edits  multipart {image, prompt, size?, seed?}
                         returns JSON {data: [{b64_json: ...}], elapsed_s}

Deploy:
  modal deploy modal/flux_kontext.py

Repo: https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
"""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent / "flux-kontext"

app = modal.App("canal-dark-flux-kontext")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .run_commands(
        "pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 "
        "--index-url https://download.pytorch.org/whl/cu124",
        # diffusers 0.34+ has FluxKontextPipeline. transformers 4.56.2
        # matches the version we use elsewhere in canal-dark.
        "pip install --no-cache-dir diffusers==0.35.1 transformers==4.56.2 "
        "accelerate==1.2.1 sentencepiece protobuf",
        "pip install --no-cache-dir 'fastapi>=0.115.0' "
        "'uvicorn[standard]>=0.32.0' python-multipart Pillow pydantic",
    )
    # Pre-cache FLUX.1-Kontext-dev (~24GB). Gated — needs HF token with
    # access (visit huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
    # and click "Agree and access").
    .run_commands(
        "python3 -c \""
        "import os; "
        "from huggingface_hub import snapshot_download; "
        "snapshot_download('black-forest-labs/FLUX.1-Kontext-dev', "
        "  cache_dir='/models', token=os.environ['HF_TOKEN']); "
        "print('cached')\"",
        secrets=[modal.Secret.from_name("huggingface-canal-dark")],
    )
    .add_local_file(str(ROOT / "server.py"), "/app/server.py")
)

_func_kwargs = dict(
    image=image,
    # L40S 48GB — Kontext-dev (~24GB bf16) fits with margin. A100-80GB
    # ($3.71/h) cuts inference 1.5× but L40S ($1.95/h) is the right
    # cost/quality balance at canal-dark's volume.
    gpu="L40S",
    timeout=900,
    scaledown_window=60,
    min_containers=0,
    secrets=[modal.Secret.from_name("huggingface-canal-dark")],
)


@app.function(**_func_kwargs)
@modal.concurrent(max_inputs=1)
@modal.web_server(port=8000, startup_timeout=600)
def serve():
    import os
    import subprocess

    os.environ.setdefault("HF_HOME", "/models")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/models")
    os.chdir("/app")
    subprocess.Popen(
        ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
    )
