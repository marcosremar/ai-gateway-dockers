"""Modal deploy script for BabelCast Qwen3-TTS standalone server.

Usage:
    modal deploy docker/modal/tts.py

Environment:
    MODAL_MAX_INPUTS   — concurrent inputs per container (default: 4)
    MODAL_PROXY_SECRET — if set, endpoints require Modal-Key/Modal-Secret headers

MUST use clean debian_slim image. The babelcast-qwen3-tts base image
has stale packages that corrupt the audio codec output.
"""

import os

import modal

app = modal.App("babelcast-tts")

image = (
    modal.Image.debian_slim(python_version="3.11")
    # hf-xet is already the default in huggingface_hub >= 0.32 — no env vars
    # needed here. (Previously had HF_HUB_ENABLE_HF_TRANSFER=0 which is now a
    # deprecated no-op.)
    .run_commands(
        "pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124",
        "pip install transformers==4.57.3 accelerate==1.12.0 qwen-tts soundfile numpy",
        "pip install 'fastapi>=0.115.0' 'uvicorn[standard]>=0.32.0' python-multipart httpx librosa pydantic-settings",
        "find / -path '*/qwen_tts/*.py' -exec sed -i 's/fix_mistral_regex=True,//' {} + 2>/dev/null; echo 'fixed'",
        "find / -name '*.pyc' -path '*/qwen_tts/*' -delete 2>/dev/null",
        "python3 -c 'from qwen_tts import Qwen3TTSModel; print(\"OK\")' && echo 'v63-clean'",
    )
    .run_commands(
        "apt-get update -qq && apt-get install -y -qq git ffmpeg > /dev/null 2>&1",
        "mkdir -p /app/api/services && touch /app/api/__init__.py /app/api/services/__init__.py",
        "git clone --depth 1 --filter=blob:none --sparse https://github.com/marcosremar/babelcast-docker.git /tmp/_update 2>/dev/null"
        " && cd /tmp/_update && git sparse-checkout set docker/api 2>/dev/null"
        " && cp -v docker/api/services/tts.py /app/api/services/tts.py"
        " && cp -v docker/api/server_tts.py /app/api/server_tts.py"
        " && cp -v docker/api/config.py /app/api/config.py"
        " && cp -v docker/api/logger.py /app/api/logger.py"
        " && rm -rf /tmp/_update && echo 'v63-files'",
        "python3 -c 'import ast; ast.parse(open(\"/app/api/services/tts.py\").read()); print(\"tts OK\")'",
        "python3 -c 'import ast; ast.parse(open(\"/app/api/server_tts.py\").read()); print(\"server OK\")'",
        "echo 'v63'",
    )
)

MAX_INPUTS = int(os.environ.get("MODAL_MAX_INPUTS", "4"))

_func_kwargs: dict = dict(
    image=image,
    gpu="L40S",
    timeout=900,
    scaledown_window=300,
    min_containers=1,
)

_proxy_secret = os.environ.get("MODAL_PROXY_SECRET")
if _proxy_secret:
    _func_kwargs["secrets"] = [modal.Secret.from_dict({"MODAL_PROXY_SECRET": _proxy_secret})]


@app.function(**_func_kwargs)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.web_server(port=8000, startup_timeout=600)
def serve():
    import subprocess, os
    os.chdir("/app/api")
    subprocess.Popen(["uvicorn", "server_tts:app", "--host", "0.0.0.0", "--port", "8000"])
