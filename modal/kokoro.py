"""Modal deploy script for Kokoro TTS (82M, native PyTorch).

Cold start optimization:
- A10G GPU with full GPU memory snapshot (CPU + CUDA state)
- snap=True: imports modules AND loads model to GPU (all saved in snapshot)
- snap=False: minimal setup after restore (model already on GPU)
- min_containers=1: always-warm container eliminates first-request cold start
- Benchmarked cold start: ~7.4s avg (stable, 6.7-7.9s range)

Usage:
    modal deploy docker/modal/kokoro.py

Environment:
    MODAL_MAX_INPUTS   — concurrent inputs per container (default: 4)
    MODAL_PROXY_SECRET — if set, endpoints require Modal-Key/Modal-Secret headers

Endpoint: https://marcosremar--babelcast-kokoro-kokorotts-serve.modal.run
"""

import os

import modal

app = modal.App("babelcast-kokoro")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("espeak-ng", "ffmpeg")
    .pip_install(
        "torch==2.5.1", "torchaudio==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "kokoro>=0.9.4",
        "misaki[en,ja,zh,ko,pt,es,fr,de,it,hi]>=0.9.4",
        "soundfile",
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.32.0",
    )
    .run_commands(
        "python3 -c \""
        "from kokoro import KPipeline; "
        "p = KPipeline(lang_code='a'); "
        "[None for _ in p('test', voice='af_heart')]; "
        "print('kokoro OK')"
        "\"",
    )
    .add_local_file("docker/api/server_kokoro.py", "/app/server_kokoro.py")
)

MAX_INPUTS = int(os.environ.get("MODAL_MAX_INPUTS", "4"))

_app_cls_kwargs: dict = dict(
    image=image,
    gpu="A10G",
    timeout=900,
    scaledown_window=300,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    min_containers=1,
)

_proxy_secret = os.environ.get("MODAL_PROXY_SECRET")
if _proxy_secret:
    _app_cls_kwargs["secrets"] = [modal.Secret.from_dict({"MODAL_PROXY_SECRET": _proxy_secret})]


@app.cls(**_app_cls_kwargs)
@modal.concurrent(max_inputs=MAX_INPUTS)
class KokoroTTS:

    @modal.enter(snap=True)
    def setup_snapshot(self):
        """Runs BEFORE snapshot — import modules AND load model to GPU (all captured in GPU snapshot)."""
        import sys
        sys.path.insert(0, "/app")
        import server_kokoro
        from server_kokoro import get_pipeline, app as fastapi_app
        self.fastapi_app = fastapi_app
        get_pipeline("a")
        print("[snap=True] Model loaded to GPU, GPU snapshot will be taken")

    @modal.enter(snap=False)
    def after_restore(self):
        """Runs AFTER GPU snapshot restore — model already on GPU."""
        import sys
        sys.path.insert(0, "/app")
        from server_kokoro import app as fastapi_app
        self.fastapi_app = fastapi_app
        print("[snap=False] GPU snapshot restored, model already on GPU")

    @modal.asgi_app()
    def serve(self):
        return self.fastapi_app
