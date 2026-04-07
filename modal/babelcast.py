"""Modal deploy script for BabelCast inference server (full pipeline).

Uses GPU memory snapshots on A10G for ~5s cold starts.
Models are loaded directly in Python (no subprocess) so the GPU state
gets captured in the snapshot.

Usage:
    modal deploy docker/modal/babelcast.py

Environment:
    MODAL_MAX_INPUTS   — concurrent inputs per container (default: 4)
    MODAL_PROXY_SECRET — if set, endpoints require Modal-Key/Modal-Secret headers
"""

import os

import modal

app = modal.App("babelcast")

vol = modal.Volume.from_name("babelcast-models", create_if_missing=True)

image = (
    modal.Image.from_registry("marcosremar/babelcast-translategemma:latest")
    .env({
        # hf-xet (new default) — benchmarked at +24% vs deprecated hf_transfer
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_XET_FIXED_DOWNLOAD_CONCURRENCY": "50",
    })
    .run_commands(
        "pip uninstall -y faster-qwen3-tts 2>/dev/null; echo ok",
        "pip install --no-cache-dir transformers==4.57.3 accelerate>=1.12.0 qwen-tts soundfile numpy librosa einops onnxruntime",
        "find / -path '*/qwen_tts/*.py' -exec sed -i 's/fix_mistral_regex=True,//' {} + 2>/dev/null; echo ok",
        "find / -name '*.pyc' -path '*/qwen_tts/*' -delete 2>/dev/null; echo ok",
        "python3 -c 'from qwen_tts import Qwen3TTSModel; print(\"qwen-tts OK\")'",
    )
    .run_commands(
        "python3 -c \""
        "from huggingface_hub import snapshot_download; "
        "snapshot_download('Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice', cache_dir='/models'); "
        "snapshot_download('Qwen/Qwen3-TTS-12Hz-0.6B-Base', cache_dir='/models'); "
        "print('Both TTS models cached')\"",
    )
    .add_local_dir("docker/api", remote_path="/app/api")
    .add_local_file("docker/start.sh", remote_path="/app/start.sh")
)

MAX_INPUTS = int(os.environ.get("MODAL_MAX_INPUTS", "4"))

_app_cls_kwargs: dict = dict(
    image=image,
    gpu="a10g",
    timeout=600,
    scaledown_window=300,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    min_containers=1,
    volumes={"/models": vol},
)

_proxy_secret = os.environ.get("MODAL_PROXY_SECRET")
if _proxy_secret:
    _app_cls_kwargs["secrets"] = [modal.Secret.from_dict({"MODAL_PROXY_SECRET": _proxy_secret})]


@app.cls(**_app_cls_kwargs)
@modal.concurrent(max_inputs=MAX_INPUTS)
class BabelCast:
    @modal.enter(snap=True)
    def load_models(self):
        """Load all models into GPU memory BEFORE snapshot."""
        import subprocess, sys, time

        sys.path.insert(0, "/app/api")

        print("[snapshot] Downloading LLM GGUF...")
        from huggingface_hub import hf_hub_download
        gguf_path = hf_hub_download(
            "bullerwins/translategemma-12b-it-GGUF",
            filename="translategemma-12b-it-Q5_K_M.gguf",
            cache_dir="/models",
        )
        print(f"[snapshot] GGUF: {gguf_path}")

        self.llama_proc = subprocess.Popen([
            sys.executable, "-m", "llama_cpp.server",
            "--host", "127.0.0.1", "--port", "8002",
            "--model", gguf_path, "--n_gpu_layers", "99", "--n_ctx", "2048",
        ], stdout=open("/tmp/llama.log", "w"), stderr=subprocess.STDOUT)

        print("[snapshot] Loading Whisper...")
        from faster_whisper import WhisperModel
        self.whisper = WhisperModel("large-v3-turbo", device="cuda")
        print("[snapshot] Whisper loaded")

        print("[snapshot] Loading TTS models...")
        from services.tts import TTSService
        self.tts = TTSService("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "cuda:0")
        self.tts.load()
        print("[snapshot] TTS loaded (CustomVoice + Base)")

        print("[snapshot] Warming up TTS...")
        try:
            self.tts.synthesize("Hello", "English", "Ryan")
            print("[snapshot] TTS warmup complete")
        except Exception as e:
            print(f"[snapshot] TTS warmup error: {e}")

        import urllib.request
        for i in range(60):
            try:
                urllib.request.urlopen("http://127.0.0.1:8002/v1/models", timeout=2)
                print(f"[snapshot] llama.cpp ready ({i*2}s)")
                break
            except Exception:
                time.sleep(2)

        print("[snapshot] All models loaded — GPU memory will be snapshotted")

    @modal.enter(snap=False)
    def after_restore(self):
        """Runs after snapshot restore."""
        print("[restore] GPU snapshot restored — all models ready")

    @modal.asgi_app()
    def serve(self):
        """Return the FastAPI app — models already loaded from snapshot."""
        import sys
        sys.path.insert(0, "/app/api")
        from server import app as fastapi_app
        return fastapi_app
