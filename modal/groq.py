"""Modal deploy script for BabelCast Lightweight (STT + TTS only, no local LLM).

Cloud-only mode: Whisper STT + Qwen3-TTS on GPU, LLM routed to cloud via AI Gateway.
Smaller footprint, faster boot — no llama.cpp, no GGUF download.

Usage:
    modal deploy docker/modal/groq.py

Environment:
    MODAL_MAX_INPUTS   — concurrent inputs per container (default: 4)
    MODAL_PROXY_SECRET — if set, endpoints require Modal-Key/Modal-Secret headers
"""

import os

import modal

app = modal.App("babelcast-groq")

vol = modal.Volume.from_name("babelcast-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "sox", "libsox-dev", "curl")
    .pip_install(
        "torch==2.5.1", "torchaudio==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "faster-whisper>=1.1.0",
        "fastapi>=0.115.0", "uvicorn[standard]>=0.32.0",
        "python-multipart", "httpx", "soundfile", "numpy",
        "huggingface-hub>=1.0.0", "hf_xet>=1.4.0",
        "pydantic-settings>=2.0", "websockets",
    )
    .pip_install(
        "transformers==4.57.3", "accelerate>=1.12.0",
        "librosa", "einops", "onnxruntime", "sox",
    )
    .run_commands(
        "pip install --no-cache-dir qwen-tts>=0.1.1",
        "find / -path '*/qwen_tts/*.py' -exec sed -i 's/fix_mistral_regex=True,//' {} + 2>/dev/null; echo ok",
        "find / -name '*.pyc' -path '*/qwen_tts/*' -delete 2>/dev/null; echo ok",
    )
    .pip_install("speechbrain")
    .run_commands(
        "python3 -c \""
        "from huggingface_hub import snapshot_download; "
        "snapshot_download('Systran/faster-whisper-large-v3-turbo', cache_dir='/models'); "
        "snapshot_download('Qwen/Qwen3-TTS-12Hz-0.6B-Base', cache_dir='/models'); "
        "print('Models cached')\"",
        "python3 -c \""
        "from speechbrain.inference.speaker import EncoderClassifier; "
        "EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', run_opts={'device': 'cpu'}); "
        "print('ECAPA-TDNN cached')\"",
    )
    .add_local_dir("docker/api", remote_path="/app/api")
    .add_local_file("docker/start-groq.sh", remote_path="/app/start-groq.sh")
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
    volumes={"/models": vol},
)

_proxy_secret = os.environ.get("MODAL_PROXY_SECRET")
if _proxy_secret:
    _app_cls_kwargs["secrets"] = [modal.Secret.from_dict({"MODAL_PROXY_SECRET": _proxy_secret})]


@app.cls(**_app_cls_kwargs)
@modal.concurrent(max_inputs=MAX_INPUTS)
class BabelCastGroq:

    @modal.enter(snap=True)
    def setup_snapshot(self):
        """Load Whisper + TTS to GPU before snapshot (no LLM)."""
        import os, sys
        sys.path.insert(0, "/app/api")
        os.chdir("/app/api")
        os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/app/.cache/huggingface")

        print("[snapshot] Loading Whisper...")
        from faster_whisper import WhisperModel
        self.whisper = WhisperModel("large-v3-turbo", device="cuda")
        print("[snapshot] Whisper loaded")

        print("[snapshot] Loading TTS...")
        from services.tts import TTSService
        self.tts = TTSService("Qwen/Qwen3-TTS-12Hz-0.6B-Base", "cuda:0")
        self.tts.load()
        print("[snapshot] TTS loaded")

        try:
            self.tts.synthesize("Hello", "English", "Ryan")
            print("[snapshot] TTS warmup OK")
        except Exception as e:
            print(f"[snapshot] TTS warmup error: {e}")

        print("[snapshot] Loading speaker verification...")
        from speechbrain.inference.speaker import EncoderClassifier
        self.speaker_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
        )
        print("[snapshot] Speaker verification loaded")

        print("[snapshot] All models loaded — GPU snapshot will be taken")

    @modal.enter(snap=False)
    def after_restore(self):
        """GPU snapshot restored — Whisper + TTS already on GPU."""
        print("[restore] GPU snapshot restored")

    @modal.asgi_app()
    def serve(self):
        import sys
        sys.path.insert(0, "/app/api")
        from server import app as fastapi_app
        return fastapi_app
