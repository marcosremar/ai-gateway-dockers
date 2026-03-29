"""Modal deploy script for BabelCast inference server (full pipeline).

Uses GPU memory snapshots on A10G for ~5s cold starts.
Models are loaded directly in Python (no subprocess) so the GPU state
gets captured in the snapshot.

Usage:
    modal deploy docker/modal/babelcast.py
"""

import modal

app = modal.App("babelcast")

image = (
    modal.Image.from_registry("marcosremar/babelcast-translategemma:latest")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
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
        "snapshot_download('Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice'); "
        "snapshot_download('Qwen/Qwen3-TTS-12Hz-0.6B-Base'); "
        "print('Both TTS models cached')\"",
    )
    .add_local_dir("docker/api", remote_path="/app/api")
    .add_local_file("docker/start.sh", remote_path="/app/start.sh")
)


@app.cls(
    image=image,
    gpu="a10g",
    timeout=600,
    scaledown_window=300,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=4)
class BabelCast:
    @modal.enter(snap=True)
    def load_models(self):
        """Load all models into GPU memory BEFORE snapshot."""
        import subprocess, os, sys, time

        # Add api dir to path so server.py imports work
        sys.path.insert(0, "/app/api")
        os.chdir("/app/api")

        # Start llama.cpp server in background (LLM)
        os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/app/.cache/huggingface")
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        # Download and start LLM (llama.cpp)
        print("[snapshot] Downloading LLM GGUF...")
        from huggingface_hub import hf_hub_download
        gguf_path = hf_hub_download(
            "bullerwins/translategemma-12b-it-GGUF",
            filename="translategemma-12b-it-Q5_K_M.gguf",
        )
        print(f"[snapshot] GGUF: {gguf_path}")

        self.llama_proc = subprocess.Popen([
            sys.executable, "-m", "llama_cpp.server",
            "--host", "127.0.0.1", "--port", "8002",
            "--model", gguf_path, "--n_gpu_layers", "99", "--n_ctx", "2048",
        ], stdout=open("/tmp/llama.log", "w"), stderr=subprocess.STDOUT)

        # Load Whisper
        print("[snapshot] Loading Whisper...")
        from faster_whisper import WhisperModel
        self.whisper = WhisperModel("large-v3-turbo", device="cuda")
        print("[snapshot] Whisper loaded")

        # Load TTS (both models)
        print("[snapshot] Loading TTS models...")
        from services.tts import TTSService
        self.tts = TTSService("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "cuda:0")
        self.tts.load()
        print("[snapshot] TTS loaded (CustomVoice + Base)")

        # Warm up TTS (triggers CUDA kernel compilation)
        print("[snapshot] Warming up TTS...")
        try:
            self.tts.synthesize("Hello", "English", "Ryan")
            print("[snapshot] TTS warmup complete")
        except Exception as e:
            print(f"[snapshot] TTS warmup error: {e}")

        # Wait for llama.cpp
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

        # Import the FastAPI app (it uses get_tts/get_whisper dependency injection)
        from server import app as fastapi_app
        return fastapi_app
