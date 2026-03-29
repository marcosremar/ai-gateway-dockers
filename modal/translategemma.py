"""Modal deploy script for BabelCast TranslateGemma (subtitles-only pipeline).

Pipeline: Whisper STT → TranslateGemma 12B (llama.cpp) → Qwen3-TTS
Groq handles LLM translation while TranslateGemma downloads in background,
then auto-promotes to local LLM when ready.

Usage:
    modal deploy docker/modal/translategemma.py

Requires: CONF_GROQ_API_KEY secret in Modal (for initial Groq fallback)
"""

import modal

app = modal.App("babelcast-translategemma")

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
        "print('TTS models cached')\"",
    )
    .run_commands(
        # Pre-download TranslateGemma GGUF (~7GB) into image layer
        "python3 -c \""
        "from huggingface_hub import hf_hub_download; "
        "hf_hub_download('bullerwins/translategemma-12b-it-GGUF', "
        "filename='translategemma-12b-it-Q5_K_M.gguf'); "
        "print('TranslateGemma GGUF cached')\"",
    )
    .add_local_dir("docker/api", remote_path="/app/api")
    .add_local_file("docker/start-babelcast-translategemma-only-subtitles.sh", remote_path="/app/start.sh")
)


@app.cls(
    image=image,
    gpu="A10G",
    timeout=900,
    scaledown_window=300,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=4)
class BabelCastTranslateGemma:

    @modal.enter(snap=True)
    def setup_snapshot(self):
        """Load all models to GPU before snapshot."""
        import subprocess, os, sys, time

        sys.path.insert(0, "/app/api")
        os.chdir("/app/api")
        os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/app/.cache/huggingface")
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        # Start llama.cpp with TranslateGemma
        print("[snapshot] Starting llama.cpp with TranslateGemma...")
        from huggingface_hub import hf_hub_download
        gguf_path = hf_hub_download(
            "bullerwins/translategemma-12b-it-GGUF",
            filename="translategemma-12b-it-Q5_K_M.gguf",
        )
        self.llama_proc = subprocess.Popen([
            sys.executable, "-m", "llama_cpp.server",
            "--host", "127.0.0.1", "--port", "8002",
            "--model", gguf_path, "--n_gpu_layers", "99", "--n_ctx", "2048",
        ], stdout=open("/tmp/llama.log", "w"), stderr=subprocess.STDOUT)

        # Load Whisper
        print("[snapshot] Loading Whisper...")
        from faster_whisper import WhisperModel
        self.whisper = WhisperModel("large-v3-turbo", device="cuda")

        # Load TTS
        print("[snapshot] Loading TTS...")
        from services.tts import TTSService
        self.tts = TTSService("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "cuda:0")
        self.tts.load()

        # Warm up TTS
        try:
            self.tts.synthesize("Hello", "English", "Ryan")
            print("[snapshot] TTS warmup OK")
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

        print("[snapshot] All models loaded — GPU snapshot will be taken")

    @modal.enter(snap=False)
    def after_restore(self):
        """GPU snapshot restored — all models already on GPU."""
        print("[restore] GPU snapshot restored")

    @modal.asgi_app()
    def serve(self):
        import sys
        sys.path.insert(0, "/app/api")
        from server import app as fastapi_app
        return fastapi_app
