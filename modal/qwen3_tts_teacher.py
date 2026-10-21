"""Modal deploy: Qwen3-TTS server with voice-clone endpoint, built from
local sources (no external git clone). Replaces the brittle babelcast-docker
sparse-checkout dependency that broke the standard `tts.py` template.

Endpoints exposed (matches `ai-gateway-dockers/qwen3-tts/server.py`):

  GET  /health
  GET  /v1/audio/voices
  POST /v1/audio/speech         JSON {input, voice, response_format}
  POST /v1/audio/speech/clone   multipart {text, reference_audio, ref_text,
                                           response_format, language?}

Usage:
  modal deploy modal/qwen3_tts_clone.py

The container starts cold (~3 min model download on first boot, ~30s after).
We keep `min_containers=1` so the gateway always finds a warm worker;
`scaledown_window=600` lets it settle after 10 idle minutes to control cost.

Required infra: Modal token configured (`modal token new`). The gateway
discovers the Modal-hosted endpoint via `modal app list` once deployed and
the URL is published in this file's `_modal_url` log line.
"""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent / "qwen3-tts"

app = modal.App("qwen3-tts-teacher-v2")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .run_commands(
        # Pinned versions match the ai-gateway-dockers/qwen3-tts/Dockerfile
        # — keep in sync if you bump there.
        "pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 "
        "--index-url https://download.pytorch.org/whl/cu124",
        "pip install --no-cache-dir transformers==4.57.3 accelerate==1.12.0 "
        "qwen-tts==0.1.1 faster-qwen3-tts==0.2.6 soundfile numpy librosa",
        "pip install --no-cache-dir 'fastapi>=0.115.0' "
        "'uvicorn[standard]>=0.32.0' python-multipart httpx pydantic",
        # Same upstream-bug shim as the Vast docker — qwen-tts 0.1.x ships a
        # stale `fix_mistral_regex` kwarg that current transformers rejects.
        "find / -path '*/qwen_tts/*.py' -exec sed -i "
        "'s/fix_mistral_regex=True,//' {} + 2>/dev/null; echo ok",
        "find / -name '*.pyc' -path '*/qwen_tts/*' -delete 2>/dev/null; echo ok",
        "python3 -c 'from qwen_tts import Qwen3TTSModel; print(\"qwen-tts OK\")'",
    )
    # Pre-download model weights into the image — cold-start drops from
    # ~3 min to <30s. Worth it for a TTS path that's hit once per chunk.
    .run_commands(
        "python3 -c \""
        "from huggingface_hub import snapshot_download; "
        "snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base', cache_dir='/models'); "
        "print('cached')\"",
    )
    .env({
        "QWEN3_TTS_MODEL": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        # Disable idle-watchdog: it os._exit(0)s after 15min idle (built for
        # RunPod/Vast pods) — on Modal that kills the container -> 404. Modal's
        # own scaledown_window handles lifecycle. This was the recurring-death bug.
        "IDLE_TIMEOUT_MIN": "0",
        # Auto-register a single cached voice on container boot — backend
        # then uses /v1/audio/speech/cached (JSON) instead of multipart
        # /v1/audio/speech/clone, skipping per-request speaker-embedding
        # extraction and the 576KB WAV upload.
        "QWEN3_TTS_AUTO_REGISTER_VOICE": "yt-clone",
        "QWEN3_TTS_AUTO_REGISTER_AUDIO": "/opt/voice-ref.wav",
        "QWEN3_TTS_AUTO_REGISTER_TEXT": "/opt/voice-ref.txt",
    })
    .add_local_file(str(ROOT / "server.py"), "/app/server.py")
    .add_local_file(str(ROOT / "idle_watchdog.py"), "/app/idle_watchdog.py")
    .add_local_file(str(ROOT / "voice-ref.wav"), "/opt/voice-ref.wav")
    .add_local_file(str(ROOT / "voice-ref.txt"), "/opt/voice-ref.txt")
)

@app.cls(
    image=image,
    gpu="L4",
    timeout=900,
    # Bumped 60 -> 600s: container stays warm 10 min between requests, so
    # back-to-back plays from the same blog reader hit a warm worker.
    scaledown_window=600,
    min_containers=2,
    max_containers=10,
    # CPU-only memory snapshot. Model loads to CPU during snap (GPU not
    # attached), snapshot captures CPU state. On cold restore: snapshot
    # deserializes (~2s), then snap=False enter moves weights to GPU
    # (~5-10s upload). Avoids GPU snapshot alpha overhead — the GPU memory
    # snapshot pathway was netting ~65s vs ~50s for AutoModel-loaded TTS,
    # because the alpha GPU snapshot serializes 4GB of CUDA state to disk.
    enable_memory_snapshot=False,
)
@modal.concurrent(max_inputs=4)
class TTSServer:
    """Qwen3-TTS server with vLLM-style CPU-snapshot cold-start acceleration.

    Loads model on CPU during snap=True (GPU unavailable then, so
    server._load_model auto-picks device_map='cpu'). Snapshot captures
    Python imports + CPU weights. On cold restore, snap=False runs with
    GPU attached and moves the model from CPU to CUDA before serving.
    """

    @modal.enter()
    def load(self):
        import os, sys
        os.environ.setdefault("HF_HOME", "/models")
        os.environ.setdefault("TRANSFORMERS_CACHE", "/models")
        if "/app" not in sys.path:
            sys.path.insert(0, "/app")
        os.chdir("/app")
        import server
        server._load_model()
        server._warmup()

    @modal.asgi_app()
    def web(self):
        import sys
        if "/app" not in sys.path:
            sys.path.insert(0, "/app")
        from server import app as fastapi_app
        return fastapi_app
