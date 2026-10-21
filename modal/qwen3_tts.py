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

app = modal.App("canal-dark-qwen3-tts")

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
    min_containers=1,
    # CPU-only memory snapshot. Model loads to CPU during snap (GPU not
    # attached), snapshot captures CPU state. On cold restore: snapshot
    # deserializes (~2s), then snap=False enter moves weights to GPU
    # (~5-10s upload). Avoids GPU snapshot alpha overhead — the GPU memory
    # snapshot pathway was netting ~65s vs ~50s for AutoModel-loaded TTS,
    # because the alpha GPU snapshot serializes 4GB of CUDA state to disk.
    enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=4)
class TTSServer:
    """Qwen3-TTS server with vLLM-style CPU-snapshot cold-start acceleration.

    Loads model on CPU during snap=True (GPU unavailable then, so
    server._load_model auto-picks device_map='cpu'). Snapshot captures
    Python imports + CPU weights. On cold restore, snap=False runs with
    GPU attached and moves the model from CPU to CUDA before serving.
    """

    @modal.enter(snap=True)
    def load_cpu(self):
        import os, sys
        os.environ.setdefault("HF_HOME", "/models")
        os.environ.setdefault("TRANSFORMERS_CACHE", "/models")
        if "/app" not in sys.path:
            sys.path.insert(0, "/app")
        os.chdir("/app")
        import server
        # GPU is not available during snap=True (no enable_gpu_snapshot).
        # _load_model picks device_map="cpu" and attempts mdl.to("cuda")
        # afterwards — that .to call will silently fail/warn (no cuda).
        # Result: model object lives on CPU, ready for snapshot.
        server._load_model()

    @modal.enter(snap=False)
    def move_to_gpu(self):
        # Post-snapshot restore: GPU now attached. Push CPU weights to cuda
        # and run warmup on GPU.
        import sys, torch
        if "/app" not in sys.path:
            sys.path.insert(0, "/app")
        import server
        if server.model is None:
            # Snapshot may not have captured globals — fall back to fresh load.
            server._load_model()
        elif torch.cuda.is_available():
            try:
                mdl = server.model
                to = getattr(mdl, "to", None)
                if callable(to):
                    to("cuda")
            except Exception as e:
                import logging
                logging.getLogger("qwen3-tts.modal").warning(
                    f"[tts.move-to-gpu.fail] {e!r} — falling back to fresh GPU load"
                )
                server.model = None
                server._load_model()
        # Light warmup on GPU so the first user request doesn't pay JIT cost.
        server._warmup()

    @modal.asgi_app()
    def web(self):
        import sys
        if "/app" not in sys.path:
            sys.path.insert(0, "/app")
        from server import app as fastapi_app
        return fastapi_app
