"""Modal deploy: Qwen3-TTS via nano-qwen3tts-vllm (3.25x faster).

Replaces the slow transformers AutoModel pathway in `qwen3_tts.py`.
Uses tsdocode/nano-qwen3tts-vllm — vLLM-style optimizations achieving
RTF ~0.4 on H100 (audio gen 2.8x faster than realtime) vs RTF ~1.5 on
the original transformers loader.

Constraints
-----------
- Compute capability >= 8.0 required → T4 (7.5) does NOT work.
  Cheapest viable Modal GPU = L4 ($0.80/h, cc 8.9, 24 GB).

Voice cloning
-------------
Server expects pre-enrolled voices as `examples/voices/<name>.pkl`.
We bake one voice (`yt-clone`) at image-build time from the reference
WAV + transcript packaged with this repo. Per-request reference audio
is NOT needed — client just passes `speaker: "yt-clone"`.

Endpoints (server)
------------------
GET  /health
GET  /voices
POST /v1/audio/speech   { text, language, speaker, volume? } -> audio/L16

Usage
-----
    modal deploy modal/qwen3_tts_vllm.py
"""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
QWEN_DIR = ROOT / "qwen3-tts"
REF_AUDIO = QWEN_DIR / "voice-ref.wav"
REF_TEXT = QWEN_DIR / "voice-ref.txt"

app = modal.App("canal-dark-qwen3-tts-vllm")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("ffmpeg", "git", "libsndfile1")
    .run_commands(
        "pip install --upgrade pip setuptools wheel",
        # PyTorch 2.10+ with CUDA per nano-qwen3tts-vllm requirements.
        "pip install --no-cache-dir torch==2.10.0 torchaudio==2.10.0 "
        "--index-url https://download.pytorch.org/whl/cu128",
        "pip install --no-cache-dir 'fastapi>=0.115.0' 'uvicorn[standard]>=0.32.0' "
        "soundfile librosa numpy huggingface-hub packaging ninja",
        # flash-attn: prebuilt wheel matching cu128 + torch 2.10 + cp311.
        # From mjun0812/flash-attention-prebuild-wheels v0.9.0 — avoids
        # 30+min source compile inside Modal.
        "pip install --no-cache-dir "
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/"
        "flash_attn-2.8.3+cu128torch2.10-cp311-cp311-linux_x86_64.whl",
        # Install nano-qwen3tts-vllm from source.
        "git clone --depth 1 https://github.com/tsdocode/nano-qwen3tts-vllm.git "
        "/opt/nano-qwen3tts-vllm",
        "cd /opt/nano-qwen3tts-vllm && pip install --no-cache-dir -e .",
    )
    # Pre-download model weights into image so cold start skips HF fetch.
    .run_commands(
        "python3 -c \""
        "from huggingface_hub import snapshot_download; "
        "snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base', cache_dir='/models'); "
        "print('cached')\"",
    )
)

# Bundle voice ref into image; enrollment runs at runtime (Qwen3TTSInterface
# requires CUDA at construction time and Modal image builds run on CPU-only).
if REF_AUDIO.exists() and REF_TEXT.exists():
    image = (
        image
        .add_local_file(str(REF_AUDIO), "/opt/voice-ref.wav", copy=True)
        .add_local_file(str(REF_TEXT), "/opt/voice-ref.txt", copy=True)
    )


@app.cls(
    image=image,
    gpu="L4",  # cc 8.9 — required by nano-qwen3tts-vllm (>=8.0)
    timeout=900,
    scaledown_window=600,
    min_containers=0,
    enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=2)
class TTSServer:
    """Boots nano-qwen3tts-vllm FastAPI server, enrolls our YouTube voice."""

    @modal.enter(snap=True)
    def load_cpu(self):
        """Pre-download / preload Python modules so snapshot captures them."""
        import os
        os.environ.setdefault("HF_HOME", "/models")
        os.environ.setdefault("TRANSFORMERS_CACHE", "/models")
        os.environ.setdefault("HF_HUB_CACHE", "/models")
        # Touch heavy imports so they're in the snapshot. Don't load model
        # to GPU — GPU not available in snap=True without enable_gpu_snapshot.
        import torch  # noqa: F401
        import soundfile  # noqa: F401
        import librosa  # noqa: F401
        import numpy  # noqa: F401
        from fastapi import FastAPI  # noqa: F401

    @modal.web_server(port=8000, startup_timeout=900)
    def web(self):
        """Enroll voice if missing (one-shot per container) then spawn uvicorn."""
        import gc
        import os
        import pickle
        import subprocess
        import sys

        os.environ.setdefault("HF_HOME", "/models")
        os.environ.setdefault("TRANSFORMERS_CACHE", "/models")
        os.environ.setdefault("HF_HUB_CACHE", "/models")

        voices_dir = "/opt/nano-qwen3tts-vllm/examples/voices"
        os.makedirs(voices_dir, exist_ok=True)
        voice_pkl = f"{voices_dir}/yt-clone.pkl"

        if not os.path.exists(voice_pkl):
            import torch
            import soundfile as sf
            from nano_qwen3tts_vllm.interface import Qwen3TTSInterface

            audio, sr = sf.read("/opt/voice-ref.wav", dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            ref_text = open("/opt/voice-ref.txt").read().strip()

            # Lower mem util so uvicorn-spawned interface has room left.
            iface = Qwen3TTSInterface.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                cache_dir="/models",
                gpu_memory_utilization=0.45,
            )
            prompt = iface.create_voice_clone_prompt(
                ref_audio=(audio, sr),
                ref_text=ref_text,
                x_vector_only_mode=False,
            )
            with open(voice_pkl, "wb") as fh:
                pickle.dump(prompt, fh)
            del iface, prompt
            gc.collect()
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass

        log = open("/tmp/server.log", "ab", buffering=0)
        subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "examples.server:app",
             "--host", "0.0.0.0", "--port", "8000", "--workers", "1"],
            stdout=log,
            stderr=log,
            cwd="/opt/nano-qwen3tts-vllm",
        )
