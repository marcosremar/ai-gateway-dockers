#!/usr/bin/env python3
"""Pre-download models at Docker build time for instant container boot."""
import os
import sys

token = os.environ.get("HF_TOKEN", "")
if not token:
    print("WARNING: HF_TOKEN not set — gated models may fail", flush=True)

from huggingface_hub import snapshot_download

cache_dir = "/app/.cache/huggingface/hub"

print("Downloading Ultravox (fixie-ai/ultravox-v0_6-llama-3_1-8b)...", flush=True)
snapshot_download(
    "fixie-ai/ultravox-v0_6-llama-3_1-8b",
    cache_dir=cache_dir,
    token=token or None,
)

print("Downloading Qwen3-TTS (Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)...", flush=True)
snapshot_download(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    cache_dir=cache_dir,
    token=token or None,
)

print("Done — all models cached in image.", flush=True)
