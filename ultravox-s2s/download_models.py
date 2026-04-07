#!/usr/bin/env python3
"""Pre-download models at Docker build time for instant container boot.

Uses hf-xet (the new default since huggingface_hub 0.32) with pinned concurrency.
Benchmarked on Vast.ai RTX 4090 (2026-04-07): HF_XET_FIXED_DOWNLOAD_CONCURRENCY=50
gives 394 MB/s avg vs 319 MB/s for the legacy hf_transfer env var (+24%).
"""
import os
import sys

# hf-xet tuning — high performance mode + pinned concurrency
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_XET_FIXED_DOWNLOAD_CONCURRENCY", "50")

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
    max_workers=8,  # parallel file workers
)

print("Downloading Qwen3-TTS (Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)...", flush=True)
snapshot_download(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    cache_dir=cache_dir,
    token=token or None,
    max_workers=8,
)

print("Done — all models cached in image.", flush=True)
