#!/usr/bin/env python3
"""Pre-download models in parallel for instant container boot.

Runs at Docker BUILD time (pre-bake) and as a safety net at BOOT time
(if the cache is empty — network volumes, lazy-load images, cold cache).

Parallelisation (cold-start plan A6):
    Previous version downloaded Ultravox then Qwen3-TTS sequentially. With
    two independent HF repos, the two downloads share ~no state — a
    ThreadPoolExecutor(max_workers=len(REPOS)) overlaps their I/O waits,
    cutting total wall time roughly in half when both repos are fresh.

    hf_xet already parallelises WITHIN a single snapshot_download (see
    HF_XET_FIXED_DOWNLOAD_CONCURRENCY=50); the win here is also overlapping
    BETWEEN the two repos so one doesn't finish before the other even
    starts to DNS-resolve.

Why threads (not asyncio)?
    snapshot_download is a blocking call that drops the GIL for its
    socket reads. asyncio wrapping would add complexity for no speedup.
    ThreadPoolExecutor with one worker per repo is the simplest form.

Uses hf-xet (the new default since huggingface_hub 0.32) with pinned concurrency.
Benchmarked on Vast.ai RTX 4090 (2026-04-07): HF_XET_FIXED_DOWNLOAD_CONCURRENCY=50
gives 394 MB/s avg vs 319 MB/s for the legacy hf_transfer env var (+24%).

IMPORTANT: snapshot_download on Whisper (or any encoder-decoder model) is
SAFE — do NOT use fastsafetensors multi-shard here; that's a Phase B
concern and is not what snapshot_download does anyway.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# hf-xet tuning — high performance mode + pinned concurrency
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_XET_FIXED_DOWNLOAD_CONCURRENCY", "50")

token = os.environ.get("HF_TOKEN", "")
if not token:
    print("WARNING: HF_TOKEN not set — gated models may fail", flush=True)

from huggingface_hub import snapshot_download

CACHE_DIR = "/app/.cache/huggingface/hub"

# (repo_id, stage_label) tuples — stage label is purely for log clarity.
REPOS = [
    ("fixie-ai/ultravox-v0_6-llama-3_1-8b", "LLM (Ultravox)"),
    ("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "TTS (Qwen3-TTS)"),
]


def _download_one(repo_id: str, label: str) -> tuple[str, float, str | None]:
    t0 = time.monotonic()
    try:
        print(f"[{label}] Downloading {repo_id}...", flush=True)
        snapshot_download(
            repo_id,
            cache_dir=CACHE_DIR,
            token=token or None,
            max_workers=8,  # parallel file workers WITHIN a repo
        )
        elapsed = time.monotonic() - t0
        print(f"[{label}] Done in {elapsed:.1f}s", flush=True)
        return repo_id, elapsed, None
    except Exception as exc:  # noqa: BLE001
        elapsed = time.monotonic() - t0
        print(f"[{label}] FAILED after {elapsed:.1f}s: {exc}", flush=True)
        return repo_id, elapsed, str(exc)


def main() -> int:
    overall_start = time.monotonic()
    errors: list[str] = []
    # One thread per repo — overlap their I/O waits.
    with ThreadPoolExecutor(max_workers=len(REPOS), thread_name_prefix="hf-dl") as pool:
        futures = {pool.submit(_download_one, repo, label): (repo, label) for repo, label in REPOS}
        for fut in as_completed(futures):
            _, _, err = fut.result()
            if err:
                errors.append(err)
    total = time.monotonic() - overall_start
    if errors:
        print(f"Completed with {len(errors)} failure(s) in {total:.1f}s — see logs above.", flush=True)
        return 1
    print(f"Done — all models cached in parallel in {total:.1f}s.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
