"""
Cold-start optimization helpers shared across ML Docker images.

These are intentionally narrow utilities (no framework, no abstractions) that
solve specific bottlenecks measured in our own benchmarks. Drop into any
ML server's startup sequence to opt in.

Validated speedups (Vast.ai RTX 4090, 64GB RAM, 2026-04-08):
  - torch.compile cache reuse:    4.7-5.7x faster on second boot (5s saved)
  - fastsafetensors loader:       4.4x faster (CPU bench), 4.8-7.5x (GPU+GDS)
  - hf-xet HP + FIXED=50:         +24% download speed
  - Whisper pre-baked at build:   eliminates ~15s runtime download

Usage from a server.py:

    from coldstart import setup_torch_compile_cache, prefetch_safetensors

    # Call once at process start (before any torch.compile)
    setup_torch_compile_cache(cache_dir="/workspace/.torch-cache")

    # Optionally prefetch weights into page cache before transformers
    # opens them — converts random reads into sequential reads.
    prefetch_safetensors("fixie-ai/ultravox-v0_6-llama-3_1-8b")
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Iterable, Optional


# ── torch.compile cache ──────────────────────────────────────────────────────


def setup_torch_compile_cache(cache_dir: str = "/workspace/.torch-cache") -> None:
    """Configure persistent torch.compile cache.

    PyTorch's Inductor + AOTAutograd compile CUDA/Triton kernels on first call.
    These kernels can be cached to disk and reused — typically saves 3-15s on
    container restart for non-trivial models.

    Must be called BEFORE any `import torch` if possible, or before any
    `torch.compile()` call. Setting via os.environ.setdefault won't override
    existing values, so it's safe to call repeatedly.

    On Vast.ai (no persistent storage) the cache lives only for the lifetime
    of the container — still useful because subsequent inferences in the same
    process re-hit the cache.

    On RunPod with a network volume mounted at /workspace, the cache survives
    pod restarts → max benefit (one cold compile per VOLUME, not per CONTAINER).

    Best-effort: if the cache dir cannot be created (read-only fs, permission
    error), silently degrades to /tmp and never raises. The optimization is
    optional — never block server startup over it.
    """
    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        fallback = "/tmp/.torch-cache"
        print(
            f"[coldstart] cannot create {cache_dir} ({e}); using {fallback}",
            file=sys.stderr,
        )
        cache_dir = fallback
        try:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        except OSError as e2:
            print(
                f"[coldstart] cannot create fallback {fallback} either ({e2}); "
                f"torch.compile cache disabled",
                file=sys.stderr,
            )
            return  # Skip env var setup — no usable cache dir

    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", cache_dir)
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")
    # Reduce noise from compile in logs
    os.environ.setdefault("TORCHDYNAMO_VERBOSE", "0")
    print(f"[coldstart] torch.compile cache → {cache_dir}", file=sys.stderr)


# ── hf-xet env (production-validated tuning) ─────────────────────────────────


def setup_hf_xet_env() -> None:
    """Apply hf-xet tuning validated on Vast.ai RTX 4090 (3 runs, 2026-04-07).

    Benchmark:  legacy hf_transfer = 319 MB/s baseline
                HP + FIXED=50      = 394 MB/s avg (+24%, 11% spread)
                HP + FIXED=100     = 410 MB/s avg (+29%, 20% spread)

    HIGH_PERFORMANCE mode requires ≥64GB RAM. We auto-detect on Linux.
    """
    os.environ.setdefault("HF_XET_FIXED_DOWNLOAD_CONCURRENCY", "50")
    # Only enable HIGH_PERFORMANCE on hosts with enough RAM
    if _host_ram_mb() >= 60000:
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")


def _host_ram_mb() -> int:
    """Return host RAM in MB. Returns 0 if unable to detect (safe fallback)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError):
        pass
    return 0


# ── safetensors prefetch (warm OS page cache) ────────────────────────────────


def prefetch_safetensors(
    repo_or_paths,  # str repo_id OR list[Path]
    *,
    revision: str = "main",
    allow_patterns: Optional[Iterable[str]] = ("*.safetensors",),
) -> float:
    """Prefetch safetensors files into the OS page cache via sequential read.

    The first time `transformers.AutoModel.from_pretrained()` opens a
    safetensors file, it does many small random reads (one per tensor metadata
    + per shard). On a cold disk this is much slower than a single sequential
    read of the whole file.

    By streaming the file once into /dev/null FIRST, the OS page cache holds
    everything, and the subsequent transformers load becomes purely RAM-speed.

    Returns the elapsed prefetch time in seconds. Best-effort — never raises.
    """
    start = time.time()
    files: list[Path] = []

    if isinstance(repo_or_paths, str):
        # Repo id — resolve files via huggingface_hub cache
        try:
            from huggingface_hub import snapshot_download

            local = snapshot_download(
                repo_or_paths,
                revision=revision,
                allow_patterns=list(allow_patterns) if allow_patterns else None,
            )
            for root, _, fnames in os.walk(local):
                for f in fnames:
                    if f.endswith(".safetensors"):
                        files.append(Path(root) / f)
        except Exception as e:
            print(f"[coldstart] prefetch resolve failed: {e}", file=sys.stderr)
            return 0.0
    else:
        files = [Path(p) for p in repo_or_paths]

    if not files:
        return 0.0

    total_bytes = 0
    for f in files:
        try:
            # Sequential read in 4MB blocks → page-cached
            size = f.stat().st_size
            with open(f, "rb", buffering=0) as fp:
                while True:
                    chunk = fp.read(4 * 1024 * 1024)
                    if not chunk:
                        break
            total_bytes += size
        except OSError as e:
            print(f"[coldstart] prefetch skip {f}: {e}", file=sys.stderr)

    elapsed = time.time() - start
    mb = total_bytes / 1024 / 1024
    if mb > 0:
        print(
            f"[coldstart] prefetched {mb:.0f} MB across {len(files)} file(s) "
            f"in {elapsed:.1f}s ({mb / elapsed:.0f} MB/s)",
            file=sys.stderr,
        )
    return elapsed


# ── fastsafetensors loader (optional, for raw weight loading) ────────────────


def load_with_fastsafetensors(files, device: str = "cuda", nogds: bool = False):
    """Load safetensors files via fastsafetensors → device tensors dict.

    Returns: dict[name, torch.Tensor] on `device`.

    Use this for advanced flows where you construct the model from a state_dict
    yourself (skipping transformers' high-level loading). Most users should
    just call `prefetch_safetensors()` and let transformers do its thing —
    the page-cache warmup gets ~80% of the speedup with zero refactoring.

    Falls back to standard safetensors.load_file() if fastsafetensors is not
    installed (ImportError) or fails.
    """
    try:
        from fastsafetensors import SafeTensorsFileLoader, SingleGroup
    except ImportError:
        print(
            "[coldstart] fastsafetensors not installed — falling back to safetensors",
            file=sys.stderr,
        )
        return _load_with_safetensors_fallback(files, device)

    files = [str(p) for p in files]
    try:
        loader = SafeTensorsFileLoader(
            pg=SingleGroup(),
            device=device,
            nogds=nogds,
            framework="pytorch",
            debug_log=False,
        )
        loader.add_filenames({0: files})
        bufs = loader.copy_files_to_device()
        keys = loader.get_keys()
        tensors = {k: bufs.get_tensor(k) for k in keys}
        loader.close()
        return tensors
    except Exception as e:
        print(
            f"[coldstart] fastsafetensors load failed: {e} — falling back",
            file=sys.stderr,
        )
        return _load_with_safetensors_fallback(files, device)


def _load_with_safetensors_fallback(files, device: str):
    from safetensors.torch import load_file
    out = {}
    for f in files:
        out.update(load_file(str(f), device=device))
    return out


# ── Bootstrap (one-call init) ────────────────────────────────────────────────


def bootstrap(
    *,
    torch_cache_dir: str = "/workspace/.torch-cache",
    enable_hf_xet: bool = True,
) -> None:
    """One-call init for the common cold-start optimizations.

    Call this from your server.py BEFORE importing torch / transformers:

        from coldstart import bootstrap
        bootstrap()
        import torch  # now picks up the cache env vars

    Best-effort: any individual setup that fails is logged and skipped.
    Never raises — the server must start even if every optimization fails.
    """
    try:
        setup_torch_compile_cache(torch_cache_dir)
    except Exception as e:
        print(f"[coldstart] torch cache setup failed (non-fatal): {e}", file=sys.stderr)
    if enable_hf_xet:
        try:
            setup_hf_xet_env()
        except Exception as e:
            print(f"[coldstart] hf-xet setup failed (non-fatal): {e}", file=sys.stderr)


if __name__ == "__main__":
    # Smoke test
    print("=== coldstart.py smoke test ===")
    setup_torch_compile_cache("/tmp/coldstart-smoke")
    setup_hf_xet_env()
    print(f"  RAM detected: {_host_ram_mb()} MB")
    print(f"  HF_XET_HIGH_PERFORMANCE = {os.environ.get('HF_XET_HIGH_PERFORMANCE', 'unset')}")
    print(f"  HF_XET_FIXED_DOWNLOAD_CONCURRENCY = {os.environ.get('HF_XET_FIXED_DOWNLOAD_CONCURRENCY', 'unset')}")
    print(f"  TORCHINDUCTOR_CACHE_DIR = {os.environ.get('TORCHINDUCTOR_CACHE_DIR', 'unset')}")
    print("OK")
