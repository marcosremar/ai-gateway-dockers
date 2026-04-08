"""
Cold-start optimization helpers shared across ML Docker images.

These are intentionally narrow utilities (no framework, no abstractions) that
solve specific bottlenecks measured in our own benchmarks. Drop into any
ML server's startup sequence to opt in.

Validated speedups (Vast.ai RTX 4090, 251GB RAM, real GPU bench 2026-04-08):
  - hf-xet HP + FIXED=50:         +24% download speed (proven in earlier bench)
  - Whisper pre-baked at build:   eliminates ~15s runtime download
  - torch.compile cache reuse:    1.06x (~0.3s saved) — much smaller than my
                                  Mac CPU test predicted (had been 4.7x). On
                                  CUDA, Inductor codegen is lighter, cache
                                  size is tiny (~8 KB), benefit is marginal.
  - fastsafetensors load:         1.74x for Ultravox 1.3GB (0.18s saved)
                                  ❌ FAILS on multi-shard models with
                                     repeated tensor keys (Whisper has the
                                     same `embed_positions.weight` in
                                     encoder + decoder shards)
                                  ❌ NO benefit when storage is fast NVMe
                                     (already saturates ~3 GB/s)
  - prefetch_safetensors:         ❌ ADDS overhead on fast NVMe (real bench
                                     showed +37ms for 277MB model). Kept in
                                     this module for slow-storage scenarios
                                     but DO NOT call from server.py paths.
  - runai-model-streamer:         ❌ Designed for S3/object storage, slower
                                     than safetensors on local NVMe. Also
                                     has cold-init overhead of ~20s on
                                     first call.

Honest takeaway: on Vast.ai/RunPod with NVMe local storage, the safetensors
deserialization path is NOT the cold-boot bottleneck. The bottlenecks are:
  1. Docker image pull (dominant for non-pre-baked images)
  2. transformers / diffusers high-level pipeline init
     (config + tokenizer + processor downloads, sequential network ops)
  3. Initial CUDA kernel compilation / warmup
  4. Model dispatch (CPU → GPU, dtype conversion)

Most "fast loaders" only help when storage is the bottleneck (S3, network
volumes, slow disks). Save them for that scenario.

Usage:

    from coldstart import bootstrap

    # Call ONCE at process start, BEFORE `import torch`
    bootstrap()
    import torch  # picks up TORCHINDUCTOR_* env vars
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
    repo_or_paths,  # str repo_id OR list[Path] OR None
    *,
    revision: str = "main",
    allow_patterns: Optional[Iterable[str]] = ("*.safetensors",),
) -> float:
    """Prefetch safetensors files into the OS page cache via sequential read.

    ⚠ DO NOT call this from server.py startup paths on fast NVMe hosts.
    Real GPU benchmark on Vast.ai RTX 4090 (2026-04-08) showed it ADDS
    ~37ms overhead for a 277MB model — the page-cache double-read is pure
    waste when the disk is already fast (3 GB/s+). Useful ONLY when:
      - Storage is slow (S3, network volume, magnetic disk)
      - File is huge (10+ GB) and the diffusers/transformers loader does
        many small reads in random order

    The theory is: transformers' from_pretrained() does many small random
    reads, so a sequential prefetch warms the page cache and turns the
    subsequent load into RAM-speed access. In practice, modern safetensors
    is already very efficient and modern NVMe can sustain random reads at
    near-sequential speeds — the optimization is usually a no-op or worse.

    Returns the elapsed prefetch time in seconds, or 0.0 if nothing was
    actually prefetched (no files found, all reads failed, etc.).
    Best-effort — never raises.
    """
    if not repo_or_paths:
        return 0.0

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
            # Strip multi-line HF error to a single short line
            err_brief = str(e).split("\n", 1)[0][:120]
            print(f"[coldstart] prefetch resolve failed: {err_brief}", file=sys.stderr)
            return 0.0
    else:
        try:
            files = [Path(p) for p in repo_or_paths]
        except TypeError:
            print(f"[coldstart] prefetch: invalid input type", file=sys.stderr)
            return 0.0

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

    if total_bytes == 0:
        # Nothing was actually prefetched — return 0.0 so callers can detect.
        return 0.0

    elapsed = time.time() - start
    mb = total_bytes / 1024 / 1024
    print(
        f"[coldstart] prefetched {mb:.0f} MB across {len(files)} file(s) "
        f"in {elapsed:.1f}s ({mb / elapsed:.0f} MB/s)",
        file=sys.stderr,
    )
    return elapsed


# ── fastsafetensors loader (optional, for raw weight loading) ────────────────


def load_with_fastsafetensors(files, device: str = "cuda:0", nogds: bool = False):
    """Load safetensors files via fastsafetensors → device tensors dict.

    Returns: dict[name, torch.Tensor] on `device`.

    Real GPU benchmark on Vast.ai RTX 4090 (2026-04-08):
        Ultravox 1.3GB single-shard:  1.74x faster than safetensors (5.4 GB/s
        vs 3.2 GB/s). 0.18s saved per cold load.

    ⚠ KNOWN LIMITATIONS (validated empirically):

    1. **Multi-shard models with repeated tensor keys FAIL**.
       fastsafetensors requires every tensor key to appear in exactly one
       file. Whisper-large-v3 (encoder + decoder both have
       `model.decoder.embed_positions.weight`) raises:
           Exception: FilesBufferOnDevice: key X must be unique among files
       This is unfixable at the loader level — would need to merge shards.

    2. **GPUDirect Storage (GDS) requires cuFile lib**.
       Without GDS, fastsafetensors uses CPU staging like safetensors.
       The 4.8-7.5x speedups in the paper require GDS hardware.

    3. **Marginal/no benefit on fast NVMe**.
       Local NVMe at ~3 GB/s is already saturating safetensors' deserializer.
       The biggest gains are when storage is slow (S3, EBS, network volumes).

    Use this for advanced flows where you construct the model from a state_dict
    yourself (skipping transformers' high-level loading). For most flows,
    transformers / diffusers' built-in loaders are already fast enough.

    `device` MUST include an index for CUDA — `"cuda:0"` not `"cuda"`. The
    fastsafetensors library rejects ambiguous device strings. CPU is fine.
    Falls back to standard safetensors.load_file() if fastsafetensors is not
    installed (ImportError) or fails.
    """
    # Normalize device string — fastsafetensors requires explicit index for CUDA
    if device == "cuda":
        device = "cuda:0"

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
