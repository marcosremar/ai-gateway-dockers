"""
LLM model loader with optional fastsafetensors acceleration (Phase B5).

Benchmark context: fastsafetensors is 4.8-7.5× faster than the safetensors
standard loader on decoder-only LLMs (Llama 7B/13B/70B, Gemma, Mistral,
Qwen). It achieves the speedup by doing direct GPU DMA via GPUDirect Storage
(or NVMe Scatter/Gather) — bypassing Python's file I/O.

What it WILL NOT handle correctly:
    Encoder-decoder models (Whisper, T5, BART) expect tensor keys across
    multiple shards with an encoder→decoder dependency that fastsafetensors
    does not flatten. We've seen it miss tensors silently, so we disable it
    for these architectures.

Usage:
    from llm_loader import load_model_fast
    model = load_model_fast("/models/translategemma-4b", device="cuda:0")

Any runtime exception (package missing, shard layout unusual, OOM during
the DMA pass, etc.) triggers a transparent fallback to the safetensors
standard loader and a warning — capture paths should never fail because
of this optimization.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger("llm-loader")

# Model families that MUST NOT go through fastsafetensors. Case-insensitive
# substring match against the repo id / local dir name.
ENCODER_DECODER_HINTS = (
    "whisper",
    "t5",
    "bart",
    "mbart",
    "marian",
    "nllb",
    "seamless",
    "encoder-decoder",
)

# Decoder-only families where fastsafetensors is known to work well.
DECODER_ONLY_HINTS = (
    "llama",
    "mistral",
    "mixtral",
    "gemma",
    "qwen",
    "phi",
    "falcon",
    "yi",
    "deepseek",
    "translategemma",
)


def _is_encoder_decoder(name: str) -> bool:
    lowered = name.lower()
    return any(h in lowered for h in ENCODER_DECODER_HINTS)


def _is_decoder_only(name: str) -> bool:
    lowered = name.lower()
    return any(h in lowered for h in DECODER_ONLY_HINTS)


def _safetensors_files(path: str) -> list[str]:
    p = Path(path)
    if p.is_file() and p.suffix == ".safetensors":
        return [str(p)]
    if p.is_dir():
        return sorted(str(x) for x in p.glob("*.safetensors"))
    return []


def load_with_fastsafetensors(files: Iterable[str], device: str = "cuda:0") -> Any:
    """
    Invoke the fastsafetensors loader. Caller is responsible for catching
    any exception and calling `load_with_safetensors` as fallback — this
    helper intentionally does NOT wrap in try/except so callers observe
    failures (and metrics count them).
    """
    import fastsafetensors  # noqa: F401 — imported for its side effect of registering loaders

    try:
        # Newer API — operator-friendly tensor dict.
        from fastsafetensors import safe_open_fast  # type: ignore[attr-defined]
    except ImportError:
        safe_open_fast = None  # type: ignore[assignment]

    tensors: dict[str, Any] = {}
    if safe_open_fast is not None:
        for f in files:
            with safe_open_fast(f, device=device) as handle:  # type: ignore[misc]
                for key in handle.keys():
                    tensors[key] = handle.get_tensor(key)
        return tensors

    # Fallback to the lower-level SafeTensorsFileLoader API.
    from fastsafetensors import SafeTensorsFileLoader  # type: ignore[attr-defined]

    loader = SafeTensorsFileLoader(pg=None, device=device, bbuf_size_kb=16384)
    loader.add_filenames({0: list(files)})
    loaded = loader.copy_files_to_device()
    for name in loaded.key_to_rank_lidx:
        tensors[name] = loaded.get_tensor(name)
    return tensors


def load_with_safetensors(files: Iterable[str], device: str = "cuda:0") -> Any:
    """Baseline loader — safe for all architectures including encoder-decoder."""
    from safetensors import safe_open  # type: ignore[import-not-found]

    tensors: dict[str, Any] = {}
    for f in files:
        with safe_open(f, framework="pt", device=device) as handle:
            for key in handle.keys():
                tensors[key] = handle.get_tensor(key)
    return tensors


def load_model_fast(
    path: str,
    *,
    device: str = "cuda:0",
    model_hint: str | None = None,
    force_baseline: bool | None = None,
) -> Any:
    """
    Load tensors from a safetensors-based model directory or shard file.

    Args:
        path: Local directory (multi-shard model) or single .safetensors file.
        device: Torch device string.
        model_hint: Override for architecture detection (e.g. "whisper-large-v3").
                    Defaults to the basename of `path`.
        force_baseline: If True, never try fastsafetensors (debugging aid).
                        Default: read from `BABELCAST_FORCE_SAFETENSORS` env var.

    Returns:
        A dict[str, Tensor] — same shape as `safetensors.safe_open(...).get_tensor()`.

    Raises:
        FileNotFoundError: if no .safetensors files found under `path`.

    Behaviour is always safe: if fastsafetensors raises at any point, we
    warn and retry via the stable safetensors path. The baseline path is
    proven to work on every architecture we ship.
    """
    if force_baseline is None:
        force_baseline = os.environ.get("BABELCAST_FORCE_SAFETENSORS") == "1"

    files = _safetensors_files(path)
    if not files:
        raise FileNotFoundError(f"No .safetensors files under {path}")

    name = model_hint or Path(path).name

    if force_baseline:
        log.info("load_model_fast: baseline forced (%s)", name)
        return load_with_safetensors(files, device=device)

    if _is_encoder_decoder(name):
        log.info("load_model_fast: encoder-decoder detected (%s) — using safetensors", name)
        return load_with_safetensors(files, device=device)

    # Decoder-only families — try fastsafetensors first, fall back on any error.
    if _is_decoder_only(name):
        try:
            log.info("load_model_fast: fastsafetensors for %s", name)
            return load_with_fastsafetensors(files, device=device)
        except Exception as err:  # noqa: BLE001 — broad catch intentional for fallback safety
            log.warning(
                "load_model_fast: fastsafetensors failed for %s (%s: %s); falling back to safetensors",
                name,
                type(err).__name__,
                err,
            )
            return load_with_safetensors(files, device=device)

    # Unknown architecture — default to safe path.
    log.info("load_model_fast: architecture unknown (%s) — using safetensors", name)
    return load_with_safetensors(files, device=device)
