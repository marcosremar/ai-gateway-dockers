"""Modal TTS benchmark — test Qwen3-TTS latency on GPU.

Installs everything from pip (no Docker image needed).
Deploys on Modal L40S GPU, runs TTS benchmark, returns results.
Measures both total latency and time-to-first-chunk (streaming).

Usage:
    modal run docker/modal/tts_benchmark.py
"""

import modal
import time

app = modal.App("babelcast-tts-bench")

hf_secret = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "sox", "libsox-dev")
    .pip_install(
        "torch", "torchaudio",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers==4.57.3", "accelerate>=1.12.0",
        "soundfile", "numpy", "huggingface-hub", "hf_transfer",
        "librosa", "einops", "onnxruntime", "sox",
    )
    .run_commands("pip install --no-deps 'qwen-tts>=0.1.1' 'faster-qwen3-tts>=0.2.1'")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


def _patch_transformers_compat():
    """Backport transformers 5.x symbols to 4.57.x for qwen-tts 0.1.1."""
    import math
    import sys
    import types

    import torch
    import transformers
    import transformers.modeling_utils as mu

    if not hasattr(mu, "ALL_ATTENTION_FUNCTIONS"):
        def _eager_attention(query, key, value, attn_mask=None, dropout_p=0.0, **kw):
            scale = 1.0 / math.sqrt(query.size(-1))
            w = torch.matmul(query, key.transpose(-2, -1)) * scale
            if attn_mask is not None:
                w = w + attn_mask
            w = torch.nn.functional.softmax(w, dim=-1)
            if dropout_p > 0.0:
                w = torch.nn.functional.dropout(w, p=dropout_p)
            return torch.matmul(w, value), w

        mu.ALL_ATTENTION_FUNCTIONS = {
            "default": _eager_attention,
            "eager": _eager_attention,
            "sdpa": torch.nn.functional.scaled_dot_product_attention,
            "flash_attention_2": _eager_attention,
        }

    if not hasattr(transformers, "modeling_layers"):
        mod = types.ModuleType("transformers.modeling_layers")
        transformers.modeling_layers = mod
        sys.modules["transformers.modeling_layers"] = mod
    if not hasattr(transformers.modeling_layers, "GradientCheckpointingLayer"):
        class GradientCheckpointingLayer(torch.nn.Module):
            pass
        transformers.modeling_layers.GradientCheckpointingLayer = GradientCheckpointingLayer

    import transformers.utils.generic as _tg
    def _noop(func=None):
        return func if func is not None else (lambda f: f)
    _tg.check_model_inputs = _noop


@app.function(image=image, gpu="L40S", timeout=600, secrets=[hf_secret])
def benchmark_tts():
    """Run TTS benchmark on GPU — total latency + streaming TTFB."""
    import io
    import traceback
    import soundfile as sf

    results = []

    try:
        _patch_transformers_compat()

        # 1. Model load
        print("Loading faster-qwen3-tts model...")
        t0 = time.perf_counter()
        from faster_qwen3_tts import FasterQwen3TTS
        model = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
        load_ms = (time.perf_counter() - t0) * 1000
        print(f"Model loaded in {load_ms:.0f}ms")
        results.append({"stage": "model_load", "latency_ms": round(load_ms, 1)})

        phrases = [
            ("Hello everyone, welcome to this meeting.", "English"),
            ("Today we are going to discuss the new project.", "English"),
            ("The results of the last quarter are very encouraging.", "English"),
            ("We have increased our turnover by fifteen percent.", "English"),
            ("The next step is to launch the beta version.", "English"),
            ("Does anyone have any questions about the schedule?", "English"),
            ("Thank you very much for your participation today.", "English"),
            ("The next meeting is scheduled for next Tuesday.", "English"),
        ]

        # 2. Streaming FIRST (to measure true cold TTFB with CUDA graph compilation)
        print("\n--- Streaming (time-to-first-chunk) ---")
        for i, (text, lang) in enumerate(phrases):
            t0 = time.perf_counter()
            ttfb_ms = None
            total_chunks = 0
            total_audio_dur = 0.0

            for audio_chunk, sr, timing in model.generate_custom_voice_streaming(
                text=text, speaker="Ryan", language=lang, chunk_size=8
            ):
                if ttfb_ms is None:
                    ttfb_ms = (time.perf_counter() - t0) * 1000
                total_chunks += 1
                total_audio_dur += len(audio_chunk) / sr

            total_ms = (time.perf_counter() - t0) * 1000
            label = "COLD" if i == 0 else "WARM"
            print(f"  [{i}] {label}: TTFB={ttfb_ms:.0f}ms, total={total_ms:.0f}ms, {total_chunks} chunks, {total_audio_dur:.1f}s audio | \"{text[:40]}\"")
            results.append({
                "stage": "tts_streaming", "iteration": i,
                "ttfb_ms": round(ttfb_ms, 1) if ttfb_ms else None,
                "total_ms": round(total_ms, 1),
                "chunks": total_chunks,
                "audio_duration_s": round(total_audio_dur, 1),
                "cold": i == 0, "text": text[:50],
            })

        # 3. Summary
        stream_results = [r for r in results if r["stage"] == "tts_streaming"]
        warm_ttfb = [r["ttfb_ms"] for r in stream_results if not r["cold"] and r["ttfb_ms"]]
        warm_total = [r["total_ms"] for r in stream_results if not r["cold"]]

        summary = {
            "model_load_ms": results[0]["latency_ms"],
            "cold_ttfb_ms": stream_results[0]["ttfb_ms"] if stream_results else None,
            "cold_total_ms": stream_results[0]["total_ms"] if stream_results else None,
            "warm_ttfb_avg_ms": round(sum(warm_ttfb) / len(warm_ttfb), 1) if warm_ttfb else 0,
            "warm_ttfb_min_ms": round(min(warm_ttfb), 1) if warm_ttfb else 0,
            "warm_ttfb_max_ms": round(max(warm_ttfb), 1) if warm_ttfb else 0,
            "warm_total_avg_ms": round(sum(warm_total) / len(warm_total), 1) if warm_total else 0,
            "gpu": "L40S",
        }

        print(f"\n{'='*60}")
        print(f"  Qwen3-TTS Streaming Benchmark (Modal L40S)")
        print(f"{'='*60}")
        print(f"  Model load:       {summary['model_load_ms']:.0f}ms")
        print(f"  ---")
        print(f"  Cold TTFB:        {summary['cold_ttfb_ms']:.0f}ms (includes CUDA graph compilation)" if summary['cold_ttfb_ms'] else "  Cold TTFB:        N/A")
        print(f"  Cold total:       {summary['cold_total_ms']:.0f}ms" if summary['cold_total_ms'] else "")
        print(f"  ---")
        print(f"  Warm TTFB avg:    {summary['warm_ttfb_avg_ms']:.0f}ms")
        print(f"  Warm TTFB range:  {summary['warm_ttfb_min_ms']:.0f}ms - {summary['warm_ttfb_max_ms']:.0f}ms")
        print(f"  Warm total avg:   {summary['warm_total_avg_ms']:.0f}ms")
        print(f"  ---")
        print(f"  OpenAI TTS avg:   ~726ms (total, no streaming)")
        if summary['warm_ttfb_avg_ms'] > 0:
            print(f"  TTFB speedup:     {726/summary['warm_ttfb_avg_ms']:.1f}x vs OpenAI")
        print(f"{'='*60}")

        return {"results": results, "summary": summary}
    except Exception as e:
        print(f"\n\nERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise


@app.local_entrypoint()
def main():
    result = benchmark_tts.remote()
    print("\n\nFinal results returned from Modal:")
    s = result["summary"]
    print(f"  Warm total avg:  {s['warm_total_avg_ms']:.0f}ms")
    print(f"  Warm TTFB avg:   {s['warm_ttfb_avg_ms']:.0f}ms")
    print(f"  OpenAI TTS:      ~726ms")
    if s['warm_ttfb_avg_ms'] > 0:
        print(f"  TTFB speedup:    {726/s['warm_ttfb_avg_ms']:.1f}x vs OpenAI")
