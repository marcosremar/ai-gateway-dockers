"""Qwen3-TTS Service — FastAPI wrapper pro ai-gateway.



OpenAI-compatible endpoints so the gateway pipeline can swap qwen3-tts in
for kokoro-tts or Modal qwen3-tts without caller changes:

  GET  /health                — readiness probe.
  GET  /v1/audio/voices       — list built-in speaker presets.
  POST /v1/audio/speech       — JSON body {input, voice, response_format}.
  POST /v1/audio/speech/clone — multipart {text, reference_audio, ref_text}.
                                Voice-cloned synthesis using the 24kHz mono
                                reference clip; ref_text is the transcript
                                the model conditions the prosody encoder on.

Env:
  IDLE_TIMEOUT_MIN     — auto-shutdown after no requests (default 15).
  QWEN3_TTS_MODEL      — HF repo id (default Qwen/Qwen3-TTS-12Hz-1.7B-Base).
  QWEN3_TTS_DTYPE      — bfloat16|float16|float32 (default bfloat16).
  QWEN3_TTS_ATTN       — flash_attention_2|sdpa|eager (default sdpa).
  QWEN3_TTS_LANG       — default language (default "English").
  QWEN3_TTS_TIMEOUT_S  — per-request synth timeout seconds (default 120).
  QWEN3_TTS_WARMUP     — set 0 to skip startup warmup (default 1).
  PORT                 — listen port (default 8000).
"""

import asyncio
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
import uvicorn

from idle_watchdog import add_idle_middleware, start_watchdog, touch_activity

# ── Logging ─────────────────────────────────────────────────────────────────

# canal-dark patch: filter kwargs to those accepted by the installed
# Qwen3-TTS lib build (e.g. drop `x_vector_only_mode` on older builds).
import inspect as _canal_inspect
def _safe_clone_kwargs(model, kwargs):
    """Filter unknown kwargs + coerce ref_audio tuples → temp WAV path.

    Older Qwen3-TTS builds accept (np.array, sr) for ref_audio; newer ones
    only accept a file path. We detect the tuple form and write a temp WAV
    so both builds are happy.
    """
    out = dict(kwargs)
    ra = out.get('ref_audio')
    if isinstance(ra, tuple) and len(ra) == 2:
        try:
            import io as _io, soundfile as _sf, tempfile as _tmp, os as _os
            arr, sr = ra
            # Try lib's accepted form: file path
            tmp = _tmp.NamedTemporaryFile(suffix='.wav', delete=False)
            _sf.write(tmp.name, arr, int(sr), subtype='PCM_16')
            out['ref_audio'] = tmp.name
        except Exception as _e:
            pass  # fall back to original tuple if soundfile missing
    try:
        sig = _canal_inspect.signature(model.generate_voice_clone)
        accepted = set(sig.parameters.keys())
        return {k: v for k, v in out.items() if k in accepted}
    except (TypeError, ValueError):
        return out
def _gvc(model, **kw):
    return model.generate_voice_clone(**_safe_clone_kwargs(model, kw))

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("qwen3-tts")


# Patch transformers' MimiConv1d._pad1d (used by Qwen3-TTS speech tokenizer).
# torch.nn.functional.pad with mode="replicate" / "reflect" has no fp16 kernel
# (`replication_pad1d not implemented for Half`), so when the model is loaded
# in fp16 the voice-clone path crashes inside the mimi audio encoder. We
# upcast the conv input to fp32 transiently and cast back. Negligible perf
# hit, conv is tiny relative to the rest of the encoder.
def _patch_mimi_pad1d() -> None:
    try:
        from transformers.models.mimi import modeling_mimi as _mimi  # type: ignore
        if getattr(_mimi.MimiConv1d._pad1d, "_qwen3_patched", False):
            return
        _orig = _mimi.MimiConv1d._pad1d
        def _safe_pad1d(hidden_states, paddings, mode="zero", value=0.0):
            if hidden_states.dtype in (torch.float16,):
                # bf16 has a native replication_pad1d kernel on recent torch,
                # only fp16 still lacks it — keep upcast scoped to fp16 only.
                out = _orig(hidden_states.to(torch.float32), paddings, mode, value)
                return out.to(hidden_states.dtype)
            return _orig(hidden_states, paddings, mode, value)
        _safe_pad1d._qwen3_patched = True  # type: ignore[attr-defined]
        _mimi.MimiConv1d._pad1d = staticmethod(_safe_pad1d)
        log.info("[tts.patch] mimi pad1d fp16 upcast installed")
    except Exception as exc:
        log.warning(f"[tts.patch.fail] {exc}")


_patch_mimi_pad1d()

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_REPO = os.environ.get("QWEN3_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
SAMPLE_RATE = 24_000
DEFAULT_LANG = os.environ.get("QWEN3_TTS_LANG", "English")
TIMEOUT_S = float(os.environ.get("QWEN3_TTS_TIMEOUT_S", "120"))
WARMUP = os.environ.get("QWEN3_TTS_WARMUP", "1") != "0"

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float16": torch.float16,   "fp16": torch.float16,   "half": torch.float16,
    "float32": torch.float32,   "fp32": torch.float32,   "float": torch.float32,
}
DTYPE_NAME = os.environ.get("QWEN3_TTS_DTYPE", "bfloat16").lower()
DTYPE = _DTYPE_MAP.get(DTYPE_NAME, torch.bfloat16)
ATTN_IMPL = os.environ.get("QWEN3_TTS_ATTN", "sdpa")  # safer default than flash_attention_2

inference_lock = asyncio.Lock()
boot_ts = time.time()
model = None
load_error: Optional[str] = None
load_traceback: Optional[str] = None
warmup_ok: bool = False
warmup_error: Optional[str] = None

# Built-in voices — actual list comes from the model. We expose a static
# fallback for the listing endpoint when the model fails to load so the
# gateway autoscaler can still call /health and trigger a redeploy.
DEFAULT_VOICES = [
    {"id": "Ryan",     "language": "en", "gender": "male"},
    {"id": "Vivian",   "language": "en", "gender": "female"},
    {"id": "Cherry",   "language": "en", "gender": "female"},
    {"id": "Ethan",    "language": "en", "gender": "male"},
    {"id": "Serena",   "language": "zh", "gender": "female"},
    {"id": "Sunny",    "language": "zh", "gender": "female"},
]

# Cached voice-clone prompts: name -> precomputed prompt items returned by
# model.create_voice_clone_prompt. Built once via /v1/voices/register or the
# auto-register-on-startup path; reused on every /v1/audio/speech/cached call
# so we skip the per-request speaker-embedding extraction (~1-3s GPU).
VOICE_CACHE: dict = {}


# ── Model loading ───────────────────────────────────────────────────────────

def _load_model() -> None:
    """Load Qwen3TTSModel using the OFFICIAL kwarg names.

    Per https://github.com/QwenLM/Qwen3-TTS the constructor expects
    `dtype=` (not `torch_dtype=`), `device_map=`, and an optional
    `attn_implementation=`. The previous implementation used `torch_dtype=`
    which the library silently ignores → model loaded in default dtype with
    nondeterministic device placement, leading to indefinite hangs on the
    first generation call.
    """
    global model, load_error, load_traceback
    log.info(f"[tts.load.start] repo={MODEL_REPO} dtype={DTYPE_NAME} attn={ATTN_IMPL}")
    try:
        # Prefer faster-qwen3-tts (CUDA Graph capture + static KV cache,
        # ~5x speedup over qwen-tts on the same model). Falls back to
        # qwen-tts if the wheel is missing or the override env disables it.
        use_faster = os.environ.get("QWEN3_TTS_USE_FASTER", "1") != "0"
        FastCls = None
        if use_faster:
            try:
                from faster_qwen3_tts import FasterQwen3TTS as FastCls  # type: ignore
                log.info("[tts.load.faster] using faster-qwen3-tts (CUDA Graphs)")
            except Exception as e:
                log.warning(f"[tts.load.faster.unavailable] {e!r} — falling back to qwen-tts")
                FastCls = None

        from qwen_tts import Qwen3TTSModel  # type: ignore

        device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        kwargs = dict(device_map=device_map, dtype=DTYPE)
        if ATTN_IMPL and ATTN_IMPL != "auto":
            kwargs["attn_implementation"] = ATTN_IMPL

        if FastCls is not None:
            if not torch.cuda.is_available():
                # FasterQwen3TTS captures CUDA graphs at construction → can't
                # load on CPU. Skip the early-load (snap=True) path; the
                # snap=False enter will retry once GPU is attached.
                log.info("[tts.load.faster.defer] no GPU yet — deferring to snap=False")
                return
            # FasterQwen3TTS uses `device=` (not `device_map=`).
            try:
                fast_kwargs = dict(
                    device="cuda",
                    dtype=DTYPE,
                    attn_implementation=(ATTN_IMPL if ATTN_IMPL and ATTN_IMPL != "auto" else "sdpa"),
                )
                mdl = FastCls.from_pretrained(MODEL_REPO, **fast_kwargs)
                log.info("[tts.load.done] backend=faster-qwen3-tts")
                model = mdl
                return
            except Exception as e:
                import traceback
                log.warning(f"[tts.load.faster.fail] {e!r}\n{traceback.format_exc()}\n— falling back to qwen-tts")

        try:
            mdl = Qwen3TTSModel.from_pretrained(MODEL_REPO, **kwargs)
        except TypeError as te:
            # Older qwen-tts wheel may not accept attn_implementation —
            # retry without it. Same for device_map vs device.
            log.warning(f"[tts.load.retry] {te} — retrying without attn_implementation")
            kwargs.pop("attn_implementation", None)
            try:
                mdl = Qwen3TTSModel.from_pretrained(MODEL_REPO, **kwargs)
            except TypeError as te2:
                log.warning(f"[tts.load.retry2] {te2} — falling back to torch_dtype/device")
                mdl = Qwen3TTSModel.from_pretrained(
                    MODEL_REPO, torch_dtype=DTYPE,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )

        # Some Qwen3TTSModel wheels are nn.Module subclasses (have .to/.eval),
        # others are wrapper objects without those methods. Be defensive.
        for fn in ("to", "eval"):
            method = getattr(mdl, fn, None)
            if callable(method):
                try:
                    method("cuda") if fn == "to" else method()
                except Exception as e:
                    log.warning(f"[tts.load.{fn}.skip] {e}")
        model = mdl
        log.info("[tts.load.done]")
    except Exception as exc:
        import traceback
        load_error = str(exc)
        load_traceback = traceback.format_exc()
        log.error(f"[tts.load.fail] {exc}\n{load_traceback}")


def _warmup() -> None:
    """JIT CUDA kernels + verify the synthesis path works end-to-end.

    The Base checkpoint (Qwen3-TTS-12Hz-1.7B-Base) only supports
    generate_voice_clone (NOT generate_custom_voice / generate_voice_design)
    AND requires a non-empty ref_text in ICL mode (the installed qwen-tts
    build drops the x_vector_only_mode kwarg, so silent_ref + empty ref_text
    used to throw "ref_text is required ... Bad index=0" and leave CUDA
    graph state half-captured — every subsequent synth then fails with
    "Offset increment outside graph capture". Fix: warm up with the same
    baked-in ref the auto-register uses (always non-empty).
    """
    global warmup_ok, warmup_error
    if model is None:
        warmup_error = "model not loaded; skipping warmup"
        return
    audio_path = os.environ.get("QWEN3_TTS_AUTO_REGISTER_AUDIO", "/opt/voice-ref.wav")
    text_path = os.environ.get("QWEN3_TTS_AUTO_REGISTER_TEXT", "/opt/voice-ref.txt")
    if not (os.path.exists(audio_path) and os.path.exists(text_path)):
        warmup_error = f"warmup ref missing: {audio_path}/{text_path}"
        log.warning(f"[tts.warmup.skip] {warmup_error}")
        return
    try:
        log.info("[tts.warmup.start]")
        t0 = time.time()
        ref_audio_arr, ref_sr = sf.read(audio_path, dtype="float32")
        if ref_audio_arr.ndim > 1:
            ref_audio_arr = ref_audio_arr.mean(axis=1)
        with open(text_path, "r") as fh:
            ref_text = fh.read().strip()
        if not ref_text:
            warmup_error = "warmup ref_text empty"
            log.warning(f"[tts.warmup.skip] {warmup_error}")
            return
        with torch.inference_mode():
            try:
                audios, sr = model.generate_custom_voice(
                    text="Hello.", language=DEFAULT_LANG, speaker="Ryan",
                )
            except (ValueError, AttributeError, NotImplementedError) as e:
                if "does not support" in str(e) or "not implemented" in str(e).lower():
                    log.info(f"[tts.warmup.fallback] preset unsupported; using clone with bundled ref")
                    audios, sr = _gvc(
                        model,
                        text="Hello.",
                        language=DEFAULT_LANG,
                        ref_audio=(ref_audio_arr, int(ref_sr)),
                        ref_text=ref_text,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.7,
                    )
                else:
                    raise
        audio = audios[0] if isinstance(audios, (list, tuple)) else audios
        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()
        n = int(np.asarray(audio).reshape(-1).shape[0])
        warmup_ok = True
        log.info(f"[tts.warmup.done] sr={sr} samples={n} elapsed={time.time()-t0:.2f}s")
    except Exception as exc:
        import traceback
        warmup_error = str(exc)
        log.error(f"[tts.warmup.fail] {exc}\n{traceback.format_exc()}")


def _register_voice_sync(name: str, ref_audio: np.ndarray, ref_sr: int,
                         ref_text: str) -> None:
    """Build a reusable voice clone prompt and stash it under VOICE_CACHE[name].

    Called from the runtime registration endpoint and from the auto-register
    bootstrap path (env var QWEN3_TTS_AUTO_REGISTER_VOICE)."""
    if model is None:
        raise RuntimeError("model not loaded")
    ref_arr = _resample_mono(ref_audio, int(ref_sr or 24_000), SAMPLE_RATE)
    MAX_REF_S = int(os.environ.get("QWEN3_TTS_MAX_REF_S", "15"))
    if ref_arr.shape[0] > MAX_REF_S * SAMPLE_RATE:
        ref_arr = ref_arr[: MAX_REF_S * SAMPLE_RATE]
    log.info(f"[tts.voice.register.start] name={name} samples={ref_arr.shape[0]} "
             f"ref_text_len={len(ref_text)}")
    t0 = time.time()
    # FasterQwen3TTS wraps qwen-tts; the prompt builder lives on the inner
    # `.model` attribute. Original Qwen3TTSModel exposes it directly.
    prompt_builder = getattr(model, "create_voice_clone_prompt", None) \
        or getattr(getattr(model, "model", None), "create_voice_clone_prompt", None)
    if prompt_builder is None:
        raise RuntimeError("create_voice_clone_prompt not found on model")
    with torch.inference_mode():
        prompt = prompt_builder(
            ref_audio=(ref_arr, SAMPLE_RATE),
            ref_text=ref_text,
            x_vector_only_mode=not bool(ref_text),
        )
    VOICE_CACHE[name] = prompt
    log.info(f"[tts.voice.register.done] name={name} elapsed={time.time()-t0:.2f}s "
             f"cache_size={len(VOICE_CACHE)}")


def _auto_register_voice() -> None:
    """If env vars point at a baked voice ref, register it on startup."""
    name = os.environ.get("QWEN3_TTS_AUTO_REGISTER_VOICE", "").strip()
    audio_path = os.environ.get("QWEN3_TTS_AUTO_REGISTER_AUDIO", "/opt/voice-ref.wav")
    text_path = os.environ.get("QWEN3_TTS_AUTO_REGISTER_TEXT", "/opt/voice-ref.txt")
    if not name:
        return
    if not (os.path.exists(audio_path) and os.path.exists(text_path)):
        log.warning(f"[tts.voice.auto.skip] name={name} missing files "
                    f"audio={audio_path} text={text_path}")
        return
    try:
        ref_audio, ref_sr = sf.read(audio_path, dtype="float32")
        if ref_audio.ndim > 1:
            ref_audio = ref_audio.mean(axis=1)
        with open(text_path, "r") as fh:
            ref_text = fh.read().strip()
        _register_voice_sync(name, ref_audio, int(ref_sr), ref_text)
    except Exception as exc:
        log.error(f"[tts.voice.auto.fail] name={name} {exc}")


async def _load_and_warmup_async() -> None:
    await asyncio.get_event_loop().run_in_executor(None, _load_model)
    # Register voice FIRST so the gateway has a usable cached voice even when
    # warmup later fails (the auto-register path uses the prompt builder
    # directly and is more permissive than the full clone pipeline).
    if model is not None:
        await asyncio.get_event_loop().run_in_executor(None, _auto_register_voice)
    if WARMUP and model is not None:
        await asyncio.get_event_loop().run_in_executor(None, _warmup)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Fire-and-forget — uvicorn must accept /health probes immediately
    # so the gateway boot poller doesn't time out during the ~30-90s the
    # weights take to download+initialise. /health reports degraded until
    # `model` is bound; /v1/audio/speech returns 503 in the same window.
    asyncio.create_task(_load_and_warmup_async())
    asyncio.create_task(start_watchdog())
    yield


app = FastAPI(title="Qwen3-TTS (ai-gateway)", lifespan=lifespan)
add_idle_middleware(app)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "healthy" if (load_error is None and warmup_error is None) else "degraded",
        "model_loaded": model is not None,
        "warmup_ok": warmup_ok,
        "warmup_error": warmup_error,
        "model": MODEL_REPO,
        "dtype": DTYPE_NAME,
        "attn_impl": ATTN_IMPL,
        "language_default": DEFAULT_LANG,
        "uptime_s": round(time.time() - boot_ts, 1),
        "gpu_available": torch.cuda.is_available(),
        "load_error": load_error,
    }


@app.get("/health/diag")
async def health_diag() -> dict:
    """Detailed diagnostic — includes traceback if load failed."""
    return {
        **(await health()),
        "load_traceback": load_traceback,
    }


@app.get("/v1/audio/voices")
async def list_voices() -> dict:
    return {"object": "list", "data": DEFAULT_VOICES}


class SpeechRequest(BaseModel):
    model: Optional[str] = None
    input: str
    voice: str = "Ryan"
    response_format: str = "wav"
    speed: float = 1.0
    language: Optional[str] = None


def _encode(audio: np.ndarray, fmt: str, sr: int = SAMPLE_RATE) -> tuple[bytes, str]:
    buf = io.BytesIO()
    if fmt in ("wav", "pcm"):
        sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue(), "audio/wav"
    if fmt == "flac":
        sf.write(buf, audio, sr, format="FLAC")
        return buf.getvalue(), "audio/flac"
    if fmt == "mp3":
        sf.write(buf, audio, sr, format="MP3")
        return buf.getvalue(), "audio/mpeg"
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue(), "audio/wav"


def _resample_mono(audio: np.ndarray, sr_in: int, sr_out: int = 24_000) -> np.ndarray:
    """Resample to mono `sr_out`. Uses librosa if available (better quality),
    falls back to a simple linear interp."""
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if int(sr_in) == int(sr_out):
        return a
    try:
        import librosa  # type: ignore
        return librosa.resample(a, orig_sr=int(sr_in), target_sr=int(sr_out)).astype(np.float32)
    except Exception:
        n_out = int(round(a.shape[0] * sr_out / sr_in))
        if n_out <= 0:
            return a
        x_old = np.linspace(0.0, 1.0, num=a.shape[0], endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return np.interp(x_new, x_old, a).astype(np.float32)


def _synth(text: str, voice: str, speed: float = 1.0,
           reference_audio: Optional[np.ndarray] = None,
           reference_sr: Optional[int] = None,
           ref_text: Optional[str] = None,
           language: Optional[str] = None,
           cached_prompt=None) -> tuple[np.ndarray, int]:
    """Synthesize. Returns (audio_float32, sample_rate).

    qwen-tts's Qwen3TTSModel exposes three top-level methods:
      - generate_voice_clone(text, language, ref_audio, ref_text, ...)  → ICL clone
      - generate_voice_design(text, language, instruct, ...)            → free voice design
      - generate_custom_voice(text, language, speaker, ...)             → preset speakers

    We pick voice-clone when reference_audio is provided, custom-voice
    otherwise. Both return (List[np.ndarray], sample_rate).
    """
    if model is None:
        raise RuntimeError(load_error or "model not loaded")

    lang = (language or DEFAULT_LANG).strip()
    # Qwen3 expects capitalized language names: "English", "Chinese", "Auto"
    # — lowercase silently routes to a fallback that may hang on the
    # speaker-conditioning branch. Normalise common aliases.
    LANG_NORM = {
        "en": "English", "english": "English",
        "zh": "Chinese", "chinese": "Chinese",
        "ja": "Japanese", "japanese": "Japanese",
        "ko": "Korean", "korean": "Korean",
        "de": "German", "german": "German",
        "fr": "French", "french": "French",
        "ru": "Russian", "russian": "Russian",
        "pt": "Portuguese", "portuguese": "Portuguese",
        "es": "Spanish", "spanish": "Spanish",
        "it": "Italian", "italian": "Italian",
        "auto": "Auto",
    }
    lang = LANG_NORM.get(lang.lower(), lang)

    mode = ('cached-clone' if cached_prompt is not None
            else ('clone' if reference_audio is not None else 'preset'))
    log.info(f"[tts.synth.start] mode={mode} "
             f"lang={lang} voice={voice} text_len={len(text)}")

    t0 = time.time()
    with torch.inference_mode():
        if cached_prompt is not None:
            est_tokens = min(2048, max(130, len(text) * 3 + 100))
            log.info(f"[tts.model.call] generate_voice_clone "
                     f"voice_clone_prompt=cached max_new_tokens={est_tokens}")
            # faster-qwen3-tts <0.2.6 marks ref_audio/ref_text as positional
            # required even when voice_clone_prompt supplied. Pass them as
            # defensive fallback (paths from auto-register env vars).
            ra_path = os.environ.get("QWEN3_TTS_AUTO_REGISTER_AUDIO", "/opt/voice-ref.wav")
            rt_path = os.environ.get("QWEN3_TTS_AUTO_REGISTER_TEXT", "/opt/voice-ref.txt")
            ra_arg = ra_path if os.path.exists(ra_path) else None
            try:
                rt_text = open(rt_path).read().strip() if os.path.exists(rt_path) else ""
            except Exception:
                rt_text = ""
            audios, sr = _gvc(model,
                text=text,
                language=lang,
                ref_audio=ra_arg,
                ref_text=rt_text,
                voice_clone_prompt=cached_prompt,
                max_new_tokens=est_tokens,
                do_sample=True,
                temperature=0.9,
                top_p=1.0,
                top_k=50,
                repetition_penalty=1.05,
            )
        elif reference_audio is not None:
            ref_arr = _resample_mono(reference_audio, int(reference_sr or 24_000), SAMPLE_RATE)
            # Truncate ref to first 15s — Qwen3-TTS-Base recommends 3-30s
            # references, and longer clips empirically degrade the prosody
            # encoder (it produced 2s outputs for full 5min refs in tests).
            MAX_REF_S = int(os.environ.get("QWEN3_TTS_MAX_REF_S", "15"))
            if ref_arr.shape[0] > MAX_REF_S * SAMPLE_RATE:
                ref_arr = ref_arr[: MAX_REF_S * SAMPLE_RATE]
                log.info(f"[tts.model.ref.trim] truncated ref to {MAX_REF_S}s")
            # Cap max_new_tokens proportional to text length. Codec is
            # 12.5Hz mono → 1 codec token = 80ms. English speech ~14
            # chars/sec → 1 char ≈ 0.9 codec tokens. The official default
            # is 8192 (no per-call cap), but with our sampling params the
            # model usually emits EOS — keep cap as safety net at 3
            # tokens/char + 100 slack:
            #   - 10 chars → 130 tokens → 10s ceiling
            #   - 91 chars → 373 tokens → 30s ceiling
            #   - 500 chars → 1600 tokens → 128s ceiling
            # Trailing garble is cut by VAD silence-detection below.
            est_tokens = min(2048, max(130, len(text) * 3 + 100))
            log.info(f"[tts.model.call] generate_voice_clone "
                     f"ref_samples={ref_arr.shape[0]} ref_sr={SAMPLE_RATE} "
                     f"ref_text_len={len(ref_text or '')} max_new_tokens={est_tokens}")
            # Sampling params per official generation_config.json. Earlier
            # tests with temp=0.7 / rep_penalty=1.1 produced 60Hz pitch
            # shift away from ref (115Hz male ref → 184Hz tenor output);
            # the higher temp + lower rep penalty matches the 0.89 speaker
            # similarity benchmark from the Qwen3-TTS technical report.
            audios, sr = _gvc(model, 
                text=text,
                language=lang,
                ref_audio=(ref_arr, SAMPLE_RATE),
                ref_text=ref_text or "",
                x_vector_only_mode=not bool(ref_text),
                max_new_tokens=est_tokens,
                do_sample=True,
                temperature=0.9,
                top_p=1.0,
                top_k=50,
                repetition_penalty=1.05,
            )
        else:
            log.info(f"[tts.model.call] generate_custom_voice speaker={voice}")
            try:
                audios, sr = model.generate_custom_voice(
                    text=text, language=lang, speaker=voice,
                )
            except Exception as e1:
                log.warning(f"[tts.model.custom.fail] {e1} — falling back to voice_design")
                audios, sr = model.generate_voice_design(
                    text=text, language=lang, instruct=f"Speak as {voice}.",
                )

    log.info(f"[tts.model.done] elapsed={time.time()-t0:.2f}s")

    audio = audios[0] if isinstance(audios, (list, tuple)) else audios
    if hasattr(audio, "cpu"):
        audio = audio.cpu().numpy()
    out = np.asarray(audio, dtype=np.float32).reshape(-1)
    raw_dur = out.shape[0] / max(int(sr), 1)
    # Light edge trim only — strip leading/trailing silence so the wav
    # starts/ends crisply for concat. Internal pauses (commas, sentence
    # boundaries) MUST be preserved or chunks come out missing words —
    # the earlier VAD-cut at first 250ms gap after 2s was truncating
    # whole second-halves of sentences (Whisper round-trip showed 31%
    # word-error-rate). Trust the EOS sampling now that we use the
    # official generation_config.json defaults; cap below is a safety
    # net not a forced truncation point.
    try:
        import librosa  # type: ignore
        trimmed, _ = librosa.effects.trim(out, top_db=35, frame_length=2048, hop_length=512)
        if trimmed.shape[0] >= int(0.3 * sr):
            out = trimmed
    except Exception as e:
        log.warning(f"[tts.synth.trim.skip] {e}")
    # Optional aggressive VAD-cut, off by default. Set
    # QWEN3_TTS_VAD_CUT=1 to re-enable for runs where you'd rather lose
    # tail content than risk hallucinated overrun.
    if os.environ.get("QWEN3_TTS_VAD_CUT", "0") == "1":
        try:
            win = max(1, int(sr) // 50)
            n = (out.shape[0] // win) * win
            rms = np.sqrt(np.mean(out[:n].reshape(-1, win) ** 2, axis=1))
            thresh = 10 ** (-32 / 20)
            silent = rms < thresh
            min_run = int(os.environ.get("QWEN3_TTS_VAD_MIN_GAP_MS", "600")) // 20
            skip_frames = int(float(os.environ.get("QWEN3_TTS_VAD_SKIP_S", "3")) * sr / win)
            run = 0; cut_at = None
            for idx in range(skip_frames, len(silent)):
                if silent[idx]:
                    run += 1
                    if run >= min_run:
                        cut_at = (idx - run + 1) * win
                        break
                else:
                    run = 0
            if cut_at is not None and cut_at >= skip_frames * win:
                out = out[:cut_at]
                log.info(f"[tts.synth.vad_cut] cut at "
                         f"{out.shape[0]/sr:.2f}s (had {raw_dur:.2f}s)")
        except Exception as e:
            log.warning(f"[tts.synth.vad.skip] {e}")
    log.info(f"[tts.synth.done] samples={out.shape[0]} sr={int(sr)} "
             f"duration_s={out.shape[0]/max(int(sr),1):.2f} raw_s={raw_dur:.2f}")
    return out, int(sr)


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest) -> Response:
    touch_activity()
    if not req.input.strip():
        raise HTTPException(400, "input must be non-empty")
    if model is None:
        raise HTTPException(503, f"model not ready: {load_error or 'still loading'}")
    log.info(f"[tts.req.speech] voice={req.voice} fmt={req.response_format} "
             f"text_len={len(req.input)}")
    async with inference_lock:
        try:
            audio, sr = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, _synth, req.input, req.voice, req.speed,
                    None, None, None, req.language,
                ),
                timeout=TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            log.error(f"[tts.req.timeout] timeout={TIMEOUT_S}s")
            raise HTTPException(504, f"synthesis timed out after {TIMEOUT_S}s")
        except Exception as exc:
            import traceback
            log.error(f"[tts.req.fail] {exc}\n{traceback.format_exc()}")
            raise HTTPException(500, f"synthesis failed: {exc}")
    body, ctype = _encode(audio, req.response_format, sr)
    log.info(f"[tts.req.resp] bytes={len(body)} ctype={ctype}")
    return Response(content=body, media_type=ctype)


@app.post("/v1/audio/speech/clone")
async def speech_clone(
    text: str = Form(...),
    reference_audio: UploadFile = File(...),
    ref_text: str = Form(""),
    response_format: str = Form("wav"),
    speed: float = Form(1.0),
    language: Optional[str] = Form(None),
) -> Response:
    """Voice-cloned synthesis. `reference_audio` is a 5-30s clip (any sample
    rate; we resample to 24 kHz mono internally). `ref_text` is the literal
    transcript of that clip — required by Qwen3-TTS-Base for prosody
    conditioning. Returns the synthesized audio in the requested format."""
    touch_activity()
    if not text.strip():
        raise HTTPException(400, "text must be non-empty")
    if model is None:
        raise HTTPException(503, f"model not ready: {load_error or 'still loading'}")

    log.info(f"[tts.req.clone] text_len={len(text)} ref_text_len={len(ref_text)} "
             f"fmt={response_format}")
    raw = await reference_audio.read()
    log.info(f"[tts.req.clone.read] ref_bytes={len(raw)}")
    try:
        ref_audio, ref_sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception as exc:
        raise HTTPException(400, f"could not decode reference_audio: {exc}")
    if ref_audio.ndim > 1:
        ref_audio = ref_audio.mean(axis=1)
    log.info(f"[tts.req.clone.decoded] ref_samples={ref_audio.shape[0]} ref_sr={int(ref_sr)} "
             f"ref_dur={ref_audio.shape[0]/max(int(ref_sr),1):.2f}s")

    async with inference_lock:
        try:
            audio, sr = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, _synth, text, "Ryan", speed,
                    ref_audio, int(ref_sr), ref_text, language,
                ),
                timeout=TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            log.error(f"[tts.req.clone.timeout] timeout={TIMEOUT_S}s")
            raise HTTPException(504, f"voice-clone timed out after {TIMEOUT_S}s")
        except Exception as exc:
            import traceback
            log.error(f"[tts.req.clone.fail] {exc}\n{traceback.format_exc()}")
            raise HTTPException(500, f"voice-clone synthesis failed: {exc}")

    body, ctype = _encode(audio, response_format, sr)
    log.info(f"[tts.req.clone.resp] bytes={len(body)} ctype={ctype}")
    return Response(content=body, media_type=ctype)


# ── Cached voice clone (skip per-request speaker embedding) ────────────────

class CachedSpeechRequest(SpeechRequest):
    """Reuses a precomputed voice clone prompt registered under `voice`.
    Falls back to 503 if the name isn't in VOICE_CACHE."""
    pass  # `voice` field already exists on SpeechRequest


@app.post("/v1/audio/speech/cached")
async def speech_cached(req: CachedSpeechRequest) -> Response:
    """Voice-cloned synthesis using a previously-registered prompt. Drops
    the per-request reference-audio upload + speaker-embedding extraction
    (~1-3s GPU on a single L4) compared to /v1/audio/speech/clone."""
    touch_activity()
    if not req.input.strip():
        raise HTTPException(400, "input must be non-empty")
    if model is None:
        raise HTTPException(503, f"model not ready: {load_error or 'still loading'}")
    cached = VOICE_CACHE.get(req.voice)
    if cached is None:
        raise HTTPException(
            404,
            f"voice '{req.voice}' not registered. POST /v1/voices/register first "
            f"or set QWEN3_TTS_AUTO_REGISTER_VOICE on container start.",
        )
    log.info(f"[tts.req.cached] voice={req.voice} fmt={req.response_format} "
             f"text_len={len(req.input)}")
    async with inference_lock:
        try:
            audio, sr = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, _synth, req.input, req.voice, req.speed,
                    None, None, None, req.language, cached,
                ),
                timeout=TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            log.error(f"[tts.req.cached.timeout] timeout={TIMEOUT_S}s")
            raise HTTPException(504, f"cached-clone timed out after {TIMEOUT_S}s")
        except Exception as exc:
            import traceback
            log.error(f"[tts.req.cached.fail] {exc}\n{traceback.format_exc()}")
            raise HTTPException(500, f"cached-clone synthesis failed: {exc}")
    body, ctype = _encode(audio, req.response_format, sr)
    log.info(f"[tts.req.cached.resp] bytes={len(body)} ctype={ctype}")
    return Response(content=body, media_type=ctype)


@app.post("/v1/voices/register")
async def voices_register(
    name: str = Form(...),
    reference_audio: UploadFile = File(...),
    ref_text: str = Form(""),
) -> dict:
    """Register a voice clone prompt under `name`. Subsequent calls to
    /v1/audio/speech/cached with `voice=<name>` reuse the cached prompt."""
    touch_activity()
    if not name.strip():
        raise HTTPException(400, "name required")
    if model is None:
        raise HTTPException(503, f"model not ready: {load_error or 'still loading'}")
    raw = await reference_audio.read()
    try:
        ref_audio, ref_sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception as exc:
        raise HTTPException(400, f"could not decode reference_audio: {exc}")
    if ref_audio.ndim > 1:
        ref_audio = ref_audio.mean(axis=1)
    async with inference_lock:
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, _register_voice_sync, name, ref_audio, int(ref_sr), ref_text,
            )
        except Exception as exc:
            import traceback
            log.error(f"[tts.voice.register.fail] {exc}\n{traceback.format_exc()}")
            raise HTTPException(500, f"register failed: {exc}")
    return {"name": name, "registered": True, "cache_size": len(VOICE_CACHE)}


@app.get("/v1/voices/cached")
async def voices_cached_list() -> dict:
    return {"voices": sorted(VOICE_CACHE.keys()), "count": len(VOICE_CACHE)}


@app.get("/v1/demo", response_class=HTMLResponse)
async def demo() -> str:
    return """<!doctype html><html><body style="font-family:sans-serif;max-width:560px;margin:40px auto">
<h2>Qwen3-TTS — quick demo</h2>
<form method="post" action="/v1/audio/speech" enctype="application/json">
  <textarea name="input" rows="3" cols="60" placeholder="Texto..."></textarea><br>
  <select name="voice"><option>Ryan</option><option>Vivian</option><option>Cherry</option></select>
  <button>Sintetizar</button>
</form>
<h3>Voice cloning (multipart):</h3>
<form method="post" action="/v1/audio/speech/clone" enctype="multipart/form-data">
  <input type="text" name="text" placeholder="Texto a sintetizar" size=60><br><br>
  <input type="text" name="ref_text" placeholder="Transcrição da reference_audio" size=60><br><br>
  <input type="file" name="reference_audio" accept="audio/*"><br><br>
  <button>Clonar voz</button>
</form>
</body></html>"""


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        log_level="info",
    )
