"""Kokoro TTS 82M — FastAPI server.

OpenAI-compatible /v1/audio/speech endpoint.
54 voices, 9 languages, 24kHz output.

Voices verified against hexgrad/Kokoro-82M HuggingFace repo.
"""

import io
import logging
import time

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("kokoro-server")

app = FastAPI(title="Kokoro TTS")

# Voice prefix → lang_code for KPipeline
LANG_MAP = {
    "a": "a", "b": "b", "e": "e", "f": "f",
    "h": "h", "i": "i", "j": "j", "p": "p", "z": "z",
}

# Actual voices in hexgrad/Kokoro-82M (verified against HuggingFace repo)
ALL_VOICES = {
    "en-US": [
        "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
        "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
        "am_michael", "am_onyx", "am_puck", "am_santa",
    ],
    "en-GB": [
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    ],
    "pt-BR": ["pf_dora", "pm_alex", "pm_santa"],
    "es":    ["ef_dora", "em_alex", "em_santa"],
    "fr":    ["ff_siwis"],
    "it":    ["if_sara", "im_nicola"],
    "ja":    ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"],
    "zh":    [
        "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
        "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    ],
    "hi":    ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"],
}

VALID_VOICE_IDS = {v for voices in ALL_VOICES.values() for v in voices}

# Lazy-loaded pipelines per language
_pipelines: dict = {}


def _detect_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_pipeline(lang_code: str):
    if lang_code not in _pipelines:
        from kokoro import KPipeline
        device = _detect_device()
        log.info(f"Loading KPipeline lang_code={lang_code} device={device}")
        try:
            _pipelines[lang_code] = KPipeline(lang_code=lang_code, device=device)
        except Exception:
            log.warning(f"Failed to load lang_code={lang_code}, falling back to 'a' (English)")
            if "a" not in _pipelines:
                _pipelines["a"] = KPipeline(lang_code="a", device=device)
            _pipelines[lang_code] = _pipelines["a"]
        log.info(f"KPipeline lang_code={lang_code} ready")
    return _pipelines[lang_code]


def voice_to_lang(voice: str) -> str:
    if voice and len(voice) >= 2 and voice[0] in LANG_MAP:
        return LANG_MAP[voice[0]]
    return "a"


@app.on_event("startup")
async def startup():
    log.info("Pre-warming English pipeline...")
    get_pipeline("a")
    log.info("Kokoro TTS ready")


class SpeechRequest(BaseModel):
    model: str = "kokoro-82m"
    input: str
    voice: str = "af_heart"
    speed: float = 1.0
    response_format: str = "wav"


@app.get("/health")
async def health():
    return {"status": "ok", "model": "kokoro-82m", "pipelines_loaded": list(_pipelines.keys())}


@app.get("/v1/audio/voices")
async def list_voices():
    voices = []
    for lang, vlist in ALL_VOICES.items():
        for v in vlist:
            voices.append({"id": v, "language": lang})
    return {"voices": voices}


@app.post("/v1/audio/speech")
async def synthesize(req: SpeechRequest):
    if not req.input or not req.input.strip():
        raise HTTPException(status_code=400, detail="input text is empty")

    voice = req.voice.lower()
    if voice not in VALID_VOICE_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{req.voice}'. Use GET /v1/audio/voices for available voices.",
        )

    t0 = time.time()
    lang_code = voice_to_lang(voice)
    pipeline = get_pipeline(lang_code)

    chunks = []
    for _gs, _ps, audio in pipeline(req.input, voice=voice, speed=req.speed):
        if audio is not None:
            chunks.append(audio)

    if not chunks:
        raise HTTPException(status_code=500, detail="No audio generated")

    audio_np = np.concatenate(chunks)
    elapsed = time.time() - t0
    duration = len(audio_np) / 24000
    log.info(f"Generated {duration:.1f}s in {elapsed * 1000:.0f}ms (voice={voice}, {duration / elapsed:.0f}x RT)")

    buf = io.BytesIO()
    fmt = req.response_format.lower()
    if fmt == "flac":
        sf.write(buf, audio_np, 24000, format="FLAC")
        ct = "audio/flac"
    elif fmt == "opus":
        sf.write(buf, audio_np, 24000, format="OGG", subtype="OPUS")
        ct = "audio/ogg"
    else:
        sf.write(buf, audio_np, 24000, format="WAV", subtype="PCM_16")
        ct = "audio/wav"

    return Response(content=buf.getvalue(), media_type=ct)
