#!/usr/bin/env python3
"""Ditto chat server — full VAD UI w/ idle loop + reply crossfade.

Mirrors test-realtime-video/scripts/chat_server_local.py UI but ditto backend.
Pipeline:
  Browser WS ↔ this server (pod:8000)
     ↓
     Groq Whisper STT  → emit {transcript}
     Groq Llama 70B    → emit {reply}
     Groq Orpheus TTS  → emit {tts_ready}
     Ditto inference   → emit {video, url}
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import time
import uuid
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response
from groq import Groq

DITTO_ROOT = Path("/root/ditto-talkinghead")
OUT_DIR = Path("/tmp/ditto_chat_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
AVATAR_PATH = os.environ.get("AVATAR_PATH", "/root/charlie_ref.mp4")
AVATAR_NAME = os.environ.get("AVATAR_NAME", "Charlie")

STT_MODEL = "whisper-large-v3-turbo"
LLM_MODEL = "llama-3.3-70b-versatile"
TTS_MODEL = "canopylabs/orpheus-v1-english"
TTS_VOICE = "austin"
SYSTEM_PROMPT = (
    f"You are a friendly English-speaking AI avatar in a spoken conversation. "
    f"Your name is {AVATAR_NAME} — ONLY mention it if someone explicitly asks your name. "
    f"Never start replies with '{AVATAR_NAME}' or any self-introduction unless the user asked. "
    "ALWAYS reply in American English regardless of input language. "
    "Keep replies VERY short — 1 sentence, max 20 words. No bullets/lists."
)
HISTORY_TURNS = 10

app = FastAPI()
groq_client = Groq(api_key=GROQ_API_KEY)

_sdk = None
_sdk_lock = asyncio.Lock()


def init_ditto():
    global _sdk
    import sys
    sys.path.insert(0, str(DITTO_ROOT))
    from stream_pipeline_offline import StreamSDK
    cfg_pkl = str(DITTO_ROOT / "checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl")
    data_root = str(DITTO_ROOT / "checkpoints/ditto_trt_Ampere_Plus")
    print(f"[init] loading Ditto SDK ...", flush=True)
    t0 = time.time()
    _sdk = StreamSDK(cfg_pkl, data_root)
    print(f"[init] Ditto loaded in {time.time()-t0:.1f}s", flush=True)


def _warmup_ditto(n: int = 3) -> None:
    """Run a few dummy renders so first real reply is fast (avatar cached, kernels JIT'd)."""
    import tempfile
    print(f"[warmup] running {n} dummy renders ...", flush=True)
    sr = 16000
    sil = np.zeros(int(sr * 0.8), dtype=np.float32)
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "warm.wav"
        sf.write(str(wav), sil, sr)
        for i in range(n):
            t0 = time.time()
            try:
                ditto_render(wav, Path(td) / f"warm_{i}.mp4")
                print(f"[warmup] {i+1}/{n} {time.time()-t0:.2f}s", flush=True)
            except Exception as e:
                print(f"[warmup] {i+1} failed: {e!r}", flush=True)
                break
    print("[warmup] done", flush=True)


@app.on_event("startup")
async def _startup():
    await asyncio.to_thread(init_ditto)
    await asyncio.to_thread(_load_body_track)
    asyncio.create_task(asyncio.to_thread(_warmup_ditto, 3))


def stt_from_audio(audio_bytes: bytes, filename: str = "in.wav") -> str:
    resp = groq_client.audio.transcriptions.create(
        file=(filename, audio_bytes), model=STT_MODEL, response_format="text",
    )
    return str(resp).strip()


def llm_reply(user_text: str, history: list | None = None) -> str:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        msgs.extend(history[-(HISTORY_TURNS * 2):])
    msgs.append({"role": "user", "content": user_text})
    r = groq_client.chat.completions.create(
        model=LLM_MODEL, messages=msgs, max_tokens=60, temperature=0.7,
    )
    return r.choices[0].message.content.strip()


def tts_groq(text: str, out_path: Path) -> None:
    text = text[:200]
    resp = groq_client.audio.speech.create(
        model=TTS_MODEL, voice=TTS_VOICE, input=text, response_format="wav",
    )
    resp.write_to_file(str(out_path))


_avatar_cache = {}
_registrar_patched = False

# Live-tunable composite params (modified via /apply, used by composite_overlay)
_compose_params = {
    "color_match": True,         # Reinhard LAB skin-tone transfer
    "color_strength": 1.0,       # 0=off, 1=full
    "brightness": 0,             # -100..100 added to L channel
    "contrast": 1.0,             # 0.5..1.5 multiplier on L
    "warp_enabled": True,        # similarity transform vs simple resize+paste
    "mask_inner": 0.85,          # 0..1
    "mask_outer": 1.0,           # 0..1
    "mask_rx": 0.42,             # ellipse rx as fraction of size
    "mask_ry": 0.55,             # ellipse ry as fraction of size
    "mask_cy": 0.48,             # vertical center as fraction
    "track_body_motion": True,   # follow body face per-frame; off=fixed at frame 0
    # Motion stability knobs (auto-tunable):
    "dpts_ema": 0.18,            # ditto landmarks EMA (lower=smoother)
    "M_ema": 0.20,               # transform matrix EMA (lower=smoother)
    "body_smooth_window": 7,     # body landmarks moving-avg window (1=off)
    "ditto_lock_head": True,     # if true, Ditto generates with use_d_keys=("exp",) only
}

# ── Server-side compositing w/ MediaPipe face landmarks (precise warp) ──────
BODY_BG = Path("/root/charlie_body_bg.mp4")
# MediaPipe FaceMesh landmark indices used for alignment:
#   33 = right eye outer, 263 = left eye outer, 1 = nose tip,
#   61 = right mouth corner, 291 = left mouth corner,
#   234 = right cheek (face contour), 454 = left cheek (face contour)
MP_KEY_IDX = [33, 263, 1, 61, 291, 234, 454]

_body_frames = None          # np.ndarray (N, H, W, 3) uint8 BGR
_body_pts = None             # list of np.ndarray per body frame OR None (smoothed)
_body_pts_raw = None         # list of raw landmarks before smoothing
_face_mask_cache = {}        # head_size -> alpha mask
_mp_face = None              # MediaPipe FaceMesh instance


def _smooth_body_pts(raw, win):
    import numpy as _np
    if win is None or win <= 1:
        return list(raw)
    out = []
    for i, p in enumerate(raw):
        if p is None:
            out.append(None); continue
        w = [raw[j] for j in range(max(0, i - win // 2), min(len(raw), i + win // 2 + 1)) if raw[j] is not None]
        out.append(_np.mean(w, axis=0).astype(_np.float32))
    return out


def _get_mp_face():
    global _mp_face
    if _mp_face is None:
        from mediapipe.tasks import python as mp_py
        from mediapipe.tasks.python import vision as mp_vision
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_py.BaseOptions(model_asset_path="/root/face_landmarker.task"),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
        )
        _mp_face = mp_vision.FaceLandmarker.create_from_options(opts)
    return _mp_face


def _detect_pts(bgr_frame):
    """Return (5,2) numpy array of key landmark pixel coords, or None."""
    import cv2 as _cv2
    import numpy as _np
    import mediapipe as mp
    rgb = _cv2.cvtColor(bgr_frame, _cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = _get_mp_face().detect(mp_img)
    if not res.face_landmarks:
        return None
    lm = res.face_landmarks[0]
    H, W = bgr_frame.shape[:2]
    pts = _np.array([[lm[i].x * W, lm[i].y * H] for i in MP_KEY_IDX], dtype=_np.float32)
    return pts


def _color_match_lab(src_bgr, ref_bgr, src_pts, ref_pts):
    """Reinhard color transfer: match mean+std of src face skin region to ref. Returns adjusted src."""
    import cv2 as _cv2
    import numpy as _np
    def skin_box(pts, frame):
        # Box around eyes-to-mouth area (skin patch)
        xs, ys = pts[:5, 0], pts[:5, 1]
        x0, y0 = int(max(0, xs.min() - 5)), int(max(0, ys.min() - 5))
        x1, y1 = int(min(frame.shape[1], xs.max() + 5)), int(min(frame.shape[0], ys.max() + 5))
        return frame[y0:y1, x0:x1]
    src_patch = skin_box(src_pts, src_bgr)
    ref_patch = skin_box(ref_pts, ref_bgr)
    if src_patch.size == 0 or ref_patch.size == 0:
        return src_bgr
    src_lab = _cv2.cvtColor(src_patch, _cv2.COLOR_BGR2LAB).astype(_np.float32)
    ref_lab = _cv2.cvtColor(ref_patch, _cv2.COLOR_BGR2LAB).astype(_np.float32)
    sm, ss = src_lab.reshape(-1, 3).mean(0), src_lab.reshape(-1, 3).std(0) + 1e-6
    rm, rs = ref_lab.reshape(-1, 3).mean(0), ref_lab.reshape(-1, 3).std(0) + 1e-6
    full_lab = _cv2.cvtColor(src_bgr, _cv2.COLOR_BGR2LAB).astype(_np.float32)
    adj = (full_lab - sm) * (rs / ss) + rm
    adj = _np.clip(adj, 0, 255).astype(_np.uint8)
    return _cv2.cvtColor(adj, _cv2.COLOR_LAB2BGR)


def _build_alpha_mask(size: int):
    p = _compose_params
    key = (size, p["mask_inner"], p["mask_outer"], p["mask_rx"], p["mask_ry"], p["mask_cy"])
    if key in _face_mask_cache:
        return _face_mask_cache[key]
    import numpy as _np
    yy, xx = _np.ogrid[:size, :size]
    cy, cx = size * p["mask_cy"], size * 0.5
    rx, ry = size * p["mask_rx"], size * p["mask_ry"]
    norm = _np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    m = _np.clip(255 * (p["mask_outer"] - norm) / max(1e-6, p["mask_outer"] - p["mask_inner"]),
                 0, 255).astype(_np.uint8)
    _face_mask_cache[key] = m
    return m


def _load_body_track():
    """Pre-detect face landmarks per body_bg frame + smooth them temporally."""
    global _body_frames, _body_pts, _body_pts_raw
    if _body_frames is not None:
        return
    import cv2 as _cv2
    import numpy as _np
    cap = _cv2.VideoCapture(str(BODY_BG))
    frames, raw_pts, last = [], [], None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
        p = _detect_pts(f)
        if p is not None:
            last = p
        raw_pts.append(last)
    cap.release()
    _body_frames = _np.stack(frames, axis=0)
    _body_pts_raw = raw_pts
    _body_pts = _smooth_body_pts(raw_pts, _compose_params.get("body_smooth_window", 7))
    valid = sum(1 for p in _body_pts if p is not None)
    print(f"[track] body_bg: {_body_frames.shape} frames, {valid}/{len(_body_pts)} landmarks", flush=True)


def composite_overlay(ditto_mp4: Path, out_mp4: Path) -> None:
    """Warp ditto frame to match body face pose via similarity transform on landmarks."""
    if not BODY_BG.exists():
        os.system(f'cp "{ditto_mp4}" "{out_mp4}"')
        return
    if _body_frames is None:
        _load_body_track()
    import cv2 as _cv2
    import subprocess as _sp
    import numpy as _np

    cap = _cv2.VideoCapture(str(ditto_mp4))
    fps = cap.get(_cv2.CAP_PROP_FPS) or 25.0
    H_bg, W_bg = _body_frames.shape[1:3]
    DITTO_W = 512

    proc = _sp.Popen([
        "ffmpeg", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{W_bg}x{H_bg}", "-r", str(fps), "-i", "pipe:0",
        "-i", str(ditto_mp4),
        "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "copy", "-shortest", "-movflags", "+faststart",
        str(out_mp4),
    ], stdin=_sp.PIPE)

    global _body_pts
    P = _compose_params
    # Re-smooth body_pts if window changed
    cur_win = P.get("body_smooth_window", 7)
    if _body_pts_raw is not None:
        _body_pts = _smooth_body_pts(_body_pts_raw, cur_win)
    body_n = len(_body_pts)

    # Pre-compute body face center+scale via cheek-to-cheek (more stable than interocular)
    body_geom = []
    for p in _body_pts:
        if p is None:
            body_geom.append(None); continue
        cx = float(p[:5].mean(axis=0)[0])
        cy = float(p[:5].mean(axis=0)[1])
        face_w = float(_np.linalg.norm(p[6] - p[5]))
        body_geom.append((cx, cy, face_w))
    fi = 0
    last_dpts = None
    smoothed_dpts = None
    EMA = float(P.get("dpts_ema", 0.18))
    color_lut = None
    M_smooth = None
    M_EMA = float(P.get("M_ema", 0.20))
    while True:
        ok, dframe = cap.read()
        if not ok:
            break
        bg_idx = fi % body_n
        bg = _body_frames[bg_idx]
        bpts = _body_pts[bg_idx]
        if not P["track_body_motion"]:
            # Lock to first valid body landmarks → no movement
            for fp in _body_pts:
                if fp is not None: bpts = fp; break
        dpts = _detect_pts(dframe)
        if dpts is None:
            dpts = last_dpts
        else:
            last_dpts = dpts
        # Temporal smoothing of ditto landmarks → kills "gelatinous" wobble from per-frame noise
        if dpts is not None:
            if smoothed_dpts is None:
                smoothed_dpts = dpts.copy()
            else:
                smoothed_dpts = EMA * dpts + (1 - EMA) * smoothed_dpts
            dpts = smoothed_dpts
        if bpts is not None and dpts is not None:
            # Color match (Reinhard LAB), once per render, with strength + bright/contrast
            if color_lut is None and P["color_match"]:
                def _box(pts, frame):
                    xs, ys = pts[:5, 0], pts[:5, 1]
                    return frame[max(0, int(ys.min())):min(frame.shape[0], int(ys.max())),
                                 max(0, int(xs.min())):min(frame.shape[1], int(xs.max()))]
                sp = _box(dpts, dframe); rp = _box(bpts, bg)
                if sp.size and rp.size:
                    sl = _cv2.cvtColor(sp, _cv2.COLOR_BGR2LAB).astype(_np.float32).reshape(-1, 3)
                    rl = _cv2.cvtColor(rp, _cv2.COLOR_BGR2LAB).astype(_np.float32).reshape(-1, 3)
                    color_lut = (sl.mean(0), sl.std(0) + 1e-6, rl.mean(0), rl.std(0) + 1e-6)
            if color_lut is not None and P["color_match"]:
                sm, ss, rm, rs = color_lut
                lab = _cv2.cvtColor(dframe, _cv2.COLOR_BGR2LAB).astype(_np.float32)
                target = (lab - sm) * (rs / ss) + rm
                lab = lab + P["color_strength"] * (target - lab)
                # Brightness + contrast on L channel
                lab[..., 0] = lab[..., 0] * P["contrast"] + P["brightness"]
                dframe = _cv2.cvtColor(_np.clip(lab, 0, 255).astype(_np.uint8), _cv2.COLOR_LAB2BGR)
            elif P["brightness"] != 0 or P["contrast"] != 1.0:
                lab = _cv2.cvtColor(dframe, _cv2.COLOR_BGR2LAB).astype(_np.float32)
                lab[..., 0] = lab[..., 0] * P["contrast"] + P["brightness"]
                dframe = _cv2.cvtColor(_np.clip(lab, 0, 255).astype(_np.uint8), _cv2.COLOR_LAB2BGR)
            if P["warp_enabled"]:
                M, _ = _cv2.estimateAffinePartial2D(dpts, bpts, method=_cv2.LMEDS)
                if M is not None:
                    if M_smooth is None:
                        M_smooth = M
                    else:
                        M_smooth = M_EMA * M + (1 - M_EMA) * M_smooth
                    Mu = M_smooth
                    bg = bg.copy()
                    warped = _cv2.warpAffine(dframe, Mu, (W_bg, H_bg),
                                              flags=_cv2.INTER_LINEAR, borderMode=_cv2.BORDER_CONSTANT)
                    d_alpha = _build_alpha_mask(DITTO_W)
                    warped_alpha = _cv2.warpAffine(d_alpha, Mu, (W_bg, H_bg), flags=_cv2.INTER_LINEAR)
                    a = warped_alpha.astype(_np.float32)[..., None] / 255.0
                    bg = (a * warped.astype(_np.float32) + (1 - a) * bg.astype(_np.float32)).astype(_np.uint8)
                else:
                    bg = bg.copy()
            else:
                # Fast path: resize+paste, scale via cheek-to-cheek
                cx_b = float(bpts[:5].mean(axis=0)[0]); cy_b = float(bpts[:5].mean(axis=0)[1])
                fw_b = float(_np.linalg.norm(bpts[6] - bpts[5]))
                cx_d = float(dpts[:5].mean(axis=0)[0]); cy_d = float(dpts[:5].mean(axis=0)[1])
                fw_d = float(_np.linalg.norm(dpts[6] - dpts[5]))
                scale = fw_b / max(1e-6, fw_d)
                hw = max(1, int(round(DITTO_W * scale)))
                dr = _cv2.resize(dframe, (hw, hw), interpolation=_cv2.INTER_LINEAR)
                tx = int(round(cx_b - cx_d * scale)); ty = int(round(cy_b - cy_d * scale))
                x0, y0 = max(0, tx), max(0, ty)
                x1, y1 = min(W_bg, tx + hw), min(H_bg, ty + hw)
                sx0, sy0 = x0 - tx, y0 - ty
                sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)
                bg = bg.copy()
                if x1 > x0 and y1 > y0:
                    m = _build_alpha_mask(hw)[sy0:sy1, sx0:sx1]
                    a = m.astype(_np.float32)[..., None] / 255.0
                    bg[y0:y1, x0:x1] = (a * dr[sy0:sy1, sx0:sx1].astype(_np.float32)
                                        + (1 - a) * bg[y0:y1, x0:x1].astype(_np.float32)).astype(_np.uint8)
        else:
            bg = bg.copy()
        proc.stdin.write(bg.tobytes())
        fi += 1
    cap.release()
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0 or not out_mp4.exists():
        raise RuntimeError(f"composite ffmpeg rc={rc}")


class _CachedRegistrar:
    def __init__(self, real, cache_key):
        self._real = real
        self._key = cache_key
    def __call__(self, source_path, **kw):
        if source_path == self._key and self._key in _avatar_cache:
            return _avatar_cache[self._key]
        r = self._real(source_path, **kw)
        if source_path == self._key:
            _avatar_cache[self._key] = r
        return r
    def __getattr__(self, name):
        return getattr(self._real, name)


def ditto_render(audio_path: Path, out_path: Path) -> None:
    global _registrar_patched
    assert _sdk is not None
    if not _registrar_patched:
        _sdk.avatar_registrar = _CachedRegistrar(_sdk.avatar_registrar, AVATAR_PATH)
        _registrar_patched = True
    keys = ("exp",) if _compose_params.get("ditto_lock_head", True) else ("exp", "pitch", "yaw", "roll", "t")
    _sdk.setup(AVATAR_PATH, str(out_path), use_d_keys=keys)
    audio, sr = librosa.core.load(str(audio_path), sr=16000)
    num_f = math.ceil(len(audio) / 16000 * 25)
    _sdk.setup_Nd(N_d=num_f, fade_in=-1, fade_out=-1, ctrl_info={})
    aud_feat = _sdk.wav2feat.wav2feat(audio)
    _sdk.audio2motion_queue.put(aud_feat)
    _sdk.close()
    cmd = (f'ffmpeg -loglevel error -y -i "{_sdk.tmp_output_path}" -i "{audio_path}" '
           f'-map 0:v -map 1:a -c:v copy -c:a aac "{out_path}"')
    rc = os.system(cmd)
    if rc != 0 or not out_path.exists():
        raise RuntimeError(f"ffmpeg mux failed (rc={rc})")


async def pipeline(ws: WebSocket, req_id: str, text: str, audio_bytes: bytes | None, history: list) -> None:
    t_start = time.time()
    timings: dict = {}

    if audio_bytes and not text:
        t0 = time.time()
        text = await asyncio.to_thread(stt_from_audio, audio_bytes)
        timings["stt_s"] = round(time.time() - t0, 2)
        await ws.send_json({"type": "transcript", "text": text, "elapsed_s": round(time.time() - t_start, 2)})

    if not text:
        await ws.send_json({"type": "error", "message": "empty input"})
        return

    t0 = time.time()
    reply = await asyncio.to_thread(llm_reply, text, history)
    timings["llm_s"] = round(time.time() - t0, 2)
    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": reply})
    if len(history) > HISTORY_TURNS * 2:
        del history[: len(history) - HISTORY_TURNS * 2]
    await ws.send_json({"type": "reply", "text": reply, "elapsed_s": round(time.time() - t_start, 2)})

    t0 = time.time()
    wav_path = OUT_DIR / f"{req_id}.wav"
    await asyncio.to_thread(tts_groq, reply, wav_path)
    timings["tts_s"] = round(time.time() - t0, 2)
    await ws.send_json({"type": "tts_ready", "elapsed_s": round(time.time() - t_start, 2)})

    t0 = time.time()
    raw_mp4 = OUT_DIR / f"{req_id}_raw.mp4"
    mp4_path = OUT_DIR / f"{req_id}.mp4"
    async with _sdk_lock:
        await asyncio.to_thread(ditto_render, wav_path, raw_mp4)
    timings["ditto_s"] = round(time.time() - t0, 2)

    t0 = time.time()
    await asyncio.to_thread(composite_overlay, raw_mp4, mp4_path)
    timings["compose_s"] = round(time.time() - t0, 2)

    timings["total_s"] = round(time.time() - t_start, 2)
    await ws.send_json({"type": "video", "url": f"/out/{req_id}.mp4", "timings": timings})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    history: list = []
    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                req_id = uuid.uuid4().hex[:8]
                asyncio.create_task(pipeline(ws, req_id, "", msg["bytes"], history))
            elif "text" in msg and msg["text"] is not None:
                data = json.loads(msg["text"])
                if data.get("type") == "text":
                    req_id = uuid.uuid4().hex[:8]
                    asyncio.create_task(pipeline(ws, req_id, data.get("text", ""), None, history))
                elif data.get("type") == "clear":
                    history.clear()
                    await ws.send_json({"type": "cleared"})
                elif data.get("type") == "ping":
                    await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass


@app.get("/out/{name}")
def serve_out(name: str):
    p = OUT_DIR / name
    if not p.exists():
        return Response(status_code=404)
    return FileResponse(p)


@app.get("/idle.mp4")
def serve_idle():
    p = Path(AVATAR_PATH)
    if not p.exists():
        return Response(status_code=404)
    return FileResponse(p, media_type="video/mp4")


@app.get("/body.jpg")
def serve_body():
    p = Path("/root/charlie_body.jpg")
    if not p.exists():
        return Response(status_code=404)
    return FileResponse(p, media_type="image/jpeg")


@app.get("/body_bg.mp4")
def serve_body_bg():
    p = Path("/root/charlie_body_bg.mp4")
    if not p.exists():
        return Response(status_code=404)
    return FileResponse(p, media_type="video/mp4")


SAMPLE_RAW = OUT_DIR / "sample_raw.mp4"
SAMPLE_OUT = OUT_DIR / "sample.mp4"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
VISION_MODEL = "anthropic/claude-sonnet-4.6"


def _b64_frame(frame_path: Path) -> str:
    import base64
    return base64.b64encode(frame_path.read_bytes()).decode()


def _eval_composite_via_llm(mp4_path: Path, n_frames: int = 4) -> dict:
    """Sample frames from mp4, send to vision LLM, return {score, defects}."""
    import httpx, tempfile, json as _json
    if not OPENROUTER_API_KEY:
        return {"score": 0, "defects": ["no api key"], "error": "OPENROUTER_API_KEY missing"}
    with tempfile.TemporaryDirectory() as td:
        # Sample n_frames evenly
        dur_str = os.popen(f'ffprobe -v error -show_entries format=duration -of csv=p=0 "{mp4_path}"').read().strip()
        try: dur = float(dur_str)
        except: dur = 2.0
        frames = []
        for i in range(n_frames):
            t = (i + 0.5) * dur / n_frames
            fp = Path(td) / f"f{i}.jpg"
            os.system(f'ffmpeg -loglevel error -y -ss {t} -i "{mp4_path}" -frames:v 1 -q:v 3 -update 1 "{fp}" 2>/dev/null')
            if fp.exists() and fp.stat().st_size > 0:
                frames.append(_b64_frame(fp))
        if not frames:
            return {"score": 0, "defects": ["frame extract failed"]}
        prompt = (
            "4 frames sampled in temporal order from a composited talking-head video. "
            "An AI-generated face is overlaid onto a real body background per frame. "
            "Critical evaluation criteria (in priority order):\n"
            "1. HEAD/FACE SYNC: Does the face move in lockstep with the body's head? "
            "Does the face appear glued to the head or does it slide/drift relative to it across frames?\n"
            "2. SEAM: Visible boundary line between composited face and body neck/hair?\n"
            "3. COLOR MATCH: Skin tone/lighting same as body?\n"
            "4. PROPORTIONS: Face the right size for the body?\n"
            "5. JITTER: Does the face wobble/jitter in a way the body head does not?\n"
            'Return strict JSON: {"score": <0-10 int>, "defects": ["short tag"], "summary": "1 line"}. '
            "10 = perfectly natural; 0 = obviously broken."
        )
        content = [{"type": "text", "text": prompt}]
        for b in frames:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})
        try:
            r = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": VISION_MODEL,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 3000, "temperature": 0,
                    "reasoning": {"exclude": True},
                },
                timeout=120,
            )
            r.raise_for_status()
            jr = r.json()
            txt = (jr.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
            if not txt:
                return {"score": 0, "defects": ["empty response"], "raw": str(jr)[:200]}
            import re as _re
            # Extract first {...} JSON object via regex
            m = _re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", txt, _re.DOTALL)
            if m:
                try:
                    return _json.loads(m.group(0))
                except Exception:
                    pass
            # Fallback: try strict parse
            if txt.startswith("```"):
                txt = txt.split("\n", 1)[1].rsplit("```", 1)[0]
                if txt.startswith("json\n"): txt = txt[5:]
            return _json.loads(txt)
        except Exception as e:
            return {"score": 0, "defects": [f"llm err: {str(e)[:80]}"], "raw": str(r.text)[:200] if 'r' in dir() else ""}


def _eval_with_recommendations(mp4_path: Path) -> dict:
    """Ask LLM to score + recommend specific parameter adjustments."""
    import httpx, base64, json as _json, re as _re, tempfile
    if not OPENROUTER_API_KEY:
        return {"score": 0, "defects": ["no key"], "next_params": {}}
    cur = {k: v for k, v in _compose_params.items()}
    with tempfile.TemporaryDirectory() as td:
        dur_str = os.popen(f'ffprobe -v error -show_entries format=duration -of csv=p=0 "{mp4_path}"').read().strip()
        try: dur = float(dur_str)
        except: dur = 2.0
        frames = []
        for i in range(4):
            t = (i + 0.5) * dur / 4
            fp = Path(td) / f"f{i}.jpg"
            os.system(f'ffmpeg -loglevel error -y -ss {t} -i "{mp4_path}" -frames:v 1 -q:v 3 -update 1 "{fp}" 2>/dev/null')
            if fp.exists() and fp.stat().st_size > 0:
                frames.append(base64.b64encode(fp.read_bytes()).decode())
        if not frames:
            return {"score": 0, "defects": ["frame extract failed"], "next_params": {}}
        prompt = (
            "4 frames in temporal order from a composited talking-head video. "
            "An AI face is overlaid onto a real body bg per frame.\n"
            f"CURRENT PARAMS: {_json.dumps(cur)}\n\n"
            "PARAM GUIDE (you may propose changes):\n"
            "- color_match (bool), color_strength (0-1): skin-tone Reinhard transfer\n"
            "- brightness (-50..50), contrast (0.5-1.5): adjust face L channel\n"
            "- mask_inner (0.5-1), mask_outer (0.5-1.2): edge fade range (closer = harder edge)\n"
            "- mask_rx (0.2-0.6), mask_ry (0.2-0.7): elliptical mask radii\n"
            "- mask_cy (0.3-0.7): vertical center of mask\n"
            "- warp_enabled (bool): rotate+scale to body landmarks\n"
            "- track_body_motion (bool): follow body face per frame\n"
            "- dpts_ema (0.05-1): smooth ditto landmarks (lower = more smooth)\n"
            "- M_ema (0.05-1): smooth transform matrix\n"
            "- body_smooth_window (1-21, odd): body landmarks moving avg\n"
            "- ditto_lock_head (bool): lock ditto's own head pose\n\n"
            "Evaluate naturalness 0-10 (10=perfect). Identify top defects. "
            "Then propose SPECIFIC param adjustments to improve the worst defects. "
            "Only change params that target the observed problems. Conservative: max 3 params per round.\n"
            'Return strict JSON: {"score": int, "defects": ["..."], "summary": "...", '
            '"next_params": {"key": value, ...}, "rationale": "why these changes"}'
        )
        content = [{"type": "text", "text": prompt}]
        for b in frames:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})
        try:
            r = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
                json={"model": VISION_MODEL, "messages": [{"role": "user", "content": content}],
                      "max_tokens": 1200, "temperature": 0},
                timeout=120,
            )
            r.raise_for_status()
            txt = (r.json().get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
            m = _re.search(r"\{.*\}", txt, _re.DOTALL)
            if m:
                return _json.loads(m.group(0))
            return _json.loads(txt)
        except Exception as e:
            return {"score": 0, "defects": [f"err: {str(e)[:80]}"], "next_params": {}}


def _autotune_iterative(n_iters: int = 5):
    """Iterative loop: eval → apply LLM-recommended changes → re-eval. Track best."""
    if not SAMPLE_RAW.exists():
        return {"error": "no sample raw"}
    history = []
    best = {"score": -1, "params": dict(_compose_params), "iter": 0}
    for i in range(n_iters):
        composite_overlay(SAMPLE_RAW, SAMPLE_OUT)
        ev = _eval_with_recommendations(SAMPLE_OUT)
        score = int(ev.get("score", 0))
        rec = ev.get("next_params", {}) or {}
        history.append({"iter": i + 1, "score": score, "defects": ev.get("defects", []),
                        "summary": ev.get("summary", ""), "applied": rec, "rationale": ev.get("rationale", ""),
                        "params_before": dict(_compose_params)})
        print(f"[iter] {i+1}/{n_iters} score={score} defects={ev.get('defects')} apply={rec}", flush=True)
        if score > best["score"]:
            best = {"score": score, "params": dict(_compose_params), "iter": i + 1}
        if score >= 9 or not rec:
            break
        for k, v in rec.items():
            if k in _compose_params:
                _compose_params[k] = v
        _face_mask_cache.clear()
    # Restore best
    for k, v in best["params"].items():
        _compose_params[k] = v
    _face_mask_cache.clear()
    composite_overlay(SAMPLE_RAW, SAMPLE_OUT)
    return {"best_score": best["score"], "best_iter": best["iter"], "best_params": best["params"], "history": history}


def _autotune_search(n_trials: int = 15):
    """Random search over compose params, score via vision LLM, apply best."""
    import random, copy
    if not SAMPLE_RAW.exists():
        return {"error": "no sample raw, click Editar first"}
    grid = {
        "color_strength": [0.0, 0.5, 1.0],
        "brightness":     [-5, 0, 5],
        "contrast":       [0.96, 1.0, 1.04],
        "mask_inner":     [0.80, 0.86, 0.90],
        "mask_outer":     [0.98, 1.02],
        "mask_rx":        [0.40, 0.44],
        "mask_ry":        [0.52, 0.58],
        "mask_cy":        [0.46, 0.50],
        # Motion-stability search: head/face sync
        "dpts_ema":       [0.10, 0.20, 0.40, 1.0],
        "M_ema":          [0.10, 0.20, 0.40, 1.0],
        "body_smooth_window": [1, 5, 9, 15],
        "warp_enabled":   [True, False],
        "track_body_motion": [True, False],
    }
    base = copy.deepcopy(_compose_params)
    results = []
    rng = random.Random(42)
    for i in range(n_trials):
        cand = copy.deepcopy(base)
        for k, vs in grid.items():
            cand[k] = rng.choice(vs)
        # Apply candidate, recompose
        for k, v in cand.items():
            _compose_params[k] = v
        _face_mask_cache.clear()
        composite_overlay(SAMPLE_RAW, SAMPLE_OUT)
        ev = _eval_composite_via_llm(SAMPLE_OUT, n_frames=4)
        score = int(ev.get("score", 0))
        results.append({"trial": i + 1, "params": cand, "score": score, "defects": ev.get("defects", []), "summary": ev.get("summary", "")})
        print(f"[autotune] {i+1}/{n_trials} score={score} defects={ev.get('defects')}", flush=True)
    results.sort(key=lambda r: -r["score"])
    best = results[0]
    for k, v in best["params"].items():
        _compose_params[k] = v
    _face_mask_cache.clear()
    composite_overlay(SAMPLE_RAW, SAMPLE_OUT)
    return {"best_score": best["score"], "best_params": best["params"], "top5": results[:5], "applied": True}


@app.post("/edit/sample")
async def edit_sample():
    """Generate a 2s sample ditto raw mp4 (cached). Then composite with current params."""
    if not SAMPLE_RAW.exists():
        wav_p = OUT_DIR / "sample.wav"
        await asyncio.to_thread(tts_groq, "Hello there, how are you doing today.", wav_p)
        async with _sdk_lock:
            await asyncio.to_thread(ditto_render, wav_p, SAMPLE_RAW)
    await asyncio.to_thread(composite_overlay, SAMPLE_RAW, SAMPLE_OUT)
    return {"url": "/edit/sample.mp4?t=" + str(int(time.time() * 1000))}


@app.get("/edit/sample.mp4")
def edit_sample_mp4():
    if not SAMPLE_OUT.exists():
        return Response(status_code=404)
    return FileResponse(SAMPLE_OUT, media_type="video/mp4")


@app.get("/edit/params")
def edit_get_params():
    return _compose_params


@app.post("/edit/autotune")
async def edit_autotune(req: dict | None = None):
    n = (req or {}).get("trials", 15)
    res = await asyncio.to_thread(_autotune_search, n)
    res["url"] = "/edit/sample.mp4?t=" + str(int(time.time() * 1000))
    res["params"] = _compose_params
    return res


@app.post("/edit/iterate")
async def edit_iterate(req: dict | None = None):
    n = (req or {}).get("iters", 5)
    res = await asyncio.to_thread(_autotune_iterative, n)
    res["url"] = "/edit/sample.mp4?t=" + str(int(time.time() * 1000))
    res["params"] = _compose_params
    return res


@app.post("/edit/params")
async def edit_set_params(params: dict):
    """Update params + recomposite sample. Returns new sample URL."""
    for k, v in params.items():
        if k in _compose_params:
            _compose_params[k] = v
    _face_mask_cache.clear()
    if SAMPLE_RAW.exists():
        await asyncio.to_thread(composite_overlay, SAMPLE_RAW, SAMPLE_OUT)
        return {"url": "/edit/sample.mp4?t=" + str(int(time.time() * 1000)), "params": _compose_params}
    return {"params": _compose_params}


@app.get("/health")
def health():
    return {"status": "ok", "sdk_loaded": _sdk is not None, "avatar": AVATAR_PATH}


@app.get("/", response_class=HTMLResponse)
def home():
    return INDEX_HTML


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Charlie · Ditto Voice Chat</title>
<style>
  *{box-sizing:border-box}
  body{margin:0;padding:24px;background:#0d1117;color:#e6edf3;
    font-family:-apple-system,system-ui,sans-serif;min-height:100vh;display:grid;
    grid-template-columns:1fr 1fr;gap:24px;max-width:1280px;margin:auto}
  h1{margin:0 0 6px;font-size:22px}
  .sub{color:#8b949e;font-size:13px;margin:0 0 14px}
  .ws{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;margin-left:8px}
  .ws.on{background:#1f6feb}.ws.off{background:#6e7681}
  .card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px}
  video{width:100%;border-radius:8px;background:#000;max-height:60vh}
  button{background:#1f6feb;color:#fff;border:0;padding:10px 18px;border-radius:8px;
    cursor:pointer;font-size:14px;margin-right:8px;margin-top:4px}
  button:hover{background:#388bfd}
  button:disabled{opacity:0.5;cursor:not-allowed}
  .micbtn{background:#238636}
  .micbtn.rec{background:#da3633;animation:pulse 1.2s infinite}
  .micbtn.denied{background:#6e7681}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.55}}
  textarea{width:100%;min-height:44px;background:#0d1117;color:#e6edf3;
    border:1px solid #30363d;border-radius:6px;padding:10px;font:inherit;font-size:14px}
  .chat{max-height:60vh;overflow-y:auto;padding-right:4px}
  .msg{padding:10px 12px;border-radius:8px;margin-bottom:8px;font-size:14px;line-height:1.4}
  .msg.user{background:#1f2937;border-left:3px solid #58a6ff}
  .msg.bot{background:#0f1c2d;border-left:3px solid #3fb950}
  .msg.bot.pending{opacity:0.6}
  .timings{color:#8b949e;font-size:11px;margin-top:4px}
  .status{color:#d29922;font-size:13px;min-height:20px;margin-top:6px}
  .vad-bar{height:6px;background:#21262d;border-radius:3px;overflow:hidden;margin-top:8px}
  .vad-fill{height:100%;background:#3fb950;width:0;transition:width 80ms linear}
  .stages{display:flex;gap:6px;margin-top:8px;font-size:11px;flex-wrap:wrap}
  .stage{padding:3px 8px;border-radius:4px;background:#21262d;color:#8b949e}
  .stage.done{background:#1f6feb;color:#fff}
</style>
</head>
<body>
<div>
<h1>Charlie · Ditto <span class="ws off" id="wsTag">WS off</span></h1>
<p class="sub">Groq Whisper · Llama 3.3 70B · Orpheus TTS · Ditto on RTX 4090 · VAD streaming</p>

<div class="card">
  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
    <button id="vadBtn" class="micbtn">🎤 <span id="vadLabel">Ativar voz</span></button>
    <button id="sendBtn">Enviar texto</button>
    <button id="clearBtn">Limpar</button>
    <button id="editBtn">⚙ Editar overlay</button>
  </div>
  <div class="vad-bar"><div class="vad-fill" id="vadFill"></div></div>
  <textarea id="textIn" placeholder="Type or talk"></textarea>
  <div class="stages" id="stages">
    <span class="stage" data-s="stt">STT</span>
    <span class="stage" data-s="llm">LLM</span>
    <span class="stage" data-s="tts">TTS</span>
    <span class="stage" data-s="video">Video</span>
  </div>
  <div class="status" id="status">conectando WS...</div>
</div>

<div class="card" style="margin-top:14px"><div class="chat" id="chat"></div></div>
</div>

<div>
<div class="card" style="position:relative;padding:0;overflow:hidden;aspect-ratio:1180/652;max-height:60vh;margin-left:auto;margin-right:auto;background:#000">
  <video id="idlePlayer" src="/body_bg.mp4" muted loop autoplay playsinline
         style="position:absolute;top:0;left:0;width:100%;height:100%;object-fit:cover;background:#000;display:block"></video>
  <video id="replyPlayer" playsinline disablepictureinpicture
         style="position:absolute;top:0;left:0;width:100%;height:100%;object-fit:cover;background:transparent;
         opacity:0;transition:opacity 0.3s ease-out;pointer-events:none"></video>
</div>

<div class="card" style="margin-top:10px;padding:10px;font-size:14px;text-align:center;font-family:monospace">
  <span id="lastTimings" style="color:#8b949e">timings: —</span>
</div>

<div id="editPanel" class="card" style="margin-top:10px;padding:12px;display:none">
  <h3 style="margin:0 0 6px 0">Edit overlay (ao vivo)</h3>
  <video id="editPreview" autoplay loop muted playsinline style="width:100%;border-radius:6px;background:#000;max-height:50vh"></video>
  <div id="editControls" style="margin-top:10px;display:grid;grid-template-columns:auto 1fr auto;gap:6px 10px;font-size:12px;align-items:center"></div>
  <div style="margin-top:8px;display:flex;gap:8px;justify-content:space-between;align-items:center;flex-wrap:wrap">
    <div style="display:flex;gap:6px;align-items:center">
      <button id="autotuneBtn">🤖 Random search</button>
      <input id="autotuneN" type="number" value="15" min="3" max="60" style="width:56px" title="trials">
      <button id="iterateBtn">♻ Iterate (LLM rec)</button>
      <input id="iterateN" type="number" value="5" min="1" max="15" style="width:48px" title="iters">
    </div>
    <button id="editClose">Fechar</button>
  </div>
  <pre id="autotuneLog" style="margin-top:8px;font-size:11px;max-height:160px;overflow:auto;background:#0d1117;padding:6px;border-radius:4px;display:none"></pre>
</div>

</div>

<script defer src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.24/dist/bundle.min.js"></script>
<script>
const MicVAD = ()=>window.vad?.MicVAD;
const el=id=>document.getElementById(id);
const vadBtn=el('vadBtn'),vadLabel=el('vadLabel'),vadFill=el('vadFill');
const sendBtn=el('sendBtn'),clearBtn=el('clearBtn'),textIn=el('textIn'),status=el('status');
const chat=el('chat'),idlePlayer=el('idlePlayer'),replyPlayer=el('replyPlayer'),wsTag=el('wsTag');
const stages={stt:el('stages').querySelector('[data-s=stt]'),llm:el('stages').querySelector('[data-s=llm]'),tts:el('stages').querySelector('[data-s=tts]'),video:el('stages').querySelector('[data-s=video]')};

let ws=null,vad=null,vadOn=false,busy=false,pendingUserMsg=null,pendingBotMsg=null;
function resetStages(){Object.values(stages).forEach(s=>s.classList.remove('done'))}
function markStage(n){if(stages[n])stages[n].classList.add('done')}

function connectWS(){
  ws=new WebSocket(`ws://${location.host}/ws`);
  ws.onopen=()=>{wsTag.textContent='WS on';wsTag.className='ws on';status.textContent='✓ pronto'};
  ws.onclose=()=>{wsTag.textContent='WS off';wsTag.className='ws off';status.textContent='reconnecting...';setTimeout(connectWS,1500)};
  ws.onerror=()=>{status.textContent='WS erro'};
  ws.onmessage=ev=>{
    const d=JSON.parse(ev.data);
    if(d.type==='transcript'){
      markStage('stt');
      if(!pendingUserMsg)pendingUserMsg=addMsg('user',d.text,`stt em ${d.elapsed_s}s`);
      else pendingUserMsg.querySelector('.text').textContent='🗣 '+d.text;
    } else if(d.type==='reply'){
      markStage('llm');
      pendingBotMsg=addMsg('bot pending',d.text,`stt+llm em ${d.elapsed_s}s · aguardando video...`);
    } else if(d.type==='tts_ready'){
      markStage('tts');
      if(pendingBotMsg)pendingBotMsg.querySelector('.timings').textContent=`tts em ${d.elapsed_s}s · renderizando ditto...`;
    } else if(d.type==='video'){
      markStage('video');
      if(pendingBotMsg){pendingBotMsg.classList.remove('pending');const t=d.timings;
        pendingBotMsg.querySelector('.timings').textContent=`stt ${t.stt_s||0}s · llm ${t.llm_s}s · tts ${t.tts_s}s · ditto ${t.ditto_s}s · total ${t.total_s}s`;}
      const t=d.timings;el('lastTimings').innerHTML=`<b style="color:#58a6ff">total ${t.total_s}s</b> · stt ${t.stt_s||0}s · llm ${t.llm_s}s · tts ${t.tts_s}s · ditto ${t.ditto_s}s`;
      playReply(d.url);
      status.textContent=vadOn?'✓ VAD ativo · fale novamente':'✓';
      busy=false;pendingUserMsg=null;pendingBotMsg=null;
    } else if(d.type==='error'){status.textContent='erro: '+d.message;busy=false}
  };
}
connectWS();

async function toggleVAD(){
  if(vadOn){stopVAD();return}
  status.textContent='pedindo mic...';
  try{
    if(!vad){
      vad=await MicVAD().new({
        onSpeechStart:()=>{if(busy)return;vadBtn.classList.add('rec');status.textContent='🎙 ouvindo...'},
        onSpeechEnd:async (audio)=>{
          vadBtn.classList.remove('rec');
          if(busy)return;
          if(audio.length<8000){status.textContent='(muito curto)';return}
          sendAudio(audio);
        },
        onVADMisfire:()=>vadBtn.classList.remove('rec'),
        onFrameProcessed:p=>vadFill.style.width=Math.min(100,p.isSpeech*100)+'%',
        positiveSpeechThreshold:0.6, negativeSpeechThreshold:0.35, minSpeechFrames:4,
      });
    }
    await vad.start();
    vadOn=true;vadBtn.classList.remove('denied');
    vadLabel.textContent='Desativar voz';status.textContent='✓ VAD ativo';
  } catch(e){vadBtn.classList.add('denied');status.textContent='mic negado: '+e.message}
}
function stopVAD(){if(vad)vad.pause();vadOn=false;vadBtn.classList.remove('rec');vadLabel.textContent='Ativar voz';vadFill.style.width='0';status.textContent='VAD pausado'}

vadBtn.onclick=toggleVAD;

// ── Edit overlay panel ────────────────────────────────────────────
const editBtn=el('editBtn'),editPanel=el('editPanel'),editPreview=el('editPreview'),editControls=el('editControls'),editClose=el('editClose');
const PARAM_DEFS=[
  {k:'color_match',type:'bool',label:'Color match (Reinhard)',desc:'Iguala tom de pele do ditto ao body via media+std LAB'},
  {k:'color_strength',type:'range',min:0,max:1,step:0.05,label:'Forca color match',desc:'0=desligado, 1=total'},
  {k:'brightness',type:'range',min:-50,max:50,step:1,label:'Brilho',desc:'Soma na luminancia L*'},
  {k:'contrast',type:'range',min:0.5,max:1.5,step:0.02,label:'Contraste',desc:'Multiplica L*'},
  {k:'warp_enabled',type:'bool',label:'Warp afim (rot+escala)',desc:'Rotaciona/escala ditto pra match pose body. Off=so resize+paste'},
  {k:'track_body_motion',type:'bool',label:'Seguir movimento body',desc:'On=cabeca segue body. Off=trava na posicao do frame 0'},
  {k:'dpts_ema',type:'range',min:0.05,max:1.0,step:0.05,label:'Suavizar ditto landmarks',desc:'Menor=mais suave (sem jitter). 1=sem suavizar'},
  {k:'M_ema',type:'range',min:0.05,max:1.0,step:0.05,label:'Suavizar transform',desc:'Suaviza matriz afim. Menor=movimento mais lento/estavel'},
  {k:'body_smooth_window',type:'range',min:1,max:21,step:2,label:'Suavizar body landmarks',desc:'Janela media-movel nos pts do body. 1=sem'},
  {k:'ditto_lock_head',type:'bool',label:'Travar cabeca Ditto',desc:'Ditto so anima expressao (sem mover cabeca propria). Requer regerar sample'},
  {k:'mask_inner',type:'range',min:0.5,max:1,step:0.01,label:'Mask inner',desc:'Onde comeca fade da borda do circulo'},
  {k:'mask_outer',type:'range',min:0.5,max:1.2,step:0.01,label:'Mask outer',desc:'Onde termina (limite externo)'},
  {k:'mask_rx',type:'range',min:0.2,max:0.6,step:0.01,label:'Mask raio X',desc:'Largura da elipse mask'},
  {k:'mask_ry',type:'range',min:0.2,max:0.7,step:0.01,label:'Mask raio Y',desc:'Altura da elipse mask'},
  {k:'mask_cy',type:'range',min:0.3,max:0.7,step:0.01,label:'Mask centro Y',desc:'Posicao vertical do centro'},
];
let editParams={};
let editTimer=null;

function buildEditUI(){
  editControls.innerHTML='';
  PARAM_DEFS.forEach(d=>{
    const lbl=document.createElement('label');lbl.textContent=d.label;lbl.title=d.desc;
    const ctrl=document.createElement(d.type==='bool'?'input':'input');
    if(d.type==='bool'){
      ctrl.type='checkbox';ctrl.checked=!!editParams[d.k];
      ctrl.onchange=()=>{editParams[d.k]=ctrl.checked;sched();};
    }else{
      ctrl.type='range';ctrl.min=d.min;ctrl.max=d.max;ctrl.step=d.step;ctrl.value=editParams[d.k];
      ctrl.oninput=()=>{editParams[d.k]=parseFloat(ctrl.value);valSpan.textContent=ctrl.value;sched();};
    }
    ctrl.title=d.desc;
    const valSpan=document.createElement('span');
    valSpan.textContent=d.type==='bool'?'':editParams[d.k];
    valSpan.style.minWidth='40px';valSpan.style.textAlign='right';
    editControls.appendChild(lbl);editControls.appendChild(ctrl);editControls.appendChild(valSpan);
  });
}
function sched(){if(editTimer)clearTimeout(editTimer);editTimer=setTimeout(applyParams,400);}
async function applyParams(){
  const r=await fetch('/edit/params',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(editParams)});
  const j=await r.json();
  if(j.url){editPreview.src=j.url;editPreview.load();}
}
async function openEdit(){
  editPanel.style.display='block';
  status.textContent='gerando sample...';
  const r=await fetch('/edit/sample',{method:'POST'});
  const j=await r.json();
  editPreview.src=j.url;editPreview.load();
  const r2=await fetch('/edit/params');editParams=await r2.json();
  buildEditUI();
  status.textContent='editor pronto';
}
editBtn.onclick=openEdit;
editClose.onclick=()=>{editPanel.style.display='none';};

el('iterateBtn').onclick=async()=>{
  const log=el('autotuneLog');log.style.display='block';
  const n=parseInt(el('iterateN').value)||5;
  log.textContent=`iterando ${n}x via Claude...`;
  el('iterateBtn').disabled=true;
  try{
    const r=await fetch('/edit/iterate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({iters:n})});
    const j=await r.json();
    log.textContent=JSON.stringify(j,null,2);
    if(j.url){editPreview.src=j.url;editPreview.load();}
    if(j.params){editParams=j.params;buildEditUI();}
  }catch(e){log.textContent='err: '+e}
  el('iterateBtn').disabled=false;
};
el('autotuneBtn').onclick=async()=>{
  const log=el('autotuneLog');log.style.display='block';
  const n=parseInt(el('autotuneN').value)||15;
  log.textContent=`rodando ${n} trials via Gemini... (~${n*8}s)`;
  el('autotuneBtn').disabled=true;
  try{
    const r=await fetch('/edit/autotune',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({trials:n})});
    const j=await r.json();
    log.textContent=JSON.stringify(j,null,2);
    if(j.url){editPreview.src=j.url;editPreview.load();}
    if(j.params){editParams=j.params;buildEditUI();}
  }catch(e){log.textContent='err: '+e}
  el('autotuneBtn').disabled=false;
};
sendBtn.onclick=()=>{const t=textIn.value.trim();if(t)sendText(t)};
textIn.onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendBtn.onclick()}};
clearBtn.onclick=()=>{chat.innerHTML='';textIn.value='';resetStages();if(ws&&ws.readyState===1)ws.send(JSON.stringify({type:'clear'}))};

function sendText(text){
  if(busy||ws.readyState!==1)return;
  busy=true;resetStages();textIn.value='';
  pendingUserMsg=addMsg('user',text,'enviando...');markStage('stt');
  status.textContent='llm → tts → ditto...';
  ws.send(JSON.stringify({type:'text',text}));
}
function sendAudio(samples){
  if(busy||ws.readyState!==1)return;
  busy=true;resetStages();status.textContent='transcribing...';
  pendingUserMsg=addMsg('user','⏳ transcrevendo...','');
  ws.send(encodeWAV(samples,16000));
}


let videosUnlocked=false;
function unlockVideos(){
  if(videosUnlocked)return;
  idlePlayer.muted=true;
  idlePlayer.play().catch(()=>{});
  videosUnlocked=true;
}
['click','touchstart'].forEach(ev=>document.addEventListener(ev,unlockVideos));

replyPlayer.addEventListener('ended',()=>{
  try{idlePlayer.currentTime=0;idlePlayer.play().catch(()=>{})}catch(e){}
  replyPlayer.style.opacity='0';
  setTimeout(()=>{replyPlayer.pause();replyPlayer.currentTime=0},500);
});
function playReply(url){
  const fullUrl=url+'?t='+Date.now();
  replyPlayer.src=fullUrl;
  replyPlayer.load();
  replyPlayer.style.opacity='1';
  const tryPlay=()=>{
    replyPlayer.play().catch(e=>{status.textContent='⚠ clique no video pra tocar'});
  };
  tryPlay();
  replyPlayer.addEventListener('canplay',tryPlay,{once:true});
}

function addMsg(role,text,timing){
  const div=document.createElement('div');
  div.className='msg '+role;
  div.innerHTML='<div class="text">'+(role.startsWith('user')?'🗣 ':'🤖 ')+text+'</div>'+
    '<div class="timings">'+(timing||'')+'</div>';
  chat.appendChild(div);chat.scrollTop=chat.scrollHeight;return div;
}
function encodeWAV(samples,sampleRate){
  const buf=new ArrayBuffer(44+samples.length*2);
  const view=new DataView(buf);
  const ws=(o,s)=>{for(let i=0;i<s.length;i++)view.setUint8(o+i,s.charCodeAt(i))};
  ws(0,'RIFF');view.setUint32(4,36+samples.length*2,true);ws(8,'WAVE');
  ws(12,'fmt ');view.setUint32(16,16,true);view.setUint16(20,1,true);view.setUint16(22,1,true);
  view.setUint32(24,sampleRate,true);view.setUint32(28,sampleRate*2,true);
  view.setUint16(32,2,true);view.setUint16(34,16,true);
  ws(36,'data');view.setUint32(40,samples.length*2,true);
  let o=44;
  for(let i=0;i<samples.length;i++){
    const s=Math.max(-1,Math.min(1,samples[i]));
    view.setInt16(o,s<0?s*0x8000:s*0x7FFF,true);o+=2;
  }
  return buf;
}
</script>
</body></html>
"""
