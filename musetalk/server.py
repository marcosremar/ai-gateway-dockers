"""
MuseTalk Service — FastAPI wrapper pro ai-gateway.

Dois caminhos de inferência:

1. POST /v1/lipsync (one-shot, file-based)
   Multipart: image (PNG/JPG/MP4) + audio (WAV/MP3) → MP4 de saída.
   Delega pro MuseTalkInference.generate() original do fork ruxir-ig.

2. WS /v1/stream (streaming real-time)
   Cliente envia uma mensagem JSON de init com a imagem de referência
   (base64) e depois binários de áudio PCM s16le 16kHz mono. Servidor
   responde com JSON de ready e depois com binários JPEG dos frames.

   Otimização-chave: preprocessing da imagem (face detection + VAE
   encode das latents) acontece UMA vez no init e fica em cache na
   sessão. Os chunks de áudio só rodam whisper→PE→UNet→VAE decode→blend,
   que é a hot path que o paper cita como 30fps+.

Env:
  IDLE_TIMEOUT_MIN   — auto-shutdown depois de inativo (default 15; 0=off)
  MUSETALK_MODELS_DIR — diretório de pesos (default /app/models)
  MUSETALK_STREAM_CHUNK_MS — ms de áudio por chunk emitido (default 1000)
"""

import asyncio
import base64
import io
import os
import sys
import tempfile
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
import uvicorn

from idle_watchdog import add_idle_middleware, start_watchdog, touch_activity

sys.path.insert(0, "/app")

# ── Inference engine + streaming state ──────────────────────────────────────

inference_engine = None
load_error: Optional[str] = None
load_traceback: Optional[str] = None
model_loaded = False

# sessionId -> preprocessed reference cache. Cada entrada é um dict com:
#   frame          : np.ndarray BGR da imagem de referência
#   coord          : bbox do rosto (x1, y1, x2, y2)
#   latent         : tensor VAE encode do face crop 256x256
#   fp             : FaceParsing instance (usado no blending)
#   extra_margin   : margem pra blending
#   parsing_mode   : 'jaw' ou 'raw'
stream_sessions: Dict[str, Dict[str, Any]] = {}

# Limite: expira sessões sem atividade após N segundos
STREAM_SESSION_TTL = 600


def _load_inference_engine():
    global inference_engine, load_error, load_traceback, model_loaded
    try:
        print("[musetalk] Carregando MuseTalkInference...", flush=True)
        t0 = time.time()
        from api.inference_service import MuseTalkInference  # type: ignore

        # Determinismo: chamadas idênticas devem produzir pixels idênticos.
        # Sem isso, cuDNN nondeterministic gera bocas ligeiramente diferentes
        # a cada call, contribuindo pra "tremor" em pipelines híbridos.
        try:
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

        engine = MuseTalkInference(use_float16=True)
        engine.load_models()
        inference_engine = engine
        model_loaded = True
        print(f"[musetalk] Modelo carregado em {time.time() - t0:.1f}s", flush=True)
    except Exception as e:  # noqa: BLE001
        load_error = str(e)
        load_traceback = traceback.format_exc()
        print(f"[musetalk] ERRO carregando modelo: {e}", flush=True, file=sys.stderr)
        print(load_traceback, flush=True, file=sys.stderr)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[musetalk] Startup — iniciando carga de modelos em background...", flush=True)
    asyncio.get_event_loop().run_in_executor(None, _load_inference_engine)
    asyncio.create_task(start_watchdog())
    asyncio.create_task(_expire_stream_sessions())
    yield
    print("[musetalk] Shutdown", flush=True)


async def _expire_stream_sessions():
    """Remove sessões WS que ficaram idle mais que STREAM_SESSION_TTL segundos."""
    while True:
        await asyncio.sleep(60)
        now = time.time()
        expired = [sid for sid, s in stream_sessions.items() if (now - s.get("last_use", 0)) > STREAM_SESSION_TTL]
        for sid in expired:
            stream_sessions.pop(sid, None)
            print(f"[musetalk] Session {sid} expirada", flush=True)


app = FastAPI(
    title="MuseTalk Service",
    description="Real-time audio-driven lip-sync pro ai-gateway.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

add_idle_middleware(app)


# ── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_vram_gb: Optional[float] = None
    load_error: Optional[str] = None
    active_streams: int = 0
    active_requests: int = 0
    # Epoch seconds da última request servida (one-shot ou streaming chunk).
    # O ai-gateway usa esse campo pra resetar o timer de idle e não matar
    # pods que estão sendo usados via bypass do gateway.
    last_request_at: float = 0.0
    # Utilização REAL da GPU (%) — lida via nvidia-smi a cada /health.
    # Crítico pra ai-gateway: se gpu_util > 5%, há processamento ativo
    # mesmo que active_requests == 0 (ex: workload externo, batch interno),
    # e o pod NÃO deve ser auto-paused.
    gpu_util: float = 0.0
    gpu_mem_used_gb: float = 0.0
    # Campos exigidos pelo readiness check do ai-gateway. MuseTalk não tem
    # STT/LLM/TTS, então mapeamos pra "loaded" assim que o modelo carregar.
    services: Optional[Dict[str, str]] = None


def _read_gpu_util() -> tuple[float, float]:
    """Retorna (utilization_percent, mem_used_gb) lendo nvidia-smi.

    Falha silenciosa se nvidia-smi não disponível — retorna (0,0).
    """
    try:
        import subprocess as _sp
        r = _sp.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        if r.returncode != 0:
            return 0.0, 0.0
        line = r.stdout.strip().split("\n")[0]
        util_s, mem_s = [x.strip() for x in line.split(",")]
        return float(util_s), float(mem_s) / 1024.0
    except Exception:
        return 0.0, 0.0


# Tracking de atividade (atualizado por todos os handlers que de fato usam GPU)
import time as _time
_last_request_at: float = 0.0
_active_requests: int = 0


def _touch_activity():
    global _last_request_at
    _last_request_at = _time.time()


def _services_status() -> Dict[str, str]:
    s = "loaded" if model_loaded else "loading"
    return {"whisper": s, "llama_cpp": s, "tts": s}


@app.get("/health", response_model=HealthResponse)
async def health():
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    gpu_vram_gb = None
    if gpu_available:
        gpu_vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1)

    gpu_util, gpu_mem_used_gb = _read_gpu_util()
    if load_error:
        return HealthResponse(status="error", model_loaded=False, gpu_available=gpu_available,
                              gpu_name=gpu_name, gpu_vram_gb=gpu_vram_gb, load_error=load_error,
                              gpu_util=gpu_util, gpu_mem_used_gb=gpu_mem_used_gb,
                              services=_services_status())
    if not model_loaded:
        return HealthResponse(status="loading", model_loaded=False, gpu_available=gpu_available,
                              gpu_name=gpu_name, gpu_vram_gb=gpu_vram_gb,
                              gpu_util=gpu_util, gpu_mem_used_gb=gpu_mem_used_gb,
                              services=_services_status())
    return HealthResponse(status="healthy", model_loaded=True, gpu_available=gpu_available,
                          gpu_name=gpu_name, gpu_vram_gb=gpu_vram_gb,
                          active_streams=len(stream_sessions),
                          active_requests=_active_requests,
                          last_request_at=_last_request_at,
                          gpu_util=gpu_util, gpu_mem_used_gb=gpu_mem_used_gb,
                          services=_services_status())


# ── Lip sync (one-shot) ─────────────────────────────────────────────────────

@app.post("/v1/lipsync")
async def lipsync(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    enhance: bool = Form(default=False),
    fps: int = Form(default=25, ge=1, le=60),
    batch_size: int = Form(default=8, ge=1, le=32),
    extra_margin: int = Form(default=10, ge=0, le=40),
):
    global _active_requests
    touch_activity()
    _touch_activity()
    _active_requests += 1
    if not model_loaded:
        _active_requests = max(0, _active_requests - 1)
        raise HTTPException(status_code=503, detail=f"Modelo ainda carregando ou falhou: {load_error}")

    try:
      with tempfile.TemporaryDirectory(prefix="musetalk_") as tmp:
        tmp_path = Path(tmp)
        img_path = tmp_path / (image.filename or "reference.png")
        aud_path = tmp_path / (audio.filename or "driver.wav")
        img_path.write_bytes(await image.read())
        aud_path.write_bytes(await audio.read())

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        out_name = "result"

        t0 = time.time()
        try:
            out_path = inference_engine.generate(  # type: ignore[union-attr]
                audio_path=str(aud_path),
                video_path=str(img_path),
                enhance=enhance,
                fps=fps,
                batch_size=batch_size,
                extra_margin=extra_margin,
                output_name=out_name,
                result_dir=str(out_dir),
            )
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            print(tb, flush=True, file=sys.stderr)
            raise HTTPException(status_code=500, detail=f"inference falhou: {e}")

        elapsed = time.time() - t0
        mp4_path = Path(out_path)
        if not mp4_path.exists():
            raise HTTPException(status_code=500, detail=f"MP4 não foi gerado em {out_path}")

        data = mp4_path.read_bytes()
        headers = {
            "X-Elapsed-Seconds": f"{elapsed:.2f}",
            "Content-Disposition": f'attachment; filename="{out_name}.mp4"',
        }
        return Response(content=data, media_type="video/mp4", headers=headers)
    finally:
      _active_requests = max(0, _active_requests - 1)
      _touch_activity()


# ── Streaming helpers ───────────────────────────────────────────────────────

def _preprocess_reference(img_bytes: bytes, bbox_shift: int, extra_margin: int,
                          parsing_mode: str, left_cheek_width: int, right_cheek_width: int) -> Dict[str, Any]:
    """Face detection + VAE encode do crop de referência. Idempotente,
    chamado UMA vez por sessão — o resultado fica cacheado e é reusado
    em todos os chunks de áudio."""
    from musetalk.utils.preprocessing import get_landmark_and_bbox, coord_placeholder  # type: ignore
    from musetalk.utils.face_parsing import FaceParsing  # type: ignore

    # Salva temp pro reader do preprocessing aceitar path
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(img_bytes)
        img_path = f.name
    try:
        coord_list, frame_list = get_landmark_and_bbox([img_path], bbox_shift)
    finally:
        try: os.unlink(img_path)
        except OSError: pass

    if not coord_list or coord_list[0] == coord_placeholder:
        raise ValueError("Nenhum rosto detectado na imagem de referência")

    bbox = coord_list[0]
    frame = frame_list[0]
    x1, y1, x2, y2 = bbox
    y2 = min(y2 + extra_margin, frame.shape[0])
    crop = frame[y1:y2, x1:x2]
    crop_256 = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)

    latent = inference_engine.vae.get_latents_for_unet(crop_256)  # type: ignore[union-attr]
    fp = FaceParsing(left_cheek_width=left_cheek_width, right_cheek_width=right_cheek_width)

    # Pré-computa mask + crop_box UMA vez aqui (caros: face parser + blur).
    # Blending por frame vira só paste com mask_array cached.
    from musetalk.utils.blending import get_image_prepare_material  # type: ignore
    mask_array, crop_box = get_image_prepare_material(
        frame, [x1, y1, x2, y2], fp=fp, mode=parsing_mode,
    )

    return {
        "frame": frame,
        "coord": (x1, y1, x2, y2),
        "latent": latent,
        "fp": fp,
        "mask_array": mask_array,
        "crop_box": crop_box,
        "extra_margin": extra_margin,
        "parsing_mode": parsing_mode,
        "last_use": time.time(),
    }


def _render_chunk_frames(session: Dict[str, Any], pcm_s16le_mono_16k: bytes, fps: int) -> list:
    """Gera a lista de frames BGR pra um chunk de áudio (PCM s16le mono 16kHz).
    Todo o pipeline exceto face detection (já cacheado)."""
    from musetalk.utils.blending import get_image_blending  # type: ignore

    assert inference_engine is not None
    engine = inference_engine

    samples = np.frombuffer(pcm_s16le_mono_16k, dtype=np.int16).astype(np.float32) / 32768.0
    librosa_length = len(samples)
    if librosa_length == 0:
        return []

    # Whisper features (single 30s-padded segment)
    feat = engine.audio_processor.feature_extractor(
        samples, return_tensors="pt", sampling_rate=16000
    ).input_features
    feat = feat.to(engine.device).to(engine.weight_dtype)
    audio_feats = engine.whisper.encoder(feat, output_hidden_states=True).hidden_states
    audio_feats = torch.stack(audio_feats, dim=2)

    import math
    sr = 16000
    audio_fps = 50
    whisper_idx_multiplier = audio_fps / fps
    num_frames = math.floor((librosa_length / sr) * fps)
    actual_length = math.floor((librosa_length / sr) * audio_fps)
    audio_feats = audio_feats[:, :actual_length, ...]
    left_pad = 2
    right_pad = 2
    padding_nums = math.ceil(whisper_idx_multiplier)
    audio_feats = torch.cat([
        torch.zeros_like(audio_feats[:, :padding_nums * left_pad]),
        audio_feats,
        torch.zeros_like(audio_feats[:, :padding_nums * 3 * right_pad]),
    ], 1)

    from einops import rearrange
    audio_prompts = []
    feat_len_per_frame = 2 * (left_pad + right_pad + 1)
    for frame_index in range(num_frames):
        audio_index = math.floor(frame_index * whisper_idx_multiplier)
        audio_clip = audio_feats[:, audio_index:audio_index + feat_len_per_frame]
        if audio_clip.shape[1] != feat_len_per_frame:
            continue
        audio_prompts.append(audio_clip)
    if not audio_prompts:
        return []
    audio_prompts = torch.cat(audio_prompts, dim=0)                # (T, 10, 5, 384)
    audio_prompts = rearrange(audio_prompts, 'b c h w -> b (c h) w')  # (T, 50, 384)
    whisper_chunks = [audio_prompts[i] for i in range(audio_prompts.shape[0])]

    if not whisper_chunks:
        return []

    # Batch pelos chunks (todos compartilham a mesma latent)
    batch_size = 8
    latent = session["latent"]
    frame = session["frame"]
    bbox = session["coord"]
    mask_array = session["mask_array"]
    crop_box = session["crop_box"]
    x1, y1, x2, y2 = bbox

    out_frames = []
    with torch.no_grad():
        for i in range(0, len(whisper_chunks), batch_size):
            batch = whisper_chunks[i:i + batch_size]
            whisper_batch = torch.stack(batch, dim=0).to(engine.device).to(engine.weight_dtype)
            latent_batch = latent.expand(len(batch), -1, -1, -1).to(dtype=engine.weight_dtype)

            audio_feature_batch = engine.pe(whisper_batch)
            pred_latents = engine.unet.model(
                latent_batch, engine.timesteps, encoder_hidden_states=audio_feature_batch
            ).sample
            recon = engine.vae.decode_latents(pred_latents)
            for res_frame in recon:
                face = res_frame.astype(np.uint8)
                try:
                    face_resized = cv2.resize(face, (x2 - x1, y2 - y1))
                except Exception:
                    continue
                blended = get_image_blending(
                    frame, face_resized, [x1, y1, x2, y2], mask_array, crop_box,
                )
                out_frames.append(blended)

    return out_frames


# ── Mouth refinement (single-frame, hybrid concat use-case) ─────────────────

@app.post("/v1/refine_mouth")
async def refine_mouth(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    extra_margin: int = Form(default=10),
    parsing_mode: str = Form(default="jaw"),
    bbox_shift: int = Form(default=0),
    left_cheek_width: int = Form(default=90),
    right_cheek_width: int = Form(default=90),
):
    """Refina SÓ a região da boca de UMA imagem usando 1 chunk curto de áudio.

    Caso de uso: pipeline concatenativo escolheu um frame, mas SyncNet detectou
    que a boca não casa com o áudio nessa janela. Esse endpoint regenera só a
    boca daquele frame específico (resto fica intocado).

    Custo: ~8-12% do /v1/lipsync (apenas 1 frame, sem video assembly).
    """
    global _active_requests
    touch_activity()
    _touch_activity()
    _active_requests += 1
    if not model_loaded:
        _active_requests = max(0, _active_requests - 1)
        raise HTTPException(status_code=503, detail=f"Model not ready: {load_error}")

    try:
        img_bytes = await image.read()
        audio_bytes = await audio.read()

        # Prepara sessão (face detect + VAE latent + mask) a partir da imagem
        session = _preprocess_reference(
            img_bytes=img_bytes, bbox_shift=bbox_shift,
            extra_margin=extra_margin, parsing_mode=parsing_mode,
            left_cheek_width=left_cheek_width, right_cheek_width=right_cheek_width,
        )

        # Carrega áudio → PCM s16le mono 16kHz (formato esperado por _render_chunk_frames)
        import soundfile as sf
        import io as _io
        try:
            data, sr = sf.read(_io.BytesIO(audio_bytes), dtype="int16")
        except Exception as e:
            raise HTTPException(400, f"audio decode failed: {e}")
        if data.ndim > 1:
            data = data.mean(axis=1).astype(np.int16)
        if sr != 16000:
            try:
                import librosa
                f = data.astype(np.float32) / 32768.0
                f = librosa.resample(f, orig_sr=sr, target_sr=16000)
                data = (f * 32768).clip(-32768, 32767).astype(np.int16)
            except Exception as e:
                raise HTTPException(400, f"resample failed: {e}")

        frames = _render_chunk_frames(session, data.tobytes(), fps=25)
        if not frames:
            raise HTTPException(500, "no frame generated (audio too short?)")

        # Pega o frame central (boca melhor alinhada com o centro do chunk de áudio)
        out = frames[len(frames) // 2]
        ok, jpeg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise HTTPException(500, "jpeg encode failed")
        return Response(content=jpeg.tobytes(), media_type="image/jpeg",
                        headers={"X-Frames-Generated": str(len(frames))})
    finally:
        _active_requests = max(0, _active_requests - 1)
        _touch_activity()


# ── Workspace backup (manual trigger; auto-backup roda via start.sh) ───────

@app.post("/v1/backup_now")
async def backup_now():
    """Trigger imediato do /app/backup_workspace.sh.

    Útil pra:
    - Forçar backup antes de pausar/encerrar pod
    - Testar credenciais B2 sem esperar 24h
    - Pipelines longos podem chamar antes de terminar pra garantir persistência

    Retorna: {ok, exit_code, log_tail (últimas 30 linhas)}.
    Não bloqueia: roda como subprocess e devolve resultado.
    """
    import subprocess as _sp
    log_path = "/var/log/workspace_backup.log"
    try:
        r = _sp.run(["/app/backup_workspace.sh"],
                    capture_output=True, text=True, timeout=1800)
        log_tail = ""
        try:
            with open(log_path) as f:
                log_tail = "\n".join(f.read().splitlines()[-30:])
        except Exception:
            pass
        return {"ok": r.returncode == 0, "exit_code": r.returncode,
                "stdout_tail": (r.stdout or "")[-2000:],
                "stderr_tail": (r.stderr or "")[-2000:],
                "log_tail": log_tail}
    except _sp.TimeoutExpired:
        raise HTTPException(504, "backup script timed out (>30 min)")
    except FileNotFoundError:
        raise HTTPException(500, "/app/backup_workspace.sh not found")


# ── Mouth refinement BATCH (block of frames, single MuseTalk call) ─────────

@app.post("/v1/refine_mouth_block")
async def refine_mouth_block(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    extra_margin: int = Form(default=10),
    parsing_mode: str = Form(default="jaw"),
    bbox_shift: int = Form(default=0),
    left_cheek_width: int = Form(default=90),
    right_cheek_width: int = Form(default=90),
    fps: int = Form(default=25, ge=10, le=60),
):
    """Refina um BLOCO de frames usando 1 template + áudio do bloco inteiro.

    Diferença pro /v1/refine_mouth: aqui retornamos TODOS os N frames gerados
    pelo MuseTalk como um TAR de JPEGs. O cliente concatenativo agrupa frames
    contíguos com sync ruim e refina em batch — produz boca temporalmente
    consistente (sem o "tremor" do refinement frame-a-frame).

    Input: 1 imagem template + N frames de áudio (N = duração * fps).
    Output: tar com N JPEGs nomeados {0000.jpg, 0001.jpg, ...}.
    """
    global _active_requests
    touch_activity(); _touch_activity()
    _active_requests += 1
    if not model_loaded:
        _active_requests = max(0, _active_requests - 1)
        raise HTTPException(503, f"Model not ready: {load_error}")

    try:
        img_bytes = await image.read()
        audio_bytes = await audio.read()

        session = _preprocess_reference(
            img_bytes=img_bytes, bbox_shift=bbox_shift,
            extra_margin=extra_margin, parsing_mode=parsing_mode,
            left_cheek_width=left_cheek_width, right_cheek_width=right_cheek_width,
        )

        import soundfile as sf
        import io as _io
        try:
            data, sr = sf.read(_io.BytesIO(audio_bytes), dtype="int16")
        except Exception as e:
            raise HTTPException(400, f"audio decode failed: {e}")
        if data.ndim > 1:
            data = data.mean(axis=1).astype(np.int16)
        if sr != 16000:
            try:
                import librosa
                f = data.astype(np.float32) / 32768.0
                f = librosa.resample(f, orig_sr=sr, target_sr=16000)
                data = (f * 32768).clip(-32768, 32767).astype(np.int16)
            except Exception as e:
                raise HTTPException(400, f"resample failed: {e}")

        frames = _render_chunk_frames(session, data.tobytes(), fps=fps)
        if not frames:
            raise HTTPException(500, "no frames generated")

        # Empacota como TAR de JPEGs
        import tarfile
        buf = _io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for i, f in enumerate(frames):
                ok, jpeg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not ok:
                    continue
                data_b = jpeg.tobytes()
                ti = tarfile.TarInfo(f"{i:04d}.jpg")
                ti.size = len(data_b)
                tar.addfile(ti, _io.BytesIO(data_b))
        return Response(content=buf.getvalue(),
                        media_type="application/x-tar",
                        headers={"X-Frames-Generated": str(len(frames))})
    finally:
        _active_requests = max(0, _active_requests - 1)
        _touch_activity()


# ── WS /v1/stream ───────────────────────────────────────────────────────────

@app.websocket("/v1/stream")
async def stream_ws(ws: WebSocket):
    """Protocolo:
      client → server (JSON init):
        {"op":"init","image":"<base64 png>","fps":25,"bbox_shift":0,
         "extra_margin":10,"parsing_mode":"jaw",
         "left_cheek_width":90,"right_cheek_width":90}
      server → client:
        {"status":"ready","session_id":"..."}
      client → server (binário): PCM s16le mono 16kHz (chunks arbitrários)
        ou JSON {"op":"flush"} / {"op":"close"}
      server → client (binário): JPEG de cada frame gerado, prefixado com
        JSON de metadata opcional.
    """
    await ws.accept()
    touch_activity()

    if not model_loaded:
        await ws.send_json({"status": "error", "error": f"modelo não carregado: {load_error}"})
        await ws.close()
        return

    # Etapa 1: receber init
    try:
        init = await ws.receive_json()
    except WebSocketDisconnect:
        return
    except Exception as e:  # noqa: BLE001
        await ws.send_json({"status": "error", "error": f"esperando init JSON: {e}"})
        await ws.close()
        return

    if init.get("op") != "init" or not init.get("image"):
        await ws.send_json({"status": "error", "error": "primeiro msg deve ser op=init com image base64"})
        await ws.close()
        return

    try:
        img_bytes = base64.b64decode(init["image"])
    except Exception as e:  # noqa: BLE001
        await ws.send_json({"status": "error", "error": f"image base64 inválida: {e}"})
        await ws.close()
        return

    fps = int(init.get("fps", 25))
    bbox_shift = int(init.get("bbox_shift", 0))
    extra_margin = int(init.get("extra_margin", 10))
    parsing_mode = init.get("parsing_mode", "jaw")
    left_cheek = int(init.get("left_cheek_width", 90))
    right_cheek = int(init.get("right_cheek_width", 90))

    try:
        session = await asyncio.get_event_loop().run_in_executor(
            None, _preprocess_reference, img_bytes, bbox_shift, extra_margin,
            parsing_mode, left_cheek, right_cheek,
        )
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, flush=True, file=sys.stderr)
        await ws.send_json({"status": "error", "error": f"preprocess falhou: {e}"})
        await ws.close()
        return

    session_id = uuid.uuid4().hex
    stream_sessions[session_id] = session
    await ws.send_json({"status": "ready", "session_id": session_id, "fps": fps,
                        "audio_format": "pcm_s16le_16k_mono"})

    # Etapa 2: loop de áudio → frames
    audio_buffer = bytearray()
    # Chunk ms configurável: gera frames quando acumular esse tanto de áudio
    chunk_ms = int(os.environ.get("MUSETALK_STREAM_CHUNK_MS", "1000"))
    chunk_bytes_target = int((16000 * 2) * chunk_ms / 1000)

    try:
        while True:
            msg = await ws.receive()
            session["last_use"] = time.time()
            if msg.get("type") == "websocket.disconnect":
                break

            # Binário → áudio
            if msg.get("bytes") is not None:
                audio_buffer.extend(msg["bytes"])
                if len(audio_buffer) >= chunk_bytes_target:
                    pcm = bytes(audio_buffer[:chunk_bytes_target])
                    del audio_buffer[:chunk_bytes_target]
                    t0 = time.time()
                    frames = await asyncio.get_event_loop().run_in_executor(
                        None, _render_chunk_frames, session, pcm, fps,
                    )
                    elapsed_ms = (time.time() - t0) * 1000
                    for f in frames:
                        ok, jpg = cv2.imencode(".jpg", f, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                        if ok:
                            await ws.send_bytes(jpg.tobytes())
                    await ws.send_json({
                        "event": "chunk_done",
                        "frames": len(frames),
                        "audio_ms": chunk_ms,
                        "inference_ms": round(elapsed_ms, 1),
                        "fps_effective": round(len(frames) / max(elapsed_ms / 1000, 1e-6), 1),
                    })
                continue

            # Texto → JSON control
            text = msg.get("text")
            if not text:
                continue
            import json
            try:
                data = json.loads(text)
            except Exception:
                continue
            op = data.get("op")
            if op == "flush":
                if len(audio_buffer) > 0:
                    pcm = bytes(audio_buffer)
                    audio_buffer.clear()
                    frames = await asyncio.get_event_loop().run_in_executor(
                        None, _render_chunk_frames, session, pcm, fps,
                    )
                    for f in frames:
                        ok, jpg = cv2.imencode(".jpg", f, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                        if ok:
                            await ws.send_bytes(jpg.tobytes())
                    await ws.send_json({"event": "flush_done", "frames": len(frames)})
            elif op == "close":
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        print(f"[musetalk] WS error: {e}\n{tb}", flush=True, file=sys.stderr)
        try:
            await ws.send_json({"status": "error", "error": str(e), "traceback": tb[-2000:]})
        except Exception:
            pass
    finally:
        stream_sessions.pop(session_id, None)
        try:
            await ws.close()
        except Exception:
            pass


# ── Demo HTML ────────────────────────────────────────────────────────────────

@app.get("/v1/demo", response_class=HTMLResponse)
async def demo():
    return """<!doctype html><html><body style="font-family:sans-serif;max-width:560px;margin:40px auto">
<h2>MuseTalk — one-shot demo</h2>
<p>Pro streaming em tempo real use WS <code>/v1/stream</code>.</p>
<form id="f" enctype="multipart/form-data" method="post" action="/v1/lipsync">
  <label>Imagem/vídeo: <input type="file" name="image" accept="image/*,video/mp4" required></label><br><br>
  <label>Áudio: <input type="file" name="audio" accept="audio/*" required></label><br><br>
  <label>FPS: <input type="number" name="fps" value="25" min="1" max="60"></label>
  <label>Batch: <input type="number" name="batch_size" value="8" min="1" max="32"></label>
  <label><input type="checkbox" name="enhance" value="true"> GFPGAN</label><br><br>
  <button>Gerar</button>
</form>
<video id="v" controls style="width:100%;margin-top:20px"></video>
<script>
document.getElementById('f').addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const r = await fetch('/v1/lipsync', { method: 'POST', body: fd });
  if (!r.ok) { alert('falhou ' + r.status); return; }
  const blob = await r.blob();
  document.getElementById('v').src = URL.createObjectURL(blob);
});
</script></body></html>"""


# ── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        log_level="info",
        access_log=False,
    )
