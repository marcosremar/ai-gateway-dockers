"""
MuseTalk Service — FastAPI wrapper pro ai-gateway.

Endpoints:
  GET  /health                   — status GPU + modelos
  POST /v1/lipsync               — (multipart) image + audio → MP4
  GET  /v1/demo                  — HTML mínimo pra teste manual
  WS   /v1/stream                — (opcional v2) áudio PCM stream → frames

Modelo: TMElyralab/MuseTalk V1.5 (latent-diffusion lip sync).
Compat: herda o pipeline do fork ruxir-ig/MuseTalk-API (montado em /app).

Env:
  IDLE_TIMEOUT_MIN   — auto-shutdown depois de inativo (default 15; 0=off)
  MUSETALK_MODELS_DIR— diretório de pesos (default /app/models)
"""

import asyncio
import base64
import io
import os
import sys
import tempfile
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel
import uvicorn

# idle_watchdog fica ao lado do server.py, importação local
from idle_watchdog import add_idle_middleware, start_watchdog, touch_activity

# MuseTalk-API (fork montado em /app/musetalk_api)
sys.path.insert(0, "/app")

# Inference engine do fork — lazy import para não quebrar o /health antes dos
# pesos baixarem.
inference_engine = None
load_error: Optional[str] = None
load_traceback: Optional[str] = None
model_loaded = False


def _load_inference_engine():
    global inference_engine, load_error, load_traceback, model_loaded
    try:
        print("[musetalk] Carregando MuseTalkInference...", flush=True)
        t0 = time.time()
        from api.inference_service import MuseTalkInference  # type: ignore

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
    # Carrega o modelo em thread separada pra não bloquear startup HTTP
    print("[musetalk] Startup — iniciando carga de modelos em background...", flush=True)
    asyncio.get_event_loop().run_in_executor(None, _load_inference_engine)
    asyncio.create_task(start_watchdog())
    yield
    print("[musetalk] Shutdown", flush=True)


app = FastAPI(
    title="MuseTalk Service",
    description="Real-time audio-driven lip-sync para o ai-gateway.",
    version="0.1.0",
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


@app.get("/health", response_model=HealthResponse)
async def health():
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    gpu_vram_gb = None
    if gpu_available:
        gpu_vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1)

    if load_error:
        return HealthResponse(
            status="error",
            model_loaded=False,
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            gpu_vram_gb=gpu_vram_gb,
            load_error=load_error,
        )
    if not model_loaded:
        return HealthResponse(
            status="loading",
            model_loaded=False,
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            gpu_vram_gb=gpu_vram_gb,
        )
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
    )


# ── Lip sync (one-shot, file-based) ──────────────────────────────────────────

@app.post("/v1/lipsync")
async def lipsync(
    image: UploadFile = File(..., description="Imagem/vídeo de referência (PNG/JPG/MP4)"),
    audio: UploadFile = File(..., description="Áudio WAV/MP3 que vai guiar os lábios"),
    enhance: bool = Form(default=False, description="GFPGAN face enhancement"),
    fps: int = Form(default=25, ge=1, le=60),
    batch_size: int = Form(default=8, ge=1, le=32),
    extra_margin: int = Form(default=10, ge=0, le=40),
):
    """Gera um MP4 com lip-sync. Bloqueia até terminar — pra streaming use WS."""
    touch_activity()
    if not model_loaded:
        raise HTTPException(status_code=503, detail=f"Modelo ainda carregando ou falhou: {load_error}")

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
            # MuseTalkInference.generate retorna o path do MP4 gerado
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

        # Lê o arquivo na memória antes de deixar o TemporaryDirectory ser deletado.
        # (StreamingResponse lazy não funciona aqui — o tmp é apagado ao sair do `with`.)
        data = mp4_path.read_bytes()

        headers = {
            "X-Elapsed-Seconds": f"{elapsed:.2f}",
            "Content-Disposition": f'attachment; filename="{out_name}.mp4"',
        }
        return Response(content=data, media_type="video/mp4", headers=headers)


# ── Demo HTML mínimo ─────────────────────────────────────────────────────────

@app.get("/v1/demo", response_class=HTMLResponse)
async def demo():
    return """<!doctype html><html><body style="font-family:sans-serif;max-width:560px;margin:40px auto">
<h2>MuseTalk demo</h2>
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


# ── WebSocket streaming (placeholder v2) ─────────────────────────────────────

@app.websocket("/v1/stream")
async def stream_ws(ws: WebSocket):
    """Stub: recebe PCM 16kHz mono s16le em binário, emitirá frames JPEG.
    A implementação completa depende de adaptar `MuseTalkInference` para
    processar buffers de 1s em modo online. Por ora apenas fecha o socket
    depois de aceitar — placeholder para iteração seguinte."""
    await ws.accept()
    touch_activity()
    await ws.send_json({"status": "not_implemented_yet", "hint": "use POST /v1/lipsync"})
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass


# ── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        log_level="info",
        access_log=False,
    )
