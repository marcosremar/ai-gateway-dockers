# musetalk — Real-Time Lip-Sync Service (ai-gateway)

Wrapper do [MuseTalk](https://github.com/TMElyralab/MuseTalk) (V1.5) seguindo
o padrão `ai-gateway-dockers/`. Usa o fork
[ruxir-ig/MuseTalk-API](https://github.com/ruxir-ig/MuseTalk-API) como base
do pipeline de inferência e adiciona:

- `/health` compatível com o autoscaler.
- `idle_watchdog.py` (auto-shutdown após 15 min inativo).
- HF cache em `/workspace/.cache/huggingface` para persistir entre restarts.
- Endpoint `POST /v1/lipsync` multipart (imagem + áudio → MP4).
- Demo HTML em `GET /v1/demo`.
- Placeholder `WS /v1/stream` (PCM in → frames out — iteração v2).

## Build

```bash
cd ai-gateway-dockers/musetalk
docker build -t marcosremar/musetalk:latest .
```

## Run local (requer GPU NVIDIA)

```bash
docker run --gpus all -p 8000:8000 \
  -v $PWD/_cache/hf:/workspace/.cache/huggingface \
  -v $PWD/_cache/models:/workspace/musetalk-models \
  -e IDLE_TIMEOUT_MIN=0 \
  marcosremar/musetalk:latest
```

Primeira subida: download de ~4GB de pesos (MuseTalk + SD-VAE + whisper-tiny
+ DWPose + face-parse). Se der para pré-popular o volume, faz diferença no
cold start.

## Hardware

| GPU       | VRAM | 256x256 FPS | Notas |
|-----------|------|-------------|-------|
| RTX A6000 | 48GB | 30+         | Recomendada — folga pra 512x512 e batch 16 |
| RTX A4000 | 16GB | ~25-30      | Funciona, batch menor |
| V100      | 16GB | 30+         | Referência do paper |
| RTX 3060  | 8GB  | ~25         | Mínimo funcional |

## Test rápido

### One-shot (file-based)

```bash
curl -F "image=@portrait.png" -F "audio=@speech.wav" \
  -o out.mp4 http://localhost:8000/v1/lipsync
```

### Streaming WS (real-time)

Cliente Python incluso — mede FPS + RTF:

```bash
pip install websockets librosa
python test_stream.py --server ws://localhost:8000 \
  --image portrait.png --audio speech.wav --chunk-ms 1000
```

Protocolo do WS `/v1/stream`:

1. Cliente envia JSON init:
   ```json
   {"op":"init","image":"<base64 png>","fps":25,"extra_margin":10,"parsing_mode":"jaw"}
   ```
2. Server responde `{"status":"ready","session_id":"..."}` (preprocessing da imagem bloqueia aqui, ~1-2s)
3. Cliente envia PCM s16le mono 16kHz em binário (chunks arbitrários)
4. Server acumula até `MUSETALK_STREAM_CHUNK_MS` (default 1000ms) e emite frames JPEG binários
5. `{"op":"flush"}` — processa o residual; `{"op":"close"}` — fecha

Otimização: face detection + VAE encode da referência rodam uma vez no
init e ficam cacheados. Os chunks só rodam whisper → PE → UNet → VAE
decode → blend — o hot path que o paper cita como 30fps+ em V100.
