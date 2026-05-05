# qwen3-tts — Multilingual TTS + Voice Cloning (ai-gateway)

Wrapper of Alibaba Qwen3-TTS following the `ai-gateway-dockers/`
convention used by `kokoro-tts` and `musetalk`:

- `/health` for the autoscaler readiness probe
- `idle_watchdog.py` (auto-shutdown after `IDLE_TIMEOUT_MIN` minutes idle)
- HF cache persisted to `/workspace` when mounted
- OpenAI-compatible **`POST /v1/audio/speech`** (drop-in for kokoro-tts)
- **`POST /v1/audio/speech/clone`** — multipart voice-cloning with
  `reference_audio` + `ref_text`
- `GET /v1/audio/voices` voice catalog
- sshd (PUBKEY auth) for SSH-patch iteration

## Build

```bash
cd ai-gateway-dockers/qwen3-tts
docker build -t marcosremar/qwen3-tts:latest .
```

## Run (NVIDIA GPU required, ~16 GB VRAM)

```bash
docker run --gpus all -p 8000:8000 -p 22:22 \
  -v $PWD/_cache/hf:/workspace/.cache/huggingface \
  -e IDLE_TIMEOUT_MIN=15 \
  marcosremar/qwen3-tts:latest
```

## Smoke test (preset voice)

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello from Qwen3","voice":"Ryan","response_format":"wav"}' \
  -o out.wav
```

## Voice cloning

```bash
curl -X POST http://localhost:8000/v1/audio/speech/clone \
  -F text="Narration text in the cloned voice." \
  -F ref_text="Transcript of the reference clip." \
  -F reference_audio=@reference.wav \
  -o cloned.wav
```

## Deploy via ai-gateway

```bash
ai-gateway gpu deploy \
  --image marcosremar/qwen3-tts:latest \
  --gpu-types "NVIDIA RTX 4090,NVIDIA A100" \
  --label canal-dark/qwen3-tts \
  --storage-gb 60 \
  --readiness-probe ssh \
  --allow-unverified
```
