# latentsync — Audio-Conditioned Latent Diffusion Lip-Sync (ai-gateway)

Wrapper of [ByteDance LatentSync](https://github.com/bytedance/LatentSync)
following the `ai-gateway-dockers/` conventions:

- `/health` for the autoscaler readiness probe
- `idle_watchdog.py` (auto-shutdown after `IDLE_TIMEOUT_MIN` minutes idle)
- HF cache + checkpoints persisted to `/workspace` when mounted
- `POST /v1/lipsync` multipart (image OR mp4 + wav/mp3 → mp4),
  drop-in compatible with `marcosremar/musetalk` so the gateway pipeline
  can swap engines without caller changes
- `GET /v1/demo` for manual smoke tests
- sshd (PUBKEY auth) so the gateway can SSH-patch the running pod

## Build

```bash
cd ai-gateway-dockers/latentsync
docker build -t marcosremar/latentsync:latest .
```

## Run (NVIDIA GPU required)

```bash
docker run --gpus all -p 8000:8000 -p 22:22 \
  -v $PWD/_cache/hf:/workspace/.cache/huggingface \
  -v $PWD/_cache/checkpoints:/workspace/latentsync-checkpoints \
  -e LATENTSYNC_VARIANT=1.5 \
  -e IDLE_TIMEOUT_MIN=15 \
  marcosremar/latentsync:latest
```

`LATENTSYNC_VARIANT`:
- `1.5` — 8 GB VRAM (RTX 3070 / 4060 Ti / 4070+)
- `1.6` — 18 GB VRAM, 512×512 quality (RTX 4090 / A100+)

## Smoke test

```bash
curl -F image=@avatar.png -F audio=@narration.wav \
  http://localhost:8000/v1/lipsync -o out.mp4
```

## Deploy via ai-gateway

```bash
ai-gateway gpu deploy \
  --image marcosremar/latentsync:latest \
  --gpu-types "NVIDIA RTX 4090,NVIDIA A100" \
  --label canal-dark/latentsync \
  --storage-gb 50 \
  --readiness-probe ssh \
  --allow-unverified
```

## Why LatentSync vs MuseTalk

| Axis | MuseTalk | LatentSync |
|------|----------|-----------|
| LSE-C (HDTF) | ~good | **7.90 (SOTA)** |
| Real-time | ✅ 30+ FPS | ❌ non-real-time |
| VRAM (min) | 8 GB | 8 GB (1.5) / 18 GB (1.6) |
| Resolution ceiling | 256×256 | 512×512 (1.6) |
| Best for | Live / faceless YT cam overlay | Premium dubbing, rendered asset |

Pipelines that already POST to MuseTalk's `/v1/lipsync` can route the same
multipart request here for higher-quality offline renders.
