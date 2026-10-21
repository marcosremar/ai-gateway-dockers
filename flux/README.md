# flux — FLUX.1 text-to-image (open-source)

Self-hosted Black Forest Labs FLUX.1 served behind ai-gateway. OpenAI-compatible `/v1/images/generations` endpoint so canal-dark / project-philosofi can route image gen identically to other providers.

## Models supported

| Model | License | Steps | VRAM | Notes |
|---|---|---|---|---|
| `black-forest-labs/FLUX.1-schnell` | **Apache 2.0** ✅ | 4 | ~22GB | Default. Comercial use OK. Fast (~3s/image @ 1024). |
| `black-forest-labs/FLUX.1-dev` | FLUX-dev (non-comm) | 28-50 | ~24GB | Higher quality. Skip pra commercial. |

Toggle via `FLUX_MODEL` env.

## Build

```bash
cd ai-gateway-dockers/flux
docker build -t marcosremar/flux:latest .
```

## Run (local GPU)

```bash
docker run --gpus all -p 8000:8000 -p 22:22 \
  -v /workspace/hf-cache:/workspace/.cache/huggingface \
  -e IDLE_TIMEOUT_MIN=15 \
  -e FLUX_MODEL=black-forest-labs/FLUX.1-schnell \
  marcosremar/flux:latest
```

## Run (Vast.ai via ai-gateway)

```bash
ai-gateway gpu deploy --image marcosremar/flux:latest --gpu RTX_4090
# returns: { url: https://xxx.proxy.vast.ai, port: 8000 }
```

## API

```bash
curl http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "blueprint diagram of stoic dichotomy of control, deep blue, technical, line art",
    "size": "1024x1024",
    "n": 1,
    "guidance_scale": 0.0,
    "seed": 42
  }'
```

Response: `{ data: [{ b64_json: "..." }] }`.

## Use cases — project-philosofi patterns

| Pattern | Prompt template | Settings |
|---|---|---|
| 1 (Cultural Bridge) | "single line drawing, beige chalk on dark navy chalkboard, minimal, [concept]" | schnell, 4 steps |
| 4 (Blueprint Stoic) | "blueprint technical diagram, [concept], deep blue background, white line art, callouts with kN annotations" | schnell, 4 steps |
| 6 (Biographical) | uses Wikimedia archive — no FLUX needed | — |
| 7 (Cold Open Mystery) | "matrix code falling, dark teal, cinematic, photorealistic" | schnell |
| 9 (Whiteboard) | "rough hand-drawn whiteboard sketch, [concept], doodle style, color accents" | schnell |

## Why open source

Per project requirement: **all generation must be open-source self-hosted**. Closed APIs (Midjourney, DALL-E, HeyGen) replaced by:

- FLUX.1-schnell (text-to-image) — Apache 2.0 ✅
- Qwen3-TTS (text-to-speech) — Apache 2.0 ✅ (already in ai-gateway-dockers/qwen3-tts)
- Wan I2V (image-to-video) — already in ai-gateway-dockers/wan-i2v
- MuseTalk (lipsync, optional) — already in ai-gateway-dockers/musetalk (skip pra Pattern 2 for now)

## License

Dockerfile/server: Apache 2.0. FLUX weights: see upstream license per model.
