# HY-Motion-1.0 — text-to-3D-motion on RTX 5090

Wraps Tencent Hunyuan's [HY-Motion-1.0](https://github.com/Tencent-Hunyuan/HY-Motion-1.0) DiT motion generator in a FastAPI server, packaged for ai-gateway Vast.ai deployment on Blackwell GPUs (RTX 5090).

## Output formats

- `bvh` — text BVH skeletal animation, easy to retarget in Blender or load into Babylon/Three via converters.
- `gltf` — binary GLB ready to drop into the avatar-engine animation list (`B.SceneLoader.ImportAnimationsAsync`).

FBX export is **not** supported (the upstream code requires the proprietary Autodesk `fbxsdkpy` SDK, which we don't bundle to keep the image friendly to all build hosts).

## Why RTX 5090

| Component | VRAM (fp16) |
|---|---|
| HY-Motion-1.0 DiT generator (1.0B params) | ~2 GB |
| CLIP-L sentence encoder | ~1 GB |
| Qwen3-8B LLM text encoder | ~16 GB |
| Diffusion intermediate activations | 2-4 GB |
| PyTorch overhead + buffers | 1-2 GB |
| **Total** | **~22-25 GB** |

The full HY-Motion-1.0 needs ~24 GB VRAM. RTX 5090 has 32 GB → comfortable headroom for batching and longer durations. The 0.46B "Lite" variant uses the same Qwen3-8B encoder so it doesn't actually save much VRAM — we ship the full model.

## Build

```bash
cd dockers/hy-motion
docker build -t marcosremar/hy-motion:latest .
docker push marcosremar/hy-motion:latest
```

The build downloads ~22 GB of model weights into the image (`HY-Motion-1.0`, `clip-vit-large-patch14`, `Qwen3-8B`). The resulting image is around 50 GB extracted. Vast.ai disk allocation should be **>=120 GB** to leave room for the layered FS plus runtime output.

The Dockerfile uses `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` + PyTorch 2.7 + CUDA 12.8 — first combination with full RTX 5090 (sm_120) kernel support. **Do not downgrade** unless you also drop the GPU target.

## Deploy via ai-gateway

```bash
bun run scripts/deploy-hy-motion.ts deploy        # boot RTX 5090 instance
bun run scripts/deploy-hy-motion.ts status        # check readiness
bun run scripts/deploy-hy-motion.ts test          # run a sample inference
bun run scripts/deploy-hy-motion.ts terminate     # shut down
```

Cold start on Vast.ai: image pull ~5-10 min (~25 GB compressed) + model load ~2-4 min = **first request ready in 7-14 min**. Subsequent requests on the same instance: **30-90 s per generation** depending on duration.

## API

### POST /predict
```json
{
  "text": "a person walks forward then waves their right hand",
  "duration": 5.0,
  "cfg_scale": 5.0,
  "num_seeds": 1,
  "format": "gltf"
}
```

Response:
```json
{
  "status": "ok",
  "format": "gltf",
  "data_base64": "Z2xURgIAAAA...",
  "duration": 5.0,
  "rewritten_text": "a person walks forward then waves their right hand",
  "latency_ms": 32450.7
}
```

### GET /health
```json
{ "status": "ok", "device": "cuda", "model_path": "/app/HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0" }
```

While loading: `{ "status": "loading", "device": "cuda" }` — gateway should poll until `ok`.

## Limitations

- **English-only prompts** (≤ 60 words). Multilingual support requires the `Text2MotionPrompter` LLM rewriter, which we don't pre-bake (would add ~16 GB).
- **No hand/finger animation** — HY-Motion-1.0 generates body motion only. Use the avatar-engine hand override (`OVERRIDE_UPPER_FROM_C3D_IDLE`) to layer a static hand pose on top.
- **No looping or in-place animations** — output is one-shot trajectory.
- **No multi-person interactions or non-humanoid characters.**
