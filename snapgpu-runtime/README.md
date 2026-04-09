# snapgpu-runtime

Base Docker image for fast-cold-start GPU containers in the `@parle/ai-gateway` autoscaler.

## What it provides

- **CUDA 12.8.1 runtime** (Ubuntu 24.04 base) — supports NVIDIA driver 570+
- **CRIU 4.x** for CPU process checkpoint/restore
- **NVIDIA cuda-checkpoint** binary for GPU memory + CUDA context capture
- **snapgpu Python SDK** + **FastAPI control plane** preinstalled
- Snapshot directory mounted at `/var/snapgpu/snapshots`

## How it integrates with ai-gateway

This image is selected by the `SnapgpuClient` provider in
`src/gpu-providers/snapgpu-client.ts`. The autoscaler:

1. Picks an underlying GPU host provider (Vast.ai or RunPod)
2. Deploys this image instead of the per-app image
3. Polls `GET /health` until uvicorn is ready
4. POSTs `/v1/apps` to register the application spec at runtime
5. Routes `/v1/speech` (or other inference) to `POST /v1/invoke/{app}/{fn}`
6. After first successful inference, POSTs `/v1/snapshots` to capture state
7. On the next cold boot, the snapshot is restored in ~2-5s instead of ~2min

## Driver requirements

- **NVIDIA driver 570+** (required for `cuda-checkpoint` to work)
- Falls back to CPU-only CRIU when driver is older — `start.sh` detects and
  exports `SNAPGPU_DISABLE_GPU_SNAPSHOT=1`

## Build

```bash
gh workflow run build-snapgpu-runtime.yml --repo marcosremar/ai-gateway-dockers
```

The build pulls the snapgpu Python source from the parent ai-gateway repo at
build time (snapgpu/, gateway/, worker/, pyproject.toml).

## App-specific variants

To bake models into the image (faster cold boot), create a child image:

```dockerfile
FROM marcosremar/snapgpu-runtime:latest
# pre-bake Whisper + GGUF + TTS
RUN python3 -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download('bullerwins/translategemma-4b-it-GGUF', 'translategemma-4b-it-Q8_0.gguf')"
COPY app.py /app/
ENV SNAPGPU_PRELOAD_APP=babelcast
```

## Local smoke test

```bash
# Requires Linux host with NVIDIA driver 570+
docker run --gpus all -p 8000:8000 \
  -v snapgpu-data:/var/snapgpu \
  marcosremar/snapgpu-runtime:latest

curl http://localhost:8000/health
# → {"status":"ok","service":"snapgpu-gateway"}
```
