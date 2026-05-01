# Hunyuan3D 2.0 Docker Image

FastAPI inference server for Tencent Hunyuan3D-2 Image/Text-to-3D generation,
packaged as a single container intended for GPU hosts on Vast.ai.

## Ports

| Port | Purpose                                         |
|------|-------------------------------------------------|
| 22   | sshd (for Vast.ai SSH tunnel / operator access) |
| 8000 | FastAPI inference API                           |

## Endpoints

- `POST /generate`            — Upload image file, get `.glb` back
- `POST /generate-from-url`   — `{"image_url": "https://...", "seed": 0, "with_texture": true}`
- `POST /generate-from-text`  — `{"prompt": "a red car", "seed": 0, "with_texture": true}`
- `GET  /health`              — Health check (never 5xx)
- `GET  /diag`                — Full diagnostic dump
- `GET  /logs?lines=500`      — Tail of `/var/log/app.log`

## VRAM

| Mode               | VRAM | Suggested GPU         |
|--------------------|------|------------------------|
| shape-only         | 6 GB | RTX 3060 / 4060        |
| shape + texture    | 12 GB+ | RTX 4070 Ti / 4090 / A100 |

Set `WITH_TEXTURE=0` to skip the paint stage and stay under 8GB.

## Entrypoint

`/app/start.sh` — same dual-process pattern as trellis2:

1. Tee everything into `/var/log/app.log`
2. Install `PUBLIC_KEY` / `SSH_PUBLIC_KEY` into `authorized_keys`
3. Generate SSH host keys on first boot
4. Start `sshd -D` in background
5. Preflight: torch + CUDA import check
6. `exec python -u /app/server.py`

## Vast.ai template

Same pattern as trellis2. The template `onstart_cmd` MUST invoke
`/app/start.sh` (not `python /app/server.py` directly), otherwise sshd
never starts and you lose debug access.

```bash
curl -X POST 'https://console.vast.ai/api/v0/template/' \
  -H "Authorization: Bearer $VAST_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "Hunyuan3D 2.0 (Image/Text-to-3D)",
    "image": "marcosremar/hunyuan3d",
    "tag": "latest",
    "image_uuid": "marcosremar/hunyuan3d:latest",
    "env": "-p 22:22 -p 8000:8000 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
    "onstart_cmd": "/app/start.sh",
    "runtype": "args",
    "use_ssh": true,
    "use_jupyter_lab": false,
    "extra_filters": {},
    "disk_space": 50
  }'
```

## Deploy via ai-gateway

```bash
ai-gateway gpu deploy --image marcosremar/hunyuan3d:latest --gpu-types "RTX 4090,H100 SXM"
ai-gateway gpu wait --timeout 600
ai-gateway gpu dev exec 'curl -s localhost:8000/health'
```

First request triggers ~10GB HuggingFace download (5–10 min on a fast host).
Subsequent calls are warm.

## Debugging

```bash
ai-gateway gpu dev sh
tail -f /var/log/app.log
ss -tlnp                   # confirm :22 + :8000 listening
ps auxf | grep python
```

If `/health` returns `status: error`, hit `/diag` for the full stack trace
and last 200 log lines without SSH.
