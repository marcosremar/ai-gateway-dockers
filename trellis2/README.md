# TRELLIS.2 Docker Image

FastAPI inference server for Microsoft TRELLIS.2 Image-to-3D generation,
packaged as a single container intended for GPU hosts on Vast.ai.

## Ports

| Port | Purpose                                         |
|------|-------------------------------------------------|
| 22   | sshd (for Vast.ai SSH tunnel / operator access) |
| 8000 | FastAPI inference API                            |

## Endpoints

- `POST /generate`           — Upload an image file, get a `.glb` back
- `POST /generate-from-url`  — `{"image_url": "https://..."}`, get a `.glb`
- `GET  /health`             — Health check (never 5xx; always returns 200)
- `GET  /diag`               — Full diagnostic dump (GPU/env/model/files)

## Entrypoint

The container entrypoint is **`/app/start.sh`**, not `python server.py`
directly. `start.sh`:

1. Creates `/var/log/app.log` and tees every subsequent log line into it
2. Installs any `PUBLIC_KEY` / `SSH_PUBLIC_KEY` env var into
   `/root/.ssh/authorized_keys`
3. Generates SSH host keys on first boot
4. Starts `sshd -D` in the background and verifies it's listening on `:22`
5. Runs Python preflight import checks (sys.path, torch)
6. Execs `python -u /app/server.py`, tee'd into `/var/log/app.log`

This means you can **always SSH in** and `tail -f /var/log/app.log` even if
the FastAPI server never comes up — critical for debugging on remote GPU
hosts where you can't attach to container stdout.

## Vast.ai template (important)

Vast.ai templates have a `runtype` field. When it's set to `"args"`, Vast
**ignores the Dockerfile `CMD`** and runs **only** the `onstart_cmd` from
the template. This means if your template says
`onstart_cmd: "python /app/server.py"` you will get **no sshd** and no log
teeing — which was exactly how the first deploy of this image broke.

The template's `onstart_cmd` **must** invoke `/app/start.sh`. Recreate /
update the template with:

```bash
curl -X POST 'https://console.vast.ai/api/v0/template/' \
  -H "Authorization: Bearer $VAST_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "TRELLIS.2 (Image-to-3D) v2",
    "image": "marcosremar/trellis2",
    "tag": "latest",
    "image_uuid": "marcosremar/trellis2:latest",
    "env": "-p 22:22 -p 8000:8000 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
    "onstart_cmd": "/app/start.sh",
    "runtype": "args",
    "use_jupyter_lab": false,
    "use_ssh": true,
    "extra_filters": {},
    "disk_space": 50
  }'
```

Key fields:

- `onstart_cmd: "/app/start.sh"` — the crucial fix over the old
  `"python /app/server.py"` value
- `env` — publishes both ports and sets the CUDA allocator config
- `runtype: "args"` — Vast's mode where `onstart_cmd` is the only command
  run. We MUST call `start.sh` here, because it's the script that launches
  sshd. No sshd = no SSH = no debugging.
- `use_ssh: true` — tells Vast.ai's wrapper to inject `PUBLIC_KEY` into the
  container env, which `start.sh` then appends to `authorized_keys`
- `disk_space: 50` — the TRELLIS.2 model weights are ~15 GB; 50 GB gives
  headroom for the image + model cache + generated `.glb` temp files

## Debugging a running container

If the FastAPI server isn't responding to `/health`, SSH in and check:

```bash
# Tail the full container log (start.sh output + server.py stdout/stderr)
tail -f /var/log/app.log

# Check what's listening
ss -tlnp

# Check the Python server process
ps auxf | grep -i python

# Re-run the server manually with verbose output
python -u /app/server.py
```

## Local smoke testing

The GitHub Actions workflow `build-trellis2.yml` runs a CPU-mode smoke
test after building the image that:

- Verifies `/app/start.sh`, `/app/server.py`, `/app/trellis2/o-voxel/` exist
- Verifies `/app/start.sh` is executable
- Imports the full set of modules `server.py` needs (fastapi, uvicorn,
  httpx, Pillow, torch) to catch import-time errors like the
  `total_mem → total_memory` typo before we push
- Verifies `sshd` is installed
