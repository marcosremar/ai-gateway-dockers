# unreal-desktop

Remote GPU desktop image for Unreal Engine on Vast.ai (or any nvidia-docker host).

## Stack

- Base: `gezp/ubuntu-desktop:22.04-cu12.2.2`
  - Ubuntu 22.04 LTS, CUDA 12.2, XFCE
  - NoMachine (port 4000) + noVNC + KasmVNC selectable via `REMOTE_DESKTOP` env
- Unreal Engine build/runtime deps: clang, lld, cmake, ninja, Vulkan, GL, audio libs
- SSH daemon (port 22) for `ai-gateway gpu dev exec/sh`
- Health endpoint (port 8000) for idle watchdog

## Build/push

CI pushes on changes to `unreal-desktop/**`. Tags:
- `marcosremar/unreal-desktop:latest`
- `marcosremar/unreal-desktop:<sha>`
- `ghcr.io/marcosremar/unreal-desktop:latest`

## Deploy

```bash
ai-gateway gpu deploy \
  --image marcosremar/unreal-desktop:latest \
  --gpu-types "NVIDIA GeForce RTX 4090,NVIDIA GeForce RTX 3090" \
  --label unreal-desktop
ai-gateway gpu wait --timeout 600
```

## Access

```bash
# Shell (preferred — goes through gateway agent)
ai-gateway gpu dev sh
ai-gateway gpu dev exec 'nvidia-smi'
ai-gateway gpu dev exec 'curl -s localhost:8000/info'

# NoMachine viewport: forward port 4000 via gateway, connect with NoMachine client
# Default user/pass: ubuntu / ubuntu (override with -e USER=... -e PASSWORD=...)
```

## Install Unreal Engine (one-time, snapshot after)

```bash
ai-gateway gpu dev exec 'git clone --depth 1 -b 5.4 https://<token>@github.com/EpicGames/UnrealEngine /root/UnrealEngine'
ai-gateway gpu dev exec 'cd /root/UnrealEngine && ./Setup.sh && ./GenerateProjectFiles.sh && make -j$(nproc)'
ai-gateway gpu dev snapshot -m "unreal 5.4 compiled"
```

## Recommended GPU

| Use case             | GPU                  | VRAM | Why                                       |
|----------------------|----------------------|------|-------------------------------------------|
| Cheapest viable      | RTX 3060 12GB        | 12   | OK for small scenes, Lumen low            |
| Best $/perf 2026     | RTX 3090             | 24   | 24GB VRAM = comfortable Nanite/Lumen      |
| Best raw perf $/h    | RTX 4090             | 24   | Path tracing, large scenes                |
| Production rendering | A6000 / L40S         | 48   | Multi-scene, MovieRender 4K-8K            |
