# gpu-dev — Interactive GPU development base image

Minimal image for iterating on GPU code before baking a project-specific Docker image.

## Contents

- Ubuntu 22.04 + CUDA 12.1 + cuDNN
- Python 3.10 with pip, uv, numpy, torch 2.4.1+cu121, fastapi, huggingface_hub
- SSH server (`sshd`) on port 22
- Health/info server on port 8000
- Common tools: git, ffmpeg, tmux, htop, vim, build-essential

## Target size

~2.5 GB — pulls in ~1-2 min on typical Vast.ai instances (vs 15+ min for 8 GB project images).

## Typical workflow

```bash
bunx ai-gateway gpu dev start           # deploy this image
bunx ai-gateway gpu dev sh              # interactive shell
bunx ai-gateway gpu dev exec "pip install ..."
bunx ai-gateway gpu dev push model.py /app/model.py
bunx ai-gateway gpu dev pull /app/out.json ./out.json
bunx ai-gateway gpu dev snapshot prompthmr:v1   # commit + push image
bunx ai-gateway gpu stop                 # pause (preserve state for quick resume)
```

## Build

Push via CI only (never locally — see CLAUDE.md):
```
marcosremar/gpu-dev:latest
```
