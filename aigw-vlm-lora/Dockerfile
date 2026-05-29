# aigw-vlm-lora — prebaked image for small-VLM LoRA/QLoRA fine-tuning jobs.
#
# Why: ai-gateway finetune jobs otherwise pip-install the whole stack (incl.
# flash-attn COMPILE, 5-15 min) on every cold pod. Baking it here moves that
# cost to CI (once) → ~10-20 min saved per run, and removes the silent
# fallback-to-gpu-dev (no ML deps) failure mode.
#
# Built on gpu-dev (CUDA 12.1 + python + torch 2.4.1/cu121 + SSH + health
# server + start.sh) so dev-mode tooling (exec/sh/push/pull) still works.
#
# Use:  ai-gateway gpu finetune --type custom --image marcosremar/aigw-vlm-lora:latest ...
# Build: CI .github/workflows/build-aigw-vlm-lora.yml on push to dockers/aigw-vlm-lora/**

FROM marcosremar/gpu-dev:latest

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/workspace/.hf_cache \
    HUGGINGFACE_HUB_CACHE=/workspace/.hf_cache \
    TOKENIZERS_PARALLELISM=false

# LoRA/QLoRA stack (torch already present from gpu-dev: 2.4.1/cu121).
RUN pip install --no-cache-dir \
        "transformers>=4.49.0" \
        peft trl accelerate datasets \
        bitsandbytes \
        "huggingface-hub>=0.27" hf_transfer safetensors \
        sentencepiece einops pillow opencv-python-headless \
        qwen-vl-utils mlflow

# Unsloth (fast QLoRA for Qwen3-VL). unsloth_zoo is its runtime helper.
RUN pip install --no-cache-dir unsloth unsloth_zoo || \
    echo "[warn] unsloth install failed — TRL path still usable"

# flash-attn: compiled ONCE here in CI (the whole point). Optional so a wheel
# mismatch doesn't fail the image build; training falls back to sdpa attention.
RUN pip install --no-cache-dir flash-attn --no-build-isolation || \
    echo "[warn] flash-attn unavailable — using sdpa attention"

# Materialize .pyc + surface import errors at build time, not run time.
RUN python -c "import torch, transformers, peft, trl, bitsandbytes, datasets; \
print('[bake] vlm-lora stack ok — torch', torch.__version__)"

RUN mkdir -p /workspace/.hf_cache
WORKDIR /workspace
