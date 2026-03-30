#!/bin/bash
# Ultravox S2S — Smart startup with GPU auto-detection
#
# Detects GPU architecture and driver version at startup.
# If Blackwell (RTX 5090) + driver 570+ detected, upgrades PyTorch to cu128
# for optimal performance. Otherwise uses cu124 (pre-installed, works everywhere).

set -e

echo "============================================================"
echo "  Ultravox Speech-to-Speech — Smart GPU Setup"
echo "============================================================"
echo ""

# ── GPU Detection ────────────────────────────────────────────────────────────

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "none")
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "0")
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
DRIVER_MAJOR=$(echo "$DRIVER_VERSION" | cut -d. -f1)

echo "GPU:      $GPU_NAME"
echo "VRAM:     ${GPU_MEMORY}MB"
echo "Driver:   $DRIVER_VERSION"
echo "Python:   $(python3 --version 2>&1)"

# Detect architecture
GPU_ARCH="unknown"
if echo "$GPU_NAME" | grep -qi "5090\|5080\|5070\|B200\|B100\|GB200\|Blackwell"; then
    GPU_ARCH="blackwell"
elif echo "$GPU_NAME" | grep -qi "4090\|4080\|4070\|4060\|L4\|L40\|Ada"; then
    GPU_ARCH="ada"
elif echo "$GPU_NAME" | grep -qi "3090\|3080\|3070\|A100\|A10\|A40\|Ampere"; then
    GPU_ARCH="ampere"
elif echo "$GPU_NAME" | grep -qi "H100\|H200\|Hopper"; then
    GPU_ARCH="hopper"
fi
echo "Arch:     $GPU_ARCH"

# ── PyTorch CUDA Optimization ────────────────────────────────────────────────

CURRENT_CUDA=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "none")
echo "PyTorch CUDA: $CURRENT_CUDA"

# Upgrade to cu128 if Blackwell/Hopper + driver 570+ and currently on cu124
if [ "$DRIVER_MAJOR" -ge 570 ] 2>/dev/null; then
    if echo "$GPU_ARCH" | grep -qE "blackwell|hopper"; then
        if echo "$CURRENT_CUDA" | grep -q "12.4"; then
            echo ""
            echo ">>> Blackwell/Hopper GPU with driver $DRIVER_VERSION detected."
            echo ">>> Upgrading PyTorch to cu128 for optimal performance..."
            # Use cached wheels if available (Vast.ai network volumes)
            CACHE_DIR="/workspace/.pip-cache"
            mkdir -p "$CACHE_DIR" 2>/dev/null || CACHE_DIR="/tmp/pip-cache"
            mkdir -p "$CACHE_DIR"
            pip install --no-cache-dir --cache-dir "$CACHE_DIR" -q \
                torch torchvision torchaudio \
                --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -3
            NEW_CUDA=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "failed")
            echo ">>> PyTorch CUDA: $CURRENT_CUDA → $NEW_CUDA"
        else
            echo "PyTorch already on optimal CUDA version ($CURRENT_CUDA)"
        fi
    else
        echo "Driver $DRIVER_VERSION supports cu128 but GPU is $GPU_ARCH — cu124 is fine."
    fi
else
    echo "Driver $DRIVER_VERSION — using pre-installed cu124 (compatible)."
fi

# ── Verify CUDA works ────────────────────────────────────────────────────────

echo ""
CUDA_OK=$(python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    vram = (props.total_memory if hasattr(props, 'total_memory') else props.total_mem) / 1024**3
    print(f'CUDA OK: {name} (sm_{cap[0]}{cap[1]}, {vram:.1f}GB)')
else:
    print('CUDA FAILED')
" 2>&1)
echo "$CUDA_OK"

if echo "$CUDA_OK" | grep -q "FAILED"; then
    echo ""
    echo "WARNING: CUDA not available! Models will not load."
    echo "This usually means the host driver ($DRIVER_VERSION) is too old for the CUDA toolkit."
    echo "Minimum driver: 525+ for cu124, 570+ for cu128"
    echo "Continuing anyway (server will report error via /health)..."
fi

# ── HuggingFace Setup ────────────────────────────────────────────────────────

echo ""
export HF_HOME="/app/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/app/.cache/huggingface/hub"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"

# Network Volume symlinks (Vast.ai, RunPod)
if [ -d "/workspace/models" ] && [ ! -L "/workspace/models" ]; then
    for model_dir in /workspace/models/models--*; do
        [ -d "$model_dir" ] || continue
        base=$(basename "$model_dir")
        target="$HF_HOME/hub/$base"
        [ -e "$target" ] || ln -sfn "$model_dir" "$target"
    done
    echo "Linked $(ls -d /workspace/models/models--* 2>/dev/null | wc -l | tr -d ' ') cached models from /workspace/models"
fi
ln -sfn /app/.cache/huggingface /workspace/huggingface 2>/dev/null || true

# ── Ensure TTS deps ──────────────────────────────────────────────────────────
# Skip pip install when running on a pre-built image (marker /app/.prebuilt exists).
# Pre-built images (Dockerfile.blackwell) have all deps pre-installed — installing
# again wastes 2-5 min and can fail on slow/no-internet hosts.

if [ -f /app/.prebuilt ]; then
    echo "Pre-built image detected — skipping pip install (deps already installed)"
else
    pip install --no-cache-dir -q "transformers>=4.45.0" "accelerate>=1.12.0" librosa einops onnxruntime sox 2>&1 | tail -3
    pip install --no-cache-dir -q --no-deps "faster-qwen3-tts>=0.2.4" 2>&1 | tail -3
fi

# ── Start Server ─────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  Starting Ultravox S2S on port 8000"
echo "  GPU: $GPU_NAME ($GPU_ARCH) | Driver: $DRIVER_VERSION"
echo "============================================================"
echo ""
cd /app/api
exec python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info
