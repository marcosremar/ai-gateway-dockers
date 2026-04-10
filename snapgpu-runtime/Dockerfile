# ── snapgpu-runtime — base image with CRIU + cuda-checkpoint ────────────────
# This image provides the snapgpu control plane (FastAPI gateway) with the
# tools needed for CPU+GPU snapshot/restore. Concrete app images (e.g.
# snapgpu-runtime-babelcast) extend this base and bake in models.
#
# Key components:
#   - CUDA 12.8.1 devel (devel needed for CRIU's cuda_plugin.so build)
#   - CRIU 4.0 compiled from source (Ubuntu 24.04 dropped the criu package)
#   - CRIU's cuda_plugin.so for GPU memory checkpoint (uses NVIDIA driver 570+ API)
#   - snapgpu Python SDK + gateway control plane
#
# Validated on: Vast.ai RTX 5080 (driver 580.126), RTX 3060 Ti (driver 570+)
# Smoke test: 2026-04-09 — CRIU compiles clean, cuda_plugin.so auto-generated
#
# Boot lifecycle:
#   1. Container starts → uvicorn gateway.main:app on port 8000
#   2. ai-gateway POSTs /v1/apps to register an App spec
#   3. ai-gateway POSTs /v1/invoke/{app}/{fn} to run inference
#   4. After model load, ai-gateway POSTs /v1/snapshots to snap the process
#   5. On next cold boot of the same container, snapgpu restores from snapshot
# ─────────────────────────────────────────────────────────────────────────────

# Use devel image (not runtime) — CRIU's CUDA plugin needs CUDA headers to compile
ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE} AS criu-builder

# ── Build CRIU 4.0 from source ────────────────────────────────────────────
# Ubuntu 24.04 (Noble) removed the criu package. We compile from the official
# v4.0 tag which includes the cuda_plugin.so for GPU memory checkpointing.
# The plugin uses NVIDIA's cuda-checkpoint API (driver 570+) under the hood.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ make git pkg-config \
        libprotobuf-dev libprotobuf-c-dev protobuf-c-compiler protobuf-compiler \
        libnl-3-dev libnet1-dev libcap-dev python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN cd /tmp \
    && git clone --depth 1 --branch v4.0 https://github.com/checkpoint-restore/criu.git \
    && cd criu \
    && make -j$(nproc) \
    && mkdir -p /opt/criu/bin /opt/criu/lib \
    && cp criu/criu /opt/criu/bin/ \
    && cp -r lib/pycriu /opt/criu/lib/ \
    # cuda_plugin.so is auto-built when CUDA headers are detected
    && (cp plugins/cuda/cuda_plugin.so /opt/criu/lib/ 2>/dev/null || true) \
    && cd / && rm -rf /tmp/criu

# ── Install cuda-checkpoint (called by cuda_plugin.so during GPU memory dump/restore) ──
# cuda_plugin.so forks cuda-checkpoint at runtime to freeze/restore the CUDA context.
# Without it in PATH, criu dump fails with "check that cuda-checkpoint is present in $PATH".
# Validated: 570.x binary works with driver 570+ (tested on TensorDock RTX 3090, driver 570.144).
RUN cd /tmp \
    && git clone --depth 1 https://github.com/NVIDIA/cuda-checkpoint.git \
    && cp cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint /opt/criu/bin/ \
    && chmod +x /opt/criu/bin/cuda-checkpoint \
    && rm -rf /tmp/cuda-checkpoint

# ── Final image ───────────────────────────────────────────────────────────
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    SNAPGPU_DB_URL=sqlite:////var/snapgpu/snapgpu.db \
    SNAPGPU_PORT=8000 \
    SNAPGPU_SNAPSHOT_DIR=/var/snapgpu/snapshots \
    HF_XET_HIGH_PERFORMANCE=1 \
    HF_XET_FIXED_DOWNLOAD_CONCURRENCY=50

# ── Runtime packages ──────────────────────────────────────────────────────
# CRIU runtime deps (protobuf-c, libnl, libnet, libcap) + Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev python3-pip \
        libprotobuf-c1 libprotobuf32t64 \
        libnet1 libnl-3-200 libcap2-bin \
        curl ca-certificates \
        ffmpeg libsndfile1 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# ── Copy CRIU binaries from builder stage ──────────────────────────────────
COPY --from=criu-builder /opt/criu/bin/criu /usr/local/bin/criu
COPY --from=criu-builder /opt/criu/bin/cuda-checkpoint /usr/local/bin/cuda-checkpoint
COPY --from=criu-builder /opt/criu/lib/ /usr/local/lib/criu/
RUN chmod +x /usr/local/bin/criu /usr/local/bin/cuda-checkpoint \
    # Verify CRIU works at build time (fails fast if deps are missing)
    && criu --version \
    # Install cuda_plugin.so where CRIU expects it (CRIU scans /usr/lib/criu/ for plugins)
    && mkdir -p /usr/lib/criu \
    && (cp /usr/local/lib/criu/cuda_plugin.so /usr/lib/criu/ 2>/dev/null || echo "No cuda_plugin.so — GPU snapshots disabled")

# ── Snapshot directory (mount as volume in production for persistence) ────
RUN mkdir -p /var/snapgpu/snapshots /var/snapgpu/db

# ── Install snapgpu Python package ────────────────────────────────────────
WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
COPY snapgpu/ /app/snapgpu/
COPY gateway/ /app/gateway/
COPY worker/ /app/worker/
# Install boto3 explicitly before -e install to avoid layer cache misses
# where an older pyproject.toml (without boto3) was cached
RUN pip install --no-cache-dir "boto3>=1.34.0"
RUN pip install --no-cache-dir -e /app

# ── Healthcheck ───────────────────────────────────────────────────────────
# Note: ai-gateway prefers no Docker HEALTHCHECK on Vast.ai images (it can
# trigger auto-destroy during model loads). We expose /health on the app
# instead and let the autoscaler poll it externally.

EXPOSE 8000

# Entrypoint runs the FastAPI gateway. Apps are registered at runtime by the
# ai-gateway via POST /v1/apps; this base image carries no app-specific code.
COPY start.sh /usr/local/bin/snapgpu-start
RUN chmod +x /usr/local/bin/snapgpu-start

CMD ["/usr/local/bin/snapgpu-start"]
