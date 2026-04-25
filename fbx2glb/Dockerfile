# ── fbx2glb — Blender headless FBX→GLB conversion service ───────────────────
# CPU-only (no GPU / CUDA). Wraps Blender 4.5 headless + the winning
# convert.sh from the autoresearch/fbx2glb research branch: Blender default
# import/export plus a POSITION rebase + inverse-bind-matrix compensation
# post-process that aligns coordinate frames with Khronos-style GLBs.
#
# API:
#   GET  /health        — { ok, blender, version }
#   POST /v1/convert    — multipart FBX upload → GLB binary response
#   GET  /              — simple HTML help
#
# Build:
#   docker build -t marcosremar/fbx2glb:latest .
#
# Run:
#   docker run -p 8000:8000 marcosremar/fbx2glb:latest
#
# Size budget: Blender tarball ~250MB + Python deps ~30MB. Final image
# is roughly ~500MB which fits comfortably in Fly.io's 8GB app image cap.
# ─────────────────────────────────────────────────────────────────────────────

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    IDLE_TIMEOUT_MIN=5 \
    BLENDER_VERSION=4.5.0 \
    PORT=8000

# ── System deps ─────────────────────────────────────────────────────────────
# Blender headless on Ubuntu 22.04 needs: libxi6, libxrender1, libxfixes3,
# libgl1-mesa-glx (for headless OpenGL context init even without display),
# libsm6, libxkbcommon0, libsndfile1 (used by some bpy ops).
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3-pip \
      curl xz-utils ca-certificates \
      libxi6 libxrender1 libxfixes3 libxkbcommon0 \
      libgl1 libsm6 libxext6 libglu1-mesa libsndfile1 \
      libfreetype6 libtbb12 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# ── Blender 4.5 (official tarball) ──────────────────────────────────────────
# Arch hard-coded to x64; ai-gateway + Fly.io both run amd64 machines. If we
# ever need arm64 (local Apple Silicon build), switch URL to ...-linux-arm64.
RUN set -eux; \
    BLENDER_SHORT="$(echo ${BLENDER_VERSION} | cut -d. -f1,2)"; \
    URL="https://download.blender.org/release/Blender${BLENDER_SHORT}/blender-${BLENDER_VERSION}-linux-x64.tar.xz"; \
    echo "Downloading ${URL}"; \
    mkdir -p /opt/blender; \
    curl -fsSL "${URL}" | tar -xJf - -C /opt/blender --strip-components=1; \
    ln -sf /opt/blender/blender /usr/local/bin/blender; \
    blender --version

# ── Python deps (host Python for the post-process; Blender bundles its own) ─
RUN pip3 install --no-cache-dir \
      fastapi==0.115.5 \
      "uvicorn[standard]==0.32.1" \
      python-multipart==0.0.19 \
      pygltflib==1.16.4 \
      numpy==1.26.4

# ── App ─────────────────────────────────────────────────────────────────────
WORKDIR /app
COPY convert.sh /app/convert.sh
COPY server.py /app/server.py
COPY idle_watchdog.py /app/idle_watchdog.py
COPY start.sh /app/start.sh
RUN chmod +x /app/convert.sh /app/start.sh

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1

CMD ["/app/start.sh"]
