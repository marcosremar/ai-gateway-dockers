# ── Storyboard-AI Whiteboard Animation Engine — ai-gateway Docker ───────────
# Wraps the OpenCV object-by-object hand-drawn whiteboard engine from
# https://github.com/Innovate-Inspire/storyboard-ai (GPL-3.0) so the
# canal-dark / project-philosofi pipeline can render Pattern 9 (whiteboard)
# videos with a *real* hand sprite traveling along contours instead of the
# old SVG-turbulence WhiteboardDraw.tsx hack.
#
# Engine:  custom OpenCV pipeline (CLAHE + adaptive threshold + grid scan
#          + per-object segmentation + hand sprite compositor)
# License: GPL-3.0 (storyboard-ai); Apache-2.0 for this Dockerfile/server
# CPU:     no GPU needed — runs on cheap Vast.ai CPU instances or locally
#
# Build:
#   docker build -t marcosremar/storyboard-whiteboard:latest .
# Run:
#   docker run -p 8000:8000 -p 22:22 \
#     -e IDLE_TIMEOUT_MIN=15 \
#     marcosremar/storyboard-whiteboard:latest
# ─────────────────────────────────────────────────────────────────────────────

ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    IDLE_TIMEOUT_MIN=15 \
    WB_RESULTS_DIR=/app/results \
    WB_JOBS_DIR=/app/jobs \
    WB_BUNDLED_HAND=/app/assets/drawing-hand.png \
    WB_BUNDLED_HAND_MASK=/app/assets/hand-mask.png

# System deps:
#   ffmpeg     — final mp4 muxing (cv2.VideoWriter uses mp4v fourcc)
#   libsm6/libxext6/libgl1 — required by opencv-python-headless on slim
#   openssh-server — gateway dev-mode SSH iteration
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 libxext6 libgl1 \
        curl ca-certificates \
        openssh-server \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd \
    && sed -i 's/#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config \
    && sed -i 's/#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config

WORKDIR /app

RUN pip install --upgrade pip \
    && pip install \
         "opencv-python-headless>=4.9,<5" \
         "numpy>=1.26,<3" \
         "Pillow>=10" \
         "fastapi>=0.115" "uvicorn[standard]>=0.32" \
         "python-multipart>=0.0.9" \
         httpx

COPY engine.py            /app/engine.py
COPY server.py            /app/server.py
COPY idle_watchdog.py     /app/idle_watchdog.py
COPY start.sh             /app/start.sh
COPY assets/              /app/assets/
RUN chmod +x /app/start.sh

RUN mkdir -p /app/results /app/jobs

EXPOSE 8000 22

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/app/start.sh"]
