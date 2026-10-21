# ── Manim Community — Open-Source Animated Math/Diagram Engine (ai-gateway) ──
# Wraps Manim CE (MIT) so the canal-dark / project-philosofi pipeline can render
# blueprint-style technical diagrams (Pattern 4 Blueprint Stoic) and whiteboard-
# style hand-drawn animations (Pattern 9) without paid tools (Midjourney/VideoScribe).
#
# Engine:  manim CE (https://github.com/ManimCommunity/manim) — Python, CPU-only
# License: MIT
# CPU:     no GPU needed — runs on cheap Vast.ai CPU instances or locally
#
# Build:
#   docker build -t marcosremar/manim:latest .
# Run:
#   docker run -p 8000:8000 -p 22:22 \
#     -e IDLE_TIMEOUT_MIN=15 \
#     marcosremar/manim:latest
# ──────────────────────────────────────────────────────────────────────────────

ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    IDLE_TIMEOUT_MIN=15 \
    MANIM_RESULTS_DIR=/app/results

# System deps required by Manim (cairo/pango for text/SVG, ffmpeg for mp4 muxing,
# texlive only on demand — skipped here to keep image small; LaTeX features
# disabled by default. Add `texlive-full` if Mathy LaTeX rendering needed.)
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
        ffmpeg \
        libcairo2-dev libpango1.0-dev pkg-config python3-dev \
        build-essential \
        curl ca-certificates \
        openssh-server \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd \
    && sed -i 's/#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config \
    && sed -i 's/#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config

WORKDIR /app

RUN pip install --upgrade pip \
    && pip install \
         "manim>=0.18,<0.20" \
         "fastapi>=0.115" "uvicorn[standard]>=0.32" \
         "pydantic>=2.5" \
         httpx

COPY server.py            /app/server.py
COPY idle_watchdog.py     /app/idle_watchdog.py
COPY start.sh             /app/start.sh
RUN chmod +x /app/start.sh

RUN mkdir -p /app/results /app/scenes

EXPOSE 8000 22

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/app/start.sh"]
