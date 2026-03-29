# Meeting Bot - Docker Image for Screen Recording
# Build: 2026-03-16 — native screen share (getDisplayMedia) instead of CDP screenshots
FROM ubuntu:24.04

# Install Node.js 20.x
RUN apt-get update && apt-get install -y curl ca-certificates gnupg
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
RUN apt-get install -y nodejs

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Core browser dependencies
    wget libnss3 libatk-bridge2.0-0 libdrm2 libxkbcommon0 \
    libxcomposite1 libxdamage1 libxrandr2 libgbm1 libxss1 libxshmfence1 \
    # Virtual display and audio
    xvfb x11vnc x11-utils pulseaudio pulseaudio-utils libasound2-plugins unclutter \
    # Media processing
    ffmpeg \
    # Virtual camera (v4l2loopback built from source at runtime to match host kernel)
    v4l-utils kmod dkms build-essential \
    # Utilities
    curl unzip \
    && rm -rf /var/lib/apt/lists/*

# Pre-download v4l2loopback source for runtime compilation
RUN apt-get update && apt-get install -y git && \
    git clone --depth 1 https://github.com/umlaeute/v4l2loopback.git /opt/v4l2loopback && \
    rm -rf /opt/v4l2loopback/.git && \
    rm -rf /var/lib/apt/lists/*

# Install AWS CLI v2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip && ./aws/install && rm -rf awscliv2.zip aws

# Install Bun (for avatar server)
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/root/.bun/bin:${PATH}"

# Application setup
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# Install Playwright's Chromium + create symlink for browser.ts compatibility
RUN npx playwright install chromium && \
    find /root/.cache/ms-playwright -name chrome -type f -executable | head -1 | xargs -I {} ln -sf {} /usr/bin/google-chrome

# Install teams-for-linux (Electron wrapper for full Teams video support in avatar mode)
# Detect architecture and install the right package
RUN ARCH=$(dpkg --print-architecture) && \
    curl -sL "https://github.com/IsmaelMartinez/teams-for-linux/releases/download/v2.7.10/teams-for-linux_2.7.10_${ARCH}.deb" -o /tmp/teams.deb && \
    apt-get update -qq && apt-get install -y -qq libgtk-3-0 libnotify4 libxtst6 xdg-utils libatspi2.0-0 libsecret-1-0 2>/dev/null && \
    dpkg -i /tmp/teams.deb || apt-get install -f -y -qq && \
    rm -f /tmp/teams.deb && rm -rf /var/lib/apt/lists/*

# Build application
COPY . .
RUN npm run build

# Copy avatar assets (TalkingHead modules + GLB models)
COPY avatar/ /app/avatar/

# Copy YouTube streaming script
COPY start_youtube_stream.sh /app/start_youtube_stream.sh
RUN chmod +x /app/start_youtube_stream.sh

# Environment configuration
ENV NODE_OPTIONS="--max-old-space-size=2048"
ENV SERVERLESS=true
ENV NODE_ENV=production
ENV DISPLAY=:99
ENV PULSE_RUNTIME_PATH=/tmp/pulse
ENV XDG_RUNTIME_DIR=/tmp/pulse

# Copy startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Expose HTTP server (health check + API), VNC (debug), RTMP (webcam PiP), Avatar HTTP
EXPOSE 8080 5900 1936 3099

ENTRYPOINT ["/start.sh"]
