# Unreal Engine remote desktop image.
# Base: gezp/ubuntu-desktop (Ubuntu 22.04 + CUDA 12.2 + XFCE + NoMachine/noVNC + GPU passthrough).
# Adds Unreal Engine build/runtime deps, Vulkan, SSH for ai-gateway dev mode, health endpoint.

FROM gezp/ubuntu-desktop:22.04-cu12.2.2

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential clang lld cmake ninja-build pkg-config \
      git curl wget ca-certificates unzip zip \
      libssl-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
      libxkbcommon-dev libwayland-dev libudev-dev \
      libsdl2-dev libfreetype-dev libpng-dev libjpeg-dev \
      libvulkan1 vulkan-tools mesa-vulkan-drivers \
      libgl1 libgl1-mesa-dri libglu1-mesa libglx-mesa0 \
      libasound2 libpulse0 \
      openssh-server tmux htop nano \
      python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/run/sshd /root/.ssh \
    && chmod 700 /root/.ssh \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config

WORKDIR /app
COPY health_server.py /app/health_server.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 22 4000 5000 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/app/start.sh"]
