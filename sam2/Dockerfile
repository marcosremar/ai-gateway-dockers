FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip python3.11-venv \
    git wget curl libgl1 libglib2.0-0 openssh-server \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

RUN pip install --upgrade pip
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# SAM2
RUN pip install git+https://github.com/facebookresearch/sam2.git

# API server
RUN pip install fastapi uvicorn python-multipart pillow numpy

# Download SAM2-large weights
RUN mkdir -p /checkpoints && \
    wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
    -O /checkpoints/sam2.1_hiera_large.pt

COPY server.py /app/server.py
WORKDIR /app

EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
