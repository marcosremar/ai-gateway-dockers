FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    git wget curl ffmpeg libgl1-mesa-glx libglib2.0-0 unzip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# PyTorch 2.0.1 + CUDA 11.8
RUN pip install --no-cache-dir \
    torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Clone HybrIK
WORKDIR /app
RUN git clone https://github.com/jeffffffli/HybrIK.git /app/HybrIK
WORKDIR /app/HybrIK

# Python deps
RUN pip install --no-cache-dir \
    numpy scipy opencv-python-headless tqdm easydict pyyaml \
    cython pycocotools matplotlib six terminaltables chumpy \
    joblib fastapi uvicorn python-multipart Pillow gdown

# Install HybrIK without setup.py (it pins opencv-python==4.1.2.30 which doesn't exist for py3.10)
# We already installed opencv-python-headless above, so just add HybrIK to the path
RUN pip install --no-cache-dir --no-deps -e .

# Download model files (SMPL-X body model + kid template)
RUN gdown 1un9yAGlGjDooPwlnwFpJrbGHRiLaBNzV -O /tmp/model_files.zip && \
    unzip /tmp/model_files.zip -d /app/HybrIK/ && \
    rm /tmp/model_files.zip

# Download HybrIK-X pretrained checkpoint (HRNet-W48 + RLE, SMPL-X)
RUN mkdir -p /app/HybrIK/pretrained_models && \
    gdown 1R0WbySXs_vceygKg_oWeLMNAZCEoCadG \
    -O /app/HybrIK/pretrained_models/hybrikx_rle_hrnet.pth

# Copy server code
COPY server.py /app/server.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Vast.ai PyWorker (optional — only needed for vast.ai Serverless)
# RUN pip install --no-cache-dir vastai-sdk
# COPY worker.py /app/worker.py

EXPOSE 8000

CMD ["/app/start.sh"]
