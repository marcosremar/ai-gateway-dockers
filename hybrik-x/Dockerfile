FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    git wget curl ffmpeg libgl1-mesa-glx libglib2.0-0 unzip \
    build-essential g++ \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# PyTorch 2.1 + CUDA 12.1 — shares base layers with babelcast on Vast.ai hosts
RUN pip install --no-cache-dir \
    torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Clone HybrIK
RUN git clone https://github.com/jeffffffli/HybrIK.git /app/HybrIK
WORKDIR /app/HybrIK

# Python deps
RUN pip install --no-cache-dir \
    "numpy<2" scipy opencv-python-headless tqdm easydict pyyaml \
    cython pycocotools matplotlib six terminaltables chumpy \
    joblib fastapi uvicorn python-multipart Pillow gdown

# Install HybrIK without setup.py deps (pins opencv==4.1.2 which doesn't exist for py3.10)
RUN pip install --no-cache-dir --no-deps -e .

# Download model files (SMPL-X body model + kid template)
RUN gdown 1un9yAGlGjDooPwlnwFpJrbGHRiLaBNzV -O /tmp/model_files.zip && \
    unzip /tmp/model_files.zip -d /app/HybrIK/ && \
    rm /tmp/model_files.zip

# Download HybrIK-X pretrained checkpoint (HRNet-W48 + RLE, SMPL-X)
RUN mkdir -p /app/HybrIK/pretrained_models && \
    gdown 1R0WbySXs_vceygKg_oWeLMNAZCEoCadG \
    -O /app/HybrIK/pretrained_models/hybrikx_rle_hrnet.pth

# Pre-bake FasterRCNN weights so startup doesn't need internet (~167MB from pytorch.org)
RUN python3 -c "from torchvision.models.detection import fasterrcnn_resnet50_fpn; fasterrcnn_resnet50_fpn(pretrained=True); print('FasterRCNN weights cached')"

# Copy server code
COPY server.py /app/server.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8000

CMD ["/app/start.sh"]
