"""Modal deploy: Microsoft TRELLIS.2-4B — image-to-3D (MIT license).

State-of-the-art native 3D generation. 4B-param flow-matching transformer
over a sparse O-Voxel structure. Outputs textured mesh + PBR materials.

Per the upstream README: 512³ ~3s, 1024³ ~17s, 1536³ ~60s on H100. We
deploy on A100-40GB which is ~1.5–2× slower but $1.84/h vs H100 $4.04/h
and fits the model + activations comfortably.

Custom endpoint (no standard image-to-3D wire format yet):

  POST /v1/3d/generate
       multipart {image, resolution=512|1024|1536, format=glb|ply}
       returns the textured mesh as a binary stream.

Deploy:
  modal deploy modal/trellis2.py

Repo: https://github.com/microsoft/TRELLIS.2

NOTE: TRELLIS.2 is NOT pip-installable. The repo's code is imported
in-place from /opt/trellis2 via PYTHONPATH. We follow setup.sh
component-by-component (basic + flash-attn + nvdiffrast + o-voxel +
cumesh + flexgemm) since `pip install -e .` is unsupported upstream.
"""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent / "trellis2"

app = modal.App("canal-dark-trellis2")

# Build follows upstream setup.sh order: basic → flash-attn → nvdiffrast
# → o-voxel → cumesh → flexgemm. Each step is its own .run_commands so
# Modal can cache layers separately — flash-attn alone takes ~15min.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git", "build-essential", "ninja-build", "ffmpeg",
        "libgl1", "libglib2.0-0", "libegl1", "libxrender1",
        "libjpeg-dev",
        # nvdiffrast's pytorch extension build invokes `clang++` for the
        # final shared-object link. Without it the build fails at the
        # linker step even though g++ is present.
        "clang",
    )
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "FORCE_CUDA": "1",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0",  # A100, A10, L40S, H100
        "PYTHONPATH": "/opt/trellis2",
    })
    # 1) torch first — flash-attn imports torch at build time.
    .run_commands(
        "pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 "
        "--index-url https://download.pytorch.org/whl/cu124",
        "pip install --no-cache-dir packaging wheel ninja",
    )
    # 2) `--basic`: TRELLIS.2 runtime deps (no install of the repo itself).
    # transformers PIN: 4.56.2 — TRELLIS.2's image_feature_extractor.py
    # accesses `self.model.layer` directly on `DINOv3ViTModel`. The
    # Oct-2025 refactor (#40994) moved that ModuleList into a new
    # `DINOv3ViTEncoder` wrapper, breaking the trellis2 access path.
    # 4.56.2 is the last release with the original flat layout.
    .run_commands(
        "pip install --no-cache-dir imageio imageio-ffmpeg tqdm easydict "
        "opencv-python-headless trimesh transformers==4.56.2 tensorboard "
        "pandas lpips zstandard kornia timm rembg onnxruntime xatlas",
        "pip install --no-cache-dir "
        "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8",
        # FastAPI server stack — not in upstream setup.sh because the
        # repo ships only example.py / app.py (Gradio), not a REST API.
        "pip install --no-cache-dir 'fastapi>=0.115.0' "
        "'uvicorn[standard]>=0.32.0' python-multipart Pillow pydantic",
    )
    # 3) `--flash-attn` — pinned to upstream's 2.7.3.
    .run_commands(
        "pip install --no-cache-dir --no-build-isolation flash-attn==2.7.3",
    )
    # 4) Clone the repo. NOT installed — used via PYTHONPATH=/opt/trellis2.
    .run_commands(
        "git clone --recursive --depth 1 -b main "
        "https://github.com/microsoft/TRELLIS.2.git /opt/trellis2",
    )
    # 5) `--nvdiffrast` (mesh rasterization).
    .run_commands(
        "git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast",
        "pip install --no-cache-dir --no-build-isolation /tmp/nvdiffrast",
    )
    # 6) `--o-voxel` (sparse voxel ops — bundled in the repo).
    .run_commands(
        "pip install --no-cache-dir --no-build-isolation /opt/trellis2/o-voxel",
    )
    # 7) `--cumesh` (CUDA mesh extraction).
    .run_commands(
        "git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/CuMesh",
        "pip install --no-cache-dir --no-build-isolation /tmp/CuMesh",
    )
    # 8) `--flexgemm` (custom GEMM kernels for the 4B transformer).
    .run_commands(
        "git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/FlexGEMM",
        "pip install --no-cache-dir --no-build-isolation /tmp/FlexGEMM",
    )
    # 9) Pre-cache TRELLIS.2-4B + DINOv3 backbone (gated). HF token is
    # required for DINOv3; mounted from `huggingface-canal-dark` secret
    # at build time so the snapshot lands in the image.
    .run_commands(
        "python3 -c \""
        "import os; "
        "from huggingface_hub import snapshot_download; "
        "snapshot_download('microsoft/TRELLIS.2-4B', cache_dir='/models'); "
        "snapshot_download('facebook/dinov3-vitl16-pretrain-lvd1689m', "
        "  cache_dir='/models', token=os.environ['HF_TOKEN']); "
        "print('cached')\"",
        secrets=[modal.Secret.from_name("huggingface-canal-dark")],
    )
    .add_local_file(str(ROOT / "server.py"), "/app/server.py")
)


_func_kwargs = dict(
    image=image,
    # A100-40GB — verified by upstream, $1.84/h on Modal.
    # 24GB minimum per the README; L40S 48GB also works but pricier
    # ($1.95/h). H100 cuts inference 1.5× but $4.04/h doesn't pay back
    # at canal-dark's ~2-img-per-video volume.
    gpu="A100",
    timeout=1800,             # 1024³ texture pass can hit ~30-45s on A100
    scaledown_window=60,      # die 60s after last request
    min_containers=0,         # pay-per-use only
    # HF_TOKEN required at runtime: pipeline.from_pretrained loads the
    # gated DINOv3 backbone. Without it, lifespan startup raises 401.
    secrets=[modal.Secret.from_name("huggingface-canal-dark")],
)


@app.function(**_func_kwargs)
@modal.concurrent(max_inputs=1)
@modal.web_server(port=8000, startup_timeout=900)
def serve():
    import os
    import subprocess

    os.environ.setdefault("HF_HOME", "/models")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/models")
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTHONPATH", "/opt/trellis2")
    os.chdir("/app")
    subprocess.Popen(
        ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
    )
