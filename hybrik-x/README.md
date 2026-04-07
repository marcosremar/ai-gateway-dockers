# hybrik-x

GPU inference server for [HybrIK-X](https://github.com/jeffffffli/HybrIK) — whole-body SMPL-X mesh recovery from a single image.

Returns **55 joint rotations** (quaternions + rotation matrices), shape betas, expression params, 3D joint positions, and optionally the full 10475-vertex mesh.

## Build

```bash
docker build -t hybrik-x .
```

> **Note:** The build downloads ~2GB of model weights from Google Drive via `gdown`. If you hit rate limits, download manually and use `COPY`:
> ```dockerfile
> COPY model_files/ /app/HybrIK/model_files/
> COPY pretrained_models/ /app/HybrIK/pretrained_models/
> ```

## Run

```bash
docker run --gpus all -p 8000:8000 hybrik-x
```

## API

### `GET /health`

```json
{ "status": "ok", "device": "cuda" }
```

### `POST /predict`

```bash
# Encode image
IMG_B64=$(base64 -i photo.jpg)

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"$IMG_B64\"}"
```

**Request:**

| Field | Type | Description |
|-------|------|-------------|
| `image_base64` | string | Base64-encoded image (JPEG/PNG) |
| `bbox` | `[x1,y1,x2,y2]` (optional) | Skip person detection |
| `include_vertices` | bool (default: false) | Include full SMPL-X mesh (adds ~125KB) |
| `flip_test` | bool (default: true) | Average normal + flipped prediction (2x slower, more accurate) |

**Response:**

| Field | Shape | Description |
|-------|-------|-------------|
| `theta_quat` | `[55][4]` | Joint rotations as quaternions [w,x,y,z] |
| `rot_mats` | `[55][3][3]` | Joint rotation matrices |
| `betas` | `[11]` | SMPL-X shape (10 betas + 1 kid) |
| `expression` | `[10]` | SMPL-X expression params |
| `transl` | `[3]` | Global translation (meters) |
| `cam_scale` | `[1]` | Camera scale |
| `joints_3d` | `[71][3]` | 3D joints, root-relative (meters) |
| `bbox` | `[4]` | Bounding box used |
| `vertices` | `[10475][3]` | SMPL-X mesh (if requested) |

**55 joints:** 22 body + 1 jaw + 2 eyes + 15 left hand + 15 right hand.

## Deploy on vast.ai Serverless

1. Uncomment the `vastai-sdk` and `worker.py` lines in the Dockerfile
2. Change `CMD` to run both `start.sh` (background) and `worker.py`
3. Push to Docker Hub: `docker push marcosremar/hybrik-x:latest`
4. Create a vast.ai Endpoint pointing to the image

## GPU Requirements

~3-4 GB VRAM. Works on T4 (16GB), RTX 3060 (12GB), or any CUDA 11.8+ GPU.
