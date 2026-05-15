import io, os, base64, json
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch

app = FastAPI()
predictor = None

def load_model():
    global predictor
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM2-large on {device}...")

    model = build_sam2(
        "sam2.1_hiera_large",
        "/checkpoints/sam2.1_hiera_large.pt",
        device=device,
    )
    predictor = SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=500,
    )
    print("SAM2-large ready.")

@app.on_event("startup")
def startup():
    load_model()

@app.get("/health")
def health():
    return {"status": "ok", "model": "sam2-large", "device": "cuda" if torch.cuda.is_available() else "cpu"}

@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    """Auto-segment image — returns list of cropped objects as base64 PNGs."""
    data = await file.read()
    img  = Image.open(io.BytesIO(data)).convert("RGB")
    arr  = np.array(img)

    masks = predictor.generate(arr)

    # Sort by area descending, keep top 20
    masks.sort(key=lambda m: m["area"], reverse=True)
    masks = masks[:20]

    results = []
    for i, mask in enumerate(masks):
        m = mask["segmentation"]
        # Bounding box
        rows = np.any(m, axis=1)
        cols = np.any(m, axis=0)
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]

        # Crop with RGBA (transparent background)
        crop = arr[r0:r1+1, c0:c1+1].copy()
        alpha = (m[r0:r1+1, c0:c1+1] * 255).astype(np.uint8)
        rgba  = np.dstack([crop, alpha])
        pil   = Image.fromarray(rgba, "RGBA")

        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        results.append({
            "id":    i,
            "area":  int(mask["area"]),
            "score": float(mask["predicted_iou"]),
            "bbox":  [int(c0), int(r0), int(c1-c0), int(r1-r0)],
            "png_b64": b64,
        })

    return JSONResponse({"count": len(results), "objects": results})

@app.post("/segment/save")
async def segment_and_save(file: UploadFile = File(...), out_dir: str = "/tmp/segments"):
    """Segment and save PNGs to disk — returns file paths."""
    os.makedirs(out_dir, exist_ok=True)
    resp = await segment(file)
    body = json.loads(resp.body)

    paths = []
    for obj in body["objects"]:
        path = f"{out_dir}/obj_{obj['id']:02d}_score{obj['score']:.2f}.png"
        img_data = base64.b64decode(obj["png_b64"])
        with open(path, "wb") as f:
            f.write(img_data)
        paths.append(path)
        del obj["png_b64"]  # remove from response

    return {"count": len(paths), "paths": paths, "objects": body["objects"]}
