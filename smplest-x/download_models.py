"""Download SMPL-X/SMPL/MANO body models + SMPLest-X checkpoint at Docker build time."""
import shutil
import os
from huggingface_hub import hf_hub_download

SMPLESTX_ROOT = "/app/SMPLest-X"
os.chdir(SMPLESTX_ROOT)

# ── SMPL-X body models (.npz) ───────────────────────────────────────────────
os.makedirs("human_models/human_model_files/smplx", exist_ok=True)
for f in ["SMPLX_NEUTRAL.npz", "SMPLX_MALE.npz", "SMPLX_FEMALE.npz"]:
    path = hf_hub_download("lithiumice/models_hub", f"4_SMPLhub/SMPLX/X_npz/{f}")
    shutil.copy(path, f"human_models/human_model_files/smplx/{f}")
    size_mb = os.path.getsize(f"human_models/human_model_files/smplx/{f}") / 1e6
    print(f"  OK: smplx/{f} ({size_mb:.1f} MB)")

# ── SMPL-X joint mapping + vertex IDs (from camenduru/SMPLer-X) ────────────
# These are required by the SMPLX singleton for joint mapping and face vertices.
for f in ["SMPLX_to_J14.pkl", "MANO_SMPLX_vertex_ids.pkl",
          "SMPL-X__FLAME_vertex_ids.npy", "SMPLX_NEUTRAL.pkl"]:
    path = hf_hub_download("camenduru/SMPLer-X", f)
    shutil.copy(path, f"human_models/human_model_files/smplx/{f}")
    size_mb = os.path.getsize(f"human_models/human_model_files/smplx/{f}") / 1e6
    print(f"  OK: smplx/{f} ({size_mb:.1f} MB)")

# ── SMPL body models (.pkl) ─────────────────────────────────────────────────
os.makedirs("human_models/human_model_files/smpl", exist_ok=True)
for f in ["SMPL_NEUTRAL.pkl", "SMPL_MALE.pkl", "SMPL_FEMALE.pkl"]:
    path = hf_hub_download("lithiumice/models_hub", f"4_SMPLhub/SMPL/X_pkl/{f}")
    shutil.copy(path, f"human_models/human_model_files/smpl/{f}")
    size_mb = os.path.getsize(f"human_models/human_model_files/smpl/{f}") / 1e6
    print(f"  OK: smpl/{f} ({size_mb:.1f} MB)")

# ── MANO hand models (.pkl) ─────────────────────────────────────────────────
os.makedirs("human_models/human_model_files/mano", exist_ok=True)
for f in ["MANO_LEFT.pkl", "MANO_RIGHT.pkl"]:
    path = hf_hub_download("lithiumice/models_hub", f"4_SMPLhub/MANO/pkl/{f}")
    shutil.copy(path, f"human_models/human_model_files/mano/{f}")
    size_mb = os.path.getsize(f"human_models/human_model_files/mano/{f}") / 1e6
    print(f"  OK: mano/{f} ({size_mb:.1f} MB)")

# ── SMPLest-X checkpoint + config (8.2GB) ───────────────────────────────────
os.makedirs("pretrained_models/smplest_x_h", exist_ok=True)
for f in ["smplest_x_h.pth.tar", "config_base.py"]:
    path = hf_hub_download("waanqii/SMPLest-X", f)
    shutil.copy(path, f"pretrained_models/smplest_x_h/{f}")
    size_mb = os.path.getsize(f"pretrained_models/smplest_x_h/{f}") / 1e6
    print(f"  OK: {f} ({size_mb:.1f} MB)")

# ── YOLOv8x detector ────────────────────────────────────────────────────────
try:
    # ultralytics auto-downloads yolov8x.pt on first use
    os.makedirs("pretrained_models", exist_ok=True)
    from ultralytics import YOLO
    model = YOLO("yolov8x.pt")
    if os.path.exists("yolov8x.pt"):
        shutil.move("yolov8x.pt", "pretrained_models/yolov8x.pt")
    print("  OK: yolov8x.pt")
except Exception as e:
    print(f"  SKIP: YOLOv8x ({e}) — will auto-download at runtime")

print("\nAll models downloaded!")
