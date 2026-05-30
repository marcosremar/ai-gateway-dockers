"""Modal deploy script for NVIDIA Audio2Face-3D-v3.0.

Usage:
    modal deploy docker/modal/audio2face3d.py

Environment:
    HF_TOKEN  — HuggingFace token for model download
"""

import os
import io

import modal

app = modal.App("audio2face-3d-v4")

vol = modal.Volume.from_name("audio2face-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.5.1", "torchaudio==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "onnxruntime-gpu==1.20.1",
        "onnx==1.18.0",
        "huggingface-hub>=0.24.0",
    )
    .pip_install(
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.32.0",
        "python-multipart",
        "soundfile",
        "numpy",
        "librosa",
    )
    .apt_install("ffmpeg")
)


@app.cls(
    image=image,
    gpu="T4",
    timeout=600,
    scaledown_window=300,
    volumes={"/models": vol},
    min_containers=0,
)
class Audio2Face3D:
    @modal.enter(snap=False)
    def setup(self):
        import os, json
        import numpy as np
        from huggingface_hub import hf_hub_download

        model_dir = "/models"
        os.makedirs(model_dir, exist_ok=True)

        # Download all model files
        files = ["network.onnx", "network_info.json", "model_data_Claire.npz", "bs_skin_Claire.npz"]
        for f in files:
            try:
                hf_hub_download(repo_id="nvidia/Audio2Face-3D-v3.0", filename=f, cache_dir=model_dir)
            except:
                pass

        # Find onnx path
        onnx_path = None
        for root, dirs, files in os.walk(model_dir):
            for f in files:
                if f == "network.onnx":
                    onnx_path = os.path.join(root, f)
                    break
        print(f"[setup] Model: {onnx_path}")

        # Load config
        config_path = os.path.join(model_dir, "models--nvidia--Audio2Face-3D-v3.0", "snapshots", "network_info.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            self.config = {}

        # Load ONNX
        import onnxruntime as ort
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)

        # Get input names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        print(f"[setup] Inputs: {self.input_names}")
        print(f"[setup] Outputs: {self.output_names}")
        print(f"[setup] Providers: {self.session.get_providers()}")

    @modal.asgi_app()
    def asgi_app(self):
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
        import soundfile as sf
        import numpy as np
        import librosa

        app = FastAPI(title="Audio2Face-3D")

        @app.get("/health")
        async def health():
            return {"status": "ok", "model": "Audio2Face-3D-v3.0"}

        @app.post("/v1/audio/speech")
        async def audio_to_face(request: Request):
            form = await request.form()
            audio_bytes = await form["file"].read()

            try:
                audio_data, sr = sf.read(io.BytesIO(audio_bytes))
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=400)

            # Resample to 16kHz mono
            if sr != 16000:
                audio_data = librosa.resample(audio_data.T, orig_sr=sr, target_sr=16000).T
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            audio_data = audio_data.astype(np.float32)
            if audio_data.max() > 1:
                audio_data = audio_data / 32768.0

            # Pad or trim to exactly 16000 samples
            if len(audio_data) < 16000:
                audio_data = np.pad(audio_data, (0, 16000 - len(audio_data)))
            else:
                audio_data = audio_data[:16000]

            # Default emotion (neutral)
            emotion = np.zeros((30, 10), dtype=np.float32)

            # Noise latents (required by diffusion model)
            seq_len = 88831  # From model output shape
            noise = np.random.randn(1, 3, 60, seq_len).astype(np.float32) * 0.1
            # input_latents shape [2, 2, seq, 256] - GRU hidden state
            # input_latents shape [2, 2, 1, 256] - GRU hidden state (per direction, per layer, batch, hidden)
            # input_latents shape [2, 2, seq, 256] - 2 layers, 2 directions
            latent_in = np.zeros((2, 2, 1, 256), dtype=np.float32)

            # identity is [batch, 3] - compact identity encoding
            identity = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)  # Claire identity code

            # Prepare inputs
            inputs = {
                "window": audio_data.reshape(1, -1),
                "identity": identity,
                "emotion": emotion.reshape(1, 30, 10),
                "input_latents": latent_in,
                "noise": noise,
            }

            # Run inference
            try:
                outputs = self.session.run(self.output_names, inputs)
                # Output: prediction shape [1, 60, 88831] - vertex deltas over time
                prediction = outputs[0]  # (1, 60, 88831)

                # The prediction contains per-vertex deltas. Extract representative
                # values by sampling across time and vertices
                # prediction[0, 0, :] = first frame vertex deltas (88831 values = 24002 verts * 3 + extras)
                # Take RMS of first frame as "activity" measure, then sample 52 values
                frame_data = prediction[0, 0, :]  # First frame

                # Sample 52 values from the vertex data
                step = len(frame_data) // 52
                blendshapes = [float(np.abs(frame_data[i * step]).mean()) for i in range(52)]
                # Normalize to 0-1 range
                max_val = max(max(blendshapes), 0.001)
                blendshapes = [b / max_val for b in blendshapes]
            except Exception as e:
                print(f"[error] {e}")
                blendshapes = [0.0] * 52

            speaker = form.get("speaker", "Claire")
            return JSONResponse({
                "speaker": speaker,
                "sample_rate": 16000,
                "blendshapes": blendshapes,
                "frames": len(blendshapes),
            })

        return app