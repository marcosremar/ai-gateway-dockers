"""Modal deploy script for MOSS-TTS standalone server.

Deploys OpenMOSS-Team/MOSS-TTS on a Modal serverless GPU.
Uses transformers AutoModel/AutoProcessor API with a simple /api/text endpoint.

Usage:
    modal deploy docker/modal/moss_tts.py

Endpoints:
    GET  /health       — health check
    POST /api/text     — simple TTS: { text, language?, temperature?, top_p?, top_k?, reference_audio? }
                         returns: { audio (base64 WAV), sample_rate, duration_seconds, generation_time }
"""

import modal

app = modal.App("babelcast-moss-tts")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .apt_install("ffmpeg", "libsndfile1", "git", "curl")
    .run_commands(
        # Upgrade pip first (old pip can't parse PEP 508 markers in MOSS-TTS)
        "pip install --upgrade pip setuptools wheel",
        # Install PyTorch + torchaudio for CUDA 12.8
        "pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu128",
        # Clone and install MOSS-TTS
        "git clone --depth 1 https://github.com/OpenMOSS/MOSS-TTS.git /opt/moss-tts",
        "cd /opt/moss-tts && pip install --no-cache-dir -e .",
        # Install FastAPI server deps
        "pip install --no-cache-dir 'fastapi>=0.115.0' 'uvicorn[standard]>=0.32.0' python-multipart soundfile numpy hf_transfer",
        # Install transformers for AutoModel/AutoProcessor
        "pip install --no-cache-dir transformers accelerate",
        # Pre-download model weights
        "python3 -c 'from huggingface_hub import snapshot_download; snapshot_download(\"OpenMOSS-Team/MOSS-TTS\")'",
        "echo 'moss-tts-v1'",
    )
)

# Wrapper server script using transformers AutoModel/AutoProcessor API
wrapper_server = """
import base64, io, os, time, logging

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

log = logging.getLogger("moss-tts-wrapper")
logging.basicConfig(level=logging.INFO)

# Disable certain CUDA backends per MOSS-TTS recommendation
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

_model = None
_processor = None
_device = None


def get_model():
    global _model, _processor, _device
    if _model is None:
        log.info("Loading MOSS-TTS model...")
        t0 = time.time()
        from transformers import AutoModel, AutoProcessor

        pretrained = "OpenMOSS-Team/MOSS-TTS"
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if _device == "cuda" else torch.float32

        _processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=True)
        _model = AutoModel.from_pretrained(
            pretrained, trust_remote_code=True, torch_dtype=dtype
        ).to(_device)
        _model.eval()
        log.info(f"Model loaded in {time.time()-t0:.1f}s on {_device}")
    return _model, _processor, _device


app = FastAPI(title="MOSS-TTS")


@app.get("/health")
def health():
    return {"status": "ok", "service": "moss-tts", "model": "OpenMOSS-Team/MOSS-TTS"}


@app.post("/api/text")
async def api_text(request: Request):
    try:
        body = await request.json()
        text = body.get("text", "")
        if not text:
            return JSONResponse(status_code=400, content={"error": "text is required"})

        model, processor, device = get_model()

        t0 = time.time()

        # Optional reference audio for voice cloning
        ref_audio_path = None
        ref_audio_b64 = body.get("reference_audio")
        if ref_audio_b64:
            import tempfile
            audio_bytes = base64.b64decode(ref_audio_b64)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(audio_bytes)
            tmp.close()
            ref_audio_path = tmp.name

        # Build message
        if ref_audio_path:
            message = processor.build_user_message(text=text, reference=[ref_audio_path])
        else:
            message = processor.build_user_message(text=text)

        # Process and generate
        batch = processor([message], mode="generation")
        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=4096,
            )

        # Decode output to audio
        decoded = list(processor.decode(outputs))
        if not decoded or not decoded[0].audio_codes_list:
            return JSONResponse(status_code=500, content={"error": "No audio generated"})

        audio = decoded[0].audio_codes_list[0]  # shape: (samples,)
        sample_rate = processor.model_config.sampling_rate

        gen_time = time.time() - t0

        # Encode as WAV base64 (use soundfile, torchaudio 2.10 needs torchcodec)
        audio_np = audio.cpu().float().numpy()
        buf = io.BytesIO()
        sf.write(buf, audio_np, sample_rate, format="WAV", subtype="PCM_16")
        buf.seek(0)
        audio_b64 = base64.b64encode(buf.read()).decode()

        # Cleanup temp file
        if ref_audio_path:
            os.unlink(ref_audio_path)

        return {
            "audio": audio_b64,
            "sample_rate": sample_rate,
            "duration_seconds": round(audio.shape[0] / sample_rate, 2),
            "generation_time": round(gen_time, 3),
        }

    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={
            "error": str(e), "traceback": traceback.format_exc()
        })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""


@app.function(
    image=image,
    gpu="L40S",
    timeout=600,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=1)  # batch_size=1 only
@modal.web_server(port=8000, startup_timeout=300)
def serve():
    import subprocess, tempfile, os
    # Write wrapper server to a temp file and run it
    server_path = "/tmp/moss_tts_server.py"
    with open(server_path, "w") as f:
        f.write(wrapper_server)
    os.chdir("/opt/moss-tts")
    subprocess.Popen(["python3", server_path])
