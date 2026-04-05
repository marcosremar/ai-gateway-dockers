# cabecao-npm-dockers

Docker images for the PARLE project.

## parle-s2s

Speech-to-Speech inference server: Faster Whisper (STT) + LLM + MOSS-TTS-Realtime (TTS).

```bash
# Pull from GitHub Container Registry
docker pull marcosremar/parle-s2s:latest

# Run with GPU
docker run --gpus all -p 8000:8000 marcosremar/parle-s2s:latest

# Override models via env vars
docker run --gpus all -p 8000:8000 \
  -e LLM_MODEL=Qwen/Qwen2.5-3B-Instruct \
  -e STT_MODEL=large-v3 \
  marcosremar/parle-s2s:latest
```

Built automatically by GitHub Actions on push to `main`.

## ultravox-pipeline

Full speech-to-speech pipeline with Service Manager, based on the [ultravox-pipeline](https://github.com/marcosremar/ultravox-pipeline) project. Bakes a pre-packaged HuggingFace snapshot into the image for fast cold starts.

```bash
docker pull marcosremar/ultravox-pipeline:latest

docker run --gpus all -p 8888:8888 marcosremar/ultravox-pipeline:latest

# Override profile
docker run --gpus all -p 8888:8888 -e ULTRAVOX_PROFILE=gpu-dev marcosremar/ultravox-pipeline:latest
```

Built automatically by GitHub Actions on push to `main`.

## ultravox

Conversational speech-to-speech server with the same transport layer as babelcast (WebSocket, WebRTC, SSE). Uses Whisper (STT) + Mistral LLM (llama.cpp) + Kokoro TTS.

```bash
docker pull marcosremar/ultravox:latest

docker run --gpus all -p 8000:8000 marcosremar/ultravox:latest

# Use Qwen instead of Mistral
docker run --gpus all -p 8000:8000 -e CONF_LLM_MODEL=qwen marcosremar/ultravox:latest
```

Endpoints: `/health`, `/v1/speech`, `/v1/transcribe`, `/v1/tts`, `/api/stream-audio` (SSE), `/ws/stream` (WebSocket), `/api/offer` (WebRTC).

Built automatically by GitHub Actions on push to `main`.
