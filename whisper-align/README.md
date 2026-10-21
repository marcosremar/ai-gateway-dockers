# whisper-align — Word-level forced alignment (ai-gateway)

Wrapper of [WhisperX](https://github.com/m-bain/whisperX) (BSD-4) that
exposes a single multipart endpoint for word-level audio alignment.
Used by the canal-dark / project-philosofi pipeline to convert
narration `voice.mp3` into `timing.json` so Remotion can sync scene
cuts to spoken sentences instead of relying on hand-tuned offsets in
`script.json`.

Pipeline:

```
script.json (scenes[]) → qwen3-tts → voice.mp3
                                       ↓
                               whisper-align (this image)
                                       ↓
       timing.json {scenes: [{start_s, end_s, text}, ...]}
                                       ↓
                                   Remotion
```

Conventions match the rest of `ai-gateway-dockers/`:

- `GET /health` for the autoscaler readiness probe.
- `idle_watchdog.py` (auto-shutdown after `IDLE_TIMEOUT_MIN` minutes idle).
- HF cache persisted to `/workspace/.cache/huggingface` when a workspace
  volume is mounted.
- sshd (PUBKEY auth) for SSH-patch iteration.

## Build

```bash
cd ai-gateway-dockers/whisper-align
docker build -t marcosremar/whisper-align:latest .
```

## Run

GPU recommended (~6 GB VRAM for `large-v3`); CPU works too — set
`WHISPERX_MODEL=base` for snappier CPU inference.

```bash
docker run --gpus all -p 8000:8000 -p 22:22 \
  -v $PWD/_cache/hf:/workspace/.cache/huggingface \
  -e IDLE_TIMEOUT_MIN=15 \
  -e WHISPERX_MODEL=large-v3 \
  marcosremar/whisper-align:latest
```

## Smoke test

Forced alignment with the script's wording (recommended for canal-dark
since the audio was synthesized FROM `script.json`):

```bash
curl -X POST http://localhost:8000/v1/audio/align \
  -F audio=@voice.mp3 \
  -F text="Marcus Aurelius wrote in his Meditations..." \
  -F language=en \
  -o timing.json
```

Free transcription + alignment (no target text):

```bash
curl -X POST http://localhost:8000/v1/audio/align \
  -F audio=@voice.mp3 \
  -o timing.json
```

Response:

```json
{
  "language": "en",
  "duration": 245.6,
  "segments": [
    {"start": 0.00, "end": 4.20, "text": "Marcus Aurelius wrote..."}
  ],
  "words": [
    {"word": "Marcus", "start": 0.12, "end": 0.45, "score": 0.92}
  ]
}
```

## Deploy via ai-gateway

```bash
ai-gateway gpu deploy \
  --image marcosremar/whisper-align:latest \
  --gpu-types "NVIDIA RTX 4090,NVIDIA L40S" \
  --label canal-dark/whisper-align \
  --storage-gb 30 \
  --readiness-probe ssh \
  --allow-unverified
```

## Notes

- We do **not** use pyannote's `DiarizationPipeline`, so no `HF_TOKEN`
  is required for default operation. (The `pyannote.audio` dependency
  is still pulled in because WhisperX imports it for VAD scaffolding.)
- WhisperX downloads two checkpoints per language on first use:
  the whisper ASR weights (~3 GB for `large-v3`) and the wav2vec2
  alignment weights (~1 GB). Both are cached under `HF_HOME`.
