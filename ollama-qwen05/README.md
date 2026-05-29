# ollama-qwen05

Ollama with **qwen2.5:0.5b pre-baked** into the image, tuned for fast
time-to-first-inference on the Vast.ai marketplace (no CRIU available there).

The model ships inside an image layer, so there is **no runtime `ollama pull`** —
removing the ~18s that dominated cold start in benchmarks. `OLLAMA_KEEP_ALIVE=-1`
pins the model in VRAM after the first load.

```bash
docker pull ghcr.io/marcosremar/ollama-qwen05:latest

docker run --gpus all -p 11434:11434 ghcr.io/marcosremar/ollama-qwen05:latest

# Inference
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:0.5b",
  "prompt": "Say \"ok\" in one word.",
  "stream": false
}'
```

Bake a different model at build time with `--build-arg MODEL=...`.

Built automatically by GitHub Actions on push to `main`.
