# ── ollama-qwen05 — fast cold-start Ollama image (qwen2.5:0.5b pre-baked) ─────
# Optimized for time-to-first-inference on the Vast.ai marketplace, where CRIU /
# snapshot restore is unavailable (containers lack CAP_SYS_ADMIN, vm:true is
# stripped). The win comes from removing the runtime `ollama pull`:
#
#   - Model baked into an image layer → no network pull on boot (~-18s).
#   - Built FROM ollama/ollama:latest, a base whose layers Vast hosts commonly
#     cache → only the small model layer downloads on a warm host.
#   - OLLAMA_KEEP_ALIVE=-1 pins the model in VRAM after first load (no reload).
#
# On a host with the base cached, deploy → first inference collapses to
# allocation + container start + VRAM load (~1-2s) instead of +pull+load.
FROM ollama/ollama:latest

ARG MODEL=qwen2.5:0.5b
ENV MODEL=${MODEL} \
    OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_MODELS=/root/.ollama/models \
    OLLAMA_KEEP_ALIVE=-1

# Pre-bake the model into this layer: start the server, wait until it answers,
# pull, then stop. The pulled blobs persist under OLLAMA_MODELS in the layer.
RUN ollama serve & SERVE_PID=$! && \
    for i in $(seq 1 60); do ollama list >/dev/null 2>&1 && break; sleep 1; done && \
    ollama pull "${MODEL}" && \
    ollama list && \
    kill "${SERVE_PID}"

EXPOSE 11434

# ollama/ollama's ENTRYPOINT is ["/bin/ollama"]; run the server. Clients poll
# /api/tags for readiness — no fixed sleep needed.
HEALTHCHECK --interval=5s --timeout=3s --start-period=5s --retries=12 \
  CMD ollama list >/dev/null 2>&1 || exit 1

CMD ["serve"]
