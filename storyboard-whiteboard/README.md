# storyboard-whiteboard — legit hand-drawn whiteboard animation (open-source)

Wraps the [storyboard-ai](https://github.com/Innovate-Inspire/storyboard-ai) OpenCV whiteboard animation engine (GPL-3.0) behind ai-gateway. Replaces the old SVG-turbulence-based `WhiteboardDraw.tsx` hack with the real object-by-object pixel-fill + hand sprite traveling along contours pipeline.

## Why this Docker exists

`canal-dark/remotion/src/styles/WhiteboardDraw.tsx` rendered hand-drawn animation via:

- SVG `<path>` with `stroke-dashoffset` interpolation
- `feTurbulence` + `feDisplacementMap` filters for "wobble"
- A static PNG hand sprite snapped to `getPointAtLength(drawn)`

That worked, kinda, but the result felt *generated* — uniform stroke speed, no per-object segmentation, no grid-scan order, the hand never lifted between strokes. The storyboard-ai engine is built around how a real artist actually fills a whiteboard:

1. CLAHE + adaptive threshold → binarised line mask
2. Split image into NxN grid cells
3. For each labelled object (LabelMe JSON masks): nearest-neighbour walk through black grid cells, drop pixels in cell-by-cell with the hand sprite hovering at each cell
4. Background pass last with bigger split + skip rate
5. Final 3s freeze on the original image

CPU only, ~4s for 1020×1020. No GPU. No subscriptions.

## Engine licence

The Python engine in `engine.py` is a near-verbatim copy of `draw-whiteboard-animations.py` from storyboard-ai (GPL-3.0). The `server.py`, `Dockerfile`, and `idle_watchdog.py` are Apache-2.0 (this repo). Distributing the combined Docker image carries the GPL-3.0 obligation if you ship binaries to third parties — fine for self-hosted ai-gateway use.

## Build

```bash
cd ai-gateway-dockers/storyboard-whiteboard
docker build -t marcosremar/storyboard-whiteboard:latest .
```

## Run (local CPU)

```bash
docker run -p 8000:8000 -p 22:22 \
  -e IDLE_TIMEOUT_MIN=15 \
  marcosremar/storyboard-whiteboard:latest
```

No GPU flag, no HuggingFace cache mount — engine has no model weights.

## Run (Vast.ai CPU via ai-gateway)

```bash
ai-gateway gpu deploy --image marcosremar/storyboard-whiteboard:latest --cpu-only
# returns: { url: https://xxx.proxy.vast.ai, port: 8000 }
```

## API

### `POST /v1/whiteboard/animate` (multipart/form-data)

| Field | Type | Default | Description |
|---|---|---|---|
| `image` | file (PNG) | — | required line drawing (e.g. FLUX schnell output) |
| `mask` | file (JSON) | — | optional LabelMe `shapes` segmentation — drives object order |
| `hand` | file (PNG) | bundled `drawing-hand.png` | custom hand sprite |
| `hand_mask` | file (PNG) | bundled `hand-mask.png` | matching hand alpha mask |
| `frame_rate` | int 5-60 | 25 | output FPS |
| `resize` | int 256-2160 | 1080 | square output size |
| `split_len` | int 8-20 | 10 | grid cell size — smaller = finer draw |
| `object_skip_rate` | int 4-15 | 8 | frame skip during per-object pass |
| `bg_object_skip_rate` | int 10-20 | 14 | frame skip during background pass |
| `end_duration_s` | int 0-10 | 3 | freeze on original image at the end |
| `response_format` | str | `url` | `url` returns JSON; `file` streams mp4 |

Sample call (FLUX-generated PNG → animated whiteboard mp4):

```bash
# 1. Generate a line-drawing PNG via flux Docker
curl -X POST http://localhost:4000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt":"single black line drawing, white background, simple geometric, octopus", "size":"1024x1024"}' \
  | jq -r .data[0].b64_json | base64 -d > /tmp/octo.png

# 2. Animate it
curl -X POST http://localhost:8000/v1/whiteboard/animate \
  -F "image=@/tmp/octo.png" \
  -F "frame_rate=25" \
  -F "resize=1080" \
  --output /tmp/octo.mp4

open /tmp/octo.mp4
```

Or with object segmentation (much better quality — draws each object in turn):

```bash
curl -X POST http://localhost:8000/v1/whiteboard/animate \
  -F "image=@./images/1.png" \
  -F "mask=@./images/1.json" \
  -F "response_format=url"
# {"job_id":"...","video_url":"/results/<id>/output.mp4","duration_s":11.4,"frames_total":286,"render_time_s":4.1}
```

Then GET `http://<host>:8000/results/<job_id>/output.mp4` to fetch the mp4.

## Response (url mode)

```json
{
  "job_id": "abc123def456",
  "video_url": "/results/abc123def456/output.mp4",
  "duration_s": 11.44,
  "frames_total": 286,
  "render_time_s": 4.07,
  "wall_time_s": 4.21
}
```

## Hard limits

| Limit | Value | Why |
|---|---|---|
| max image upload | 50MB | sanity |
| max mask JSON | 10MB | LabelMe files rarely exceed 1MB |
| max hand/hand_mask | 10MB each | sprites are tiny (~100KB) |
| max render duration | 300s (`WB_MAX_RENDER_SECONDS`) | even 4K rarely > 60s |

## Use-case — project-philosofi Pattern 9 (Whiteboard)

```
flux Docker  ──>  /v1/images/generations  ──>  line-drawing PNG (1024x1024)
                                                       │
                                                       ▼
                       (optional) LabelMe JSON mask via labelme/anylabeling tool
                                                       │
                                                       ▼
storyboard-whiteboard  ──>  /v1/whiteboard/animate  ──>  MP4 of hand drawing it
                                                       │
                                                       ▼
              Remotion StoryboardWhiteboard composition <OffthreadVideo>
                                                       │
                                                       ▼
                               final pattern-9 ad embedded with TTS narration
```

See `/Users/marcos/projects/canal-dark/project/project-philosofi/pipeline/tech-mapping.md` for the full per-pattern stack.

## License

- Dockerfile / server.py / idle_watchdog.py: Apache 2.0
- engine.py (storyboard-ai/Innovate-Inspire): **GPL-3.0**
- bundled hand sprite assets: same upstream (storyboard-ai, GPL-3.0)
