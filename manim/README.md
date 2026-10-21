# manim — Manim Community animated diagrams (open-source)

Self-hosted [Manim Community](https://github.com/ManimCommunity/manim) (MIT) served behind ai-gateway. Custom `POST /v1/diagrams/generate` endpoint so canal-dark / project-philosofi can render exact, programmatic technical diagrams (Pattern 4 Blueprint Stoic) and partial whiteboard-style animations (Pattern 9) without paid tools (Midjourney, VideoScribe, etc).

## Why Manim instead of FLUX (or alongside)

FLUX is great for vibes-y blueprints but bad at exact relations: arrows that have to land on the right node, callouts with precise kN labels, geometric proofs, vector field flows. Manim is code-driven, so the output is reproducible and the diagram is mathematically exact.

| Pattern | Tool | Use |
|---|---|---|
| 4 Blueprint Stoic | **Manim** primary | Dichotomy of control, virtue cone, four-cause schematics, syllogisms |
| 4 Blueprint Stoic | FLUX backup | Cinematic blueprint background plates |
| 9 Whiteboard | **Manim** partial | Geometric scribbles, arrow draw-on, formula reveal |
| 9 Whiteboard | rough.js + Remotion | Hand-drawn texture wrap |

## Build

```bash
cd ai-gateway-dockers/manim
docker build -t marcosremar/manim:latest .
```

## Run (local CPU)

```bash
docker run -p 8000:8000 -p 22:22 \
  -e IDLE_TIMEOUT_MIN=15 \
  marcosremar/manim:latest
```

No GPU flag, no HuggingFace cache mount — Manim has no model weights.

## Run (Vast.ai CPU via ai-gateway)

```bash
ai-gateway gpu deploy --image marcosremar/manim:latest --cpu-only
# returns: { url: https://xxx.proxy.vast.ai, port: 8000 }
```

## API

```bash
curl http://localhost:8000/v1/diagrams/generate \
  -H "Content-Type: application/json" \
  -d '{
    "scene_class": "BlueprintDichotomy",
    "scene_code": "from manim import *\n\nclass BlueprintDichotomy(Scene):\n    def construct(self):\n        title = Text(\"Dichotomy of Control\", color=BLUE).to_edge(UP)\n        in_circle = Circle(radius=2, color=GOLD).shift(LEFT*3)\n        out_circle = Circle(radius=2, color=GRAY).shift(RIGHT*3)\n        in_lbl = Text(\"In our control\", font_size=24).next_to(in_circle, DOWN)\n        out_lbl = Text(\"Not in our control\", font_size=24).next_to(out_circle, DOWN)\n        self.play(Write(title))\n        self.play(Create(in_circle), Create(out_circle))\n        self.play(Write(in_lbl), Write(out_lbl))\n        self.wait(2)",
    "quality": "h",
    "response_format": "url"
  }'
```

Response (url mode):

```json
{
  "video_url": "/results/abc123def456/output.mp4",
  "duration_s": 4.2,
  "render_time_s": 8.1,
  "job_id": "abc123def456"
}
```

Response (b64_json mode): `{ "video_b64": "...", "duration_s": ..., "render_time_s": ... }`.

## Quality presets

| Flag | Resolution | FPS | Use |
|---|---|---|---|
| `l` | 480p | 15 | dev iteration |
| `m` | 720p | 30 | preview |
| `h` | 1080p | 60 | shipping |

## Security — AST sandbox

Manim runs arbitrary Python, so `server.py` parses each `scene_code` with the `ast` module before writing it to disk. Rejected:

- imports of `os`, `sys`, `subprocess`, `socket`, `shutil`, `urllib`, `http`, `requests`, `httpx`, `pickle`, `ctypes`, `importlib`, `threading`, `multiprocessing`, etc.
- builtin calls to `eval`, `exec`, `compile`, `open`, `__import__`, `getattr`, `setattr`, `input`
- attribute access on `__class__`, `__bases__`, `__subclasses__`, `__mro__`, `__globals__`, `__builtins__`, etc.
- `global` / `nonlocal` statements

Whitelisted imports: `manim`, `numpy`, `math` only. Anything else returns 400.

It is not a perfect sandbox — Python AST sandboxing never is — but it raises the bar enough that a typical prompt-injected scene fails closed. Run the container as non-root in production if untrusted code is expected.

Hard limits:

- `MAX_CODE_SIZE` 64KB per scene
- `MAX_RENDER_SECONDS` 180s subprocess timeout (manim killed at 3 min)

## Use cases — project-philosofi patterns

| Pattern | Manim scene archetype |
|---|---|
| 4 Blueprint Stoic | `Circle` + `Arrow` + `Text` overlays for dichotomy / four causes / virtue triangle. `BLUE_E` palette + `GOLD` accent matches `#1a3a52` / `#c9a961`. |
| 9 Whiteboard | `Write()` / `Create()` animations on `MathTex` and `SVGMobject` with stroke-dasharray feel. Combine with `rough.js` SVG output if more hand-drawn jitter needed. |

See `/Users/marcos/projects/canal-dark/project/project-philosofi/pipeline/tech-mapping.md` for the per-pattern stack.

## License

Dockerfile/server: Apache 2.0. Manim Community: MIT.
