# fbx2glb

Blender 4.5 headless FBX→GLB conversion service with the
POSITION-rebase / inverse-bind-matrix compensation post-process
discovered by the autoresearch/fbx2glb research loop. CPU only — no
GPU required.

Metric on the 7-asset Khronos validation set (FBX round-trip vs
reference GLB, animation-weighted RMSE, higher = closer to reference):

| Converter                    | Score (7 pairs) |
|------------------------------|-----------------|
| FBX2glTF                     | ~0.74           |
| Blender defaults             | 0.873           |
| **This service (Blender + post-process)** | **0.891** |

## Run locally

```bash
docker build -t marcosremar/fbx2glb:latest .
docker run --rm -p 8000:8000 marcosremar/fbx2glb:latest
```

Then:

```bash
curl -fsS -X POST -F 'file=@input.fbx' \
  http://localhost:8000/v1/convert --output output.glb
```

## Endpoints

- `GET  /health` — `{ ok, blender }`
- `GET  /` — HTML help
- `POST /v1/convert` — multipart `file` (FBX) → `model/gltf-binary` response.
  Sets `Content-Disposition` with a `.glb` filename derived from the upload,
  plus `X-Convert-Elapsed-Ms` and `X-Blender-Version` headers.

Errors: `400` malformed input, `413` oversize upload, `500` conversion
failure (body carries Blender stderr), `504` wall-clock timeout.

## Env

| Var | Default | Purpose |
|-----|---------|---------|
| `PORT` | `8000` | HTTP port |
| `IDLE_TIMEOUT_MIN` | `15` | Minutes before the container self-exits (`0` disables) |
| `MAX_FBX_BYTES` | `268435456` | Hard cap on upload size (256 MB) |
| `CONVERT_TIMEOUT_SEC` | `240` | Per-request wall-clock budget |

## Deploy to Fly.io

```bash
cd fbx2glb
fly launch --no-deploy --copy-config --name <your-app-name>
fly deploy
```

`fly.toml` is preconfigured for `shared-cpu-2x` / 2 GB (needed head-room
for mid-size Mixamo FBX imports) and `auto_stop_machines = "stop"` so
you only pay while converting. `primary_region = "gru"` (São Paulo) —
edit for your closest region.

## Deploy via ai-gateway

The image is built by `.github/workflows/build-fbx2glb.yml` on every
push to `main` that touches this directory, and pushed to both
`marcosremar/fbx2glb:latest` (Docker Hub) and
`ghcr.io/marcosremar/fbx2glb:latest` (GHCR). The ai-gateway knows how
to pull and run these the same way it does for every other Docker
in this repo — add an entry to the gateway's image registry and
route `/fbx2glb/*` to it.

## How the post-process works

`convert.sh` runs `blender --background --python` to import the FBX
and export a default GLB, then runs a short Python pass that:

1. Finds the skin root joint and computes its world transform `M`.
2. Takes only the **rotational** part of `M` (dropping translation,
   which encodes where the armature sits in the scene and usually
   matches the reference GLB).
3. For every POSITION accessor of a mesh bound to that skin, applies
   `M⁻¹` so the vertex positions are expressed in the root's rest
   frame (matches the Khronos convention of keeping POSITION in
   model-world space).
4. Right-multiplies each inverseBindMatrix by `M` — this is the exact
   compensation that preserves the skinning result: `ib * M * M⁻¹ * v = ib * v`.
5. Clears the root node's rotation since we absorbed it into POSITION + IBM.

The math is in `convert.sh`; the winning branch in the research repo
is `autoresearch/5024f13a-3ca`.
