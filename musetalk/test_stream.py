"""
Cliente de teste pro endpoint WS /v1/stream do MuseTalk.

Mede latência end-to-end e FPS efetivo. Uso:

  python test_stream.py --server ws://host:8000 --image portrait.png --audio speech.wav

O áudio WAV é convertido pra PCM s16le 16kHz mono e enviado em chunks
de MUSETALK_STREAM_CHUNK_MS (default 1s).
"""

import argparse
import asyncio
import base64
import json
import os
import struct
import sys
import time
import wave
from pathlib import Path

import websockets


def _load_wav_as_pcm(path: Path) -> bytes:
    """Retorna PCM s16le mono 16kHz do WAV de entrada. Resample simples
    via librosa se disponível, senão exige WAV já em formato correto."""
    try:
        import librosa  # type: ignore
        import numpy as np  # type: ignore
        samples, sr = librosa.load(str(path), sr=16000, mono=True)
        return (samples * 32767).astype("<i2").tobytes()
    except ImportError:
        pass

    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getframerate() != 16000 or wf.getsampwidth() != 2:
            raise ValueError(
                f"WAV deve ser 16kHz mono s16le (got {wf.getframerate()}Hz "
                f"{wf.getnchannels()}ch {wf.getsampwidth()*8}bit). "
                f"Instale librosa: pip install librosa"
            )
        return wf.readframes(wf.getnframes())


async def run(args):
    image_b64 = base64.b64encode(Path(args.image).read_bytes()).decode()
    pcm = _load_wav_as_pcm(Path(args.audio))
    audio_sec = len(pcm) / (16000 * 2)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    url = args.server.rstrip("/") + "/v1/stream"
    print(f"[test_stream] WS → {url}")
    print(f"[test_stream] áudio: {audio_sec:.2f}s PCM s16le 16kHz mono")
    print(f"[test_stream] chunk alvo: {args.chunk_ms}ms")

    t_start = time.time()
    first_frame_at = None
    total_frames = 0
    chunk_events = []

    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({
            "op": "init",
            "image": image_b64,
            "fps": args.fps,
            "extra_margin": 10,
            "parsing_mode": "jaw",
        }))
        ready_raw = await ws.recv()
        ready = json.loads(ready_raw)
        if ready.get("status") != "ready":
            print(f"[test_stream] init falhou: {ready}", file=sys.stderr)
            return 1
        print(f"[test_stream] sessão={ready['session_id']} ({time.time()-t_start:.2f}s)")
        t_ready = time.time()

        # Task concorrente: consumir frames
        async def reader():
            nonlocal first_frame_at, total_frames
            while True:
                try:
                    msg = await ws.recv()
                except websockets.ConnectionClosed:
                    return
                if isinstance(msg, bytes):
                    if first_frame_at is None:
                        first_frame_at = time.time()
                    frame_path = out_dir / f"frame_{total_frames:06d}.jpg"
                    frame_path.write_bytes(msg)
                    total_frames += 1
                else:
                    data = json.loads(msg)
                    if data.get("event") == "chunk_done":
                        chunk_events.append(data)
                        print(f"[test_stream] chunk_done: frames={data['frames']} "
                              f"inference_ms={data['inference_ms']} "
                              f"fps_effective={data['fps_effective']}")
                    elif data.get("event") == "flush_done":
                        print(f"[test_stream] flush_done: frames={data['frames']}")

        reader_task = asyncio.create_task(reader())

        # Enviar áudio em chunks "real-time-ish" (sem delay artificial; só para
        # permitir que o server processe à medida que chegar). Para simular
        # streaming real usar asyncio.sleep entre chunks.
        chunk_bytes = int((16000 * 2) * args.chunk_ms / 1000)
        for i in range(0, len(pcm), chunk_bytes):
            await ws.send(pcm[i:i + chunk_bytes])
        await ws.send(json.dumps({"op": "flush"}))
        await asyncio.sleep(0.5)
        await ws.send(json.dumps({"op": "close"}))
        await reader_task

    t_end = time.time()
    print("\n=== RESUMO ===")
    print(f"tempo total              : {t_end - t_start:.2f}s")
    print(f"tempo até ready (preproc): {t_ready - t_start:.2f}s")
    if first_frame_at:
        print(f"tempo até 1º frame       : {first_frame_at - t_ready:.2f}s")
    print(f"frames recebidos         : {total_frames}")
    print(f"duração áudio            : {audio_sec:.2f}s")
    if total_frames and audio_sec > 0:
        inference_seconds = (t_end - t_ready)
        fps_sustained = total_frames / inference_seconds
        rtf = inference_seconds / audio_sec
        print(f"fps sustentado           : {fps_sustained:.1f}")
        print(f"RTF (inf/áudio)          : {rtf:.2f}  ({'REAL-TIME ✅' if rtf <= 1.0 else 'slower than RT'})")
    print(f"frames salvos em         : {out_dir}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=os.environ.get("MUSETALK_URL", "ws://localhost:8000"))
    ap.add_argument("--image",  required=True)
    ap.add_argument("--audio",  required=True)
    ap.add_argument("--outdir", default="/tmp/musetalk_frames")
    ap.add_argument("--fps",    type=int, default=25)
    ap.add_argument("--chunk-ms", dest="chunk_ms", type=int, default=1000)
    args = ap.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
