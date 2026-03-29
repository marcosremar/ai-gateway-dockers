"""
BabelCast Test Lifecycle Server — Minimal fake GPU server for testing
the full service lifecycle without real models.

Simulates:
- /health with progressive service loading (downloading → loading → loaded)
- /v1/transcribe (fake STT)
- /v1/translate/text (fake LLM)
- /v1/tts (fake TTS)

Boot: instant. No models, no GPU, no HuggingFace.
"""

import json
import time
import asyncio
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

# Simulate progressive model loading
BOOT_TIME = time.time()
STT_READY_AFTER = 5    # seconds after boot
LLM_READY_AFTER = 10   # seconds after boot
TTS_READY_AFTER = 15   # seconds after boot


def get_service_status(name, ready_after):
    elapsed = time.time() - BOOT_TIME
    if elapsed < ready_after * 0.3:
        return "downloading"
    elif elapsed < ready_after * 0.7:
        return "loading"
    elif elapsed < ready_after:
        return "compiling" if name == "tts" else "loading"
    else:
        return "loaded"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress logs

    def _json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            stt = get_service_status("whisper", STT_READY_AFTER)
            llm = get_service_status("llama_cpp", LLM_READY_AFTER)
            tts = get_service_status("tts", TTS_READY_AFTER)

            self._json(200, {
                "status": "ok",
                "uptime": round(time.time() - BOOT_TIME, 1),
                "services": {
                    "whisper": stt,
                    "llama_cpp": llm,
                    "tts": tts,
                },
                "model_warmth": {
                    "stt": {"warm": stt == "loaded", "requests": 0, "first_latency_ms": None, "avg_latency_ms": None},
                    "llm": {"warm": llm == "loaded", "requests": 0, "first_latency_ms": None, "avg_latency_ms": None},
                    "tts": {"warm": tts == "loaded", "requests": 0, "first_latency_ms": None, "avg_latency_ms": None},
                },
            })
        elif self.path == "/version":
            self._json(200, {"version": "test-lifecycle-1.0", "type": "test"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        if self.path.startswith("/v1/transcribe"):
            # Fake STT — return dummy transcription
            time.sleep(0.05)  # 50ms latency
            self._json(200, {
                "text": "This is a test transcription from the lifecycle server.",
                "language": "en",
                "duration": 1.0,
            })

        elif self.path.startswith("/v1/translate"):
            # Fake LLM — return dummy translation
            try:
                data = json.loads(body)
                text = data.get("text", "hello")
            except:
                text = "hello"
            time.sleep(0.03)  # 30ms latency
            self._json(200, {
                "translated_text": f"[translated] {text}",
                "source_lang": "en",
                "target_lang": "fr",
            })

        elif self.path == "/v1/tts":
            # Fake TTS — return tiny WAV
            time.sleep(0.04)  # 40ms latency
            # Minimal WAV header (44 bytes) + 1000 bytes of silence
            import struct
            data_size = 1000
            wav = bytearray(44 + data_size)
            wav[0:4] = b"RIFF"
            struct.pack_into("<I", wav, 4, 36 + data_size)
            wav[8:12] = b"WAVE"
            wav[12:16] = b"fmt "
            struct.pack_into("<I", wav, 16, 16)
            struct.pack_into("<H", wav, 20, 1)  # PCM
            struct.pack_into("<H", wav, 22, 1)  # mono
            struct.pack_into("<I", wav, 24, 16000)  # sample rate
            struct.pack_into("<I", wav, 28, 32000)  # byte rate
            struct.pack_into("<H", wav, 32, 2)  # block align
            struct.pack_into("<H", wav, 34, 16)  # bits per sample
            wav[36:40] = b"data"
            struct.pack_into("<I", wav, 40, data_size)

            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav)))
            self.end_headers()
            self.wfile.write(bytes(wav))

        else:
            self._json(404, {"error": "not found"})


if __name__ == "__main__":
    port = 8000
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Test lifecycle server running on :{port}")
    print(f"STT ready in {STT_READY_AFTER}s, LLM in {LLM_READY_AFTER}s, TTS in {TTS_READY_AFTER}s")
    server.serve_forever()
