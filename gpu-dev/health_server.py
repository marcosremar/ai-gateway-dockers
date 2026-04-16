"""Minimal health server for gpu-dev base image.

Exposes:
  GET /health      — always 200 {"status":"ok"}; also touches a last_ping file
                     which the idle watchdog uses to track activity.
  GET /info        — python/torch/cuda version info
"""
import json
import os
import subprocess
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

PING_FILE = "/tmp/gpu_dev_last_ping"


def update_ping():
    try:
        with open(PING_FILE, "w") as f:
            f.write(str(int(time.time())))
    except Exception:
        pass


class Handler(BaseHTTPRequestHandler):
    def _json(self, status, body):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/health":
            update_ping()
            self._json(200, {"status": "ok", "service": "gpu-dev", "ts": int(time.time())})
            return
        if self.path == "/info":
            info = {"service": "gpu-dev"}
            try:
                import torch
                info["pytorch"] = torch.__version__
                info["cuda_available"] = torch.cuda.is_available()
                if torch.cuda.is_available():
                    info["cuda_device"] = torch.cuda.get_device_name(0)
                    info["cuda_version"] = torch.version.cuda
            except Exception as e:
                info["torch_error"] = str(e)
            try:
                info["python"] = subprocess.check_output(["python3", "--version"], text=True).strip()
            except Exception:
                pass
            self._json(200, info)
            return
        self._json(404, {"error": "not found"})

    def log_message(self, *_args):
        pass  # quiet


if __name__ == "__main__":
    os.makedirs(os.path.dirname(PING_FILE) or "/tmp", exist_ok=True)
    update_ping()
    port = int(os.environ.get("HEALTH_PORT", "8000"))
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
