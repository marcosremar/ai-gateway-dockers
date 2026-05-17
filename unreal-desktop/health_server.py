"""Minimal health server for unreal-desktop image.

Exposes:
  GET /health  -> 200 {"status":"ok"}; touches last_ping file for idle watchdog.
  GET /info    -> system + GPU + Unreal install status.
"""
import json
import os
import subprocess
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

PING_FILE = "/tmp/unreal_desktop_last_ping"
UNREAL_PATH = os.environ.get("UNREAL_PATH", "/root/UnrealEngine")


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
            self._json(200, {"status": "ok", "service": "unreal-desktop", "ts": int(time.time())})
            return
        if self.path == "/info":
            info = {"service": "unreal-desktop"}
            try:
                info["nvidia_smi"] = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                    text=True, stderr=subprocess.DEVNULL,
                ).strip()
            except Exception as e:
                info["nvidia_smi_error"] = str(e)
            try:
                info["vulkan"] = subprocess.check_output(
                    ["vulkaninfo", "--summary"], text=True, stderr=subprocess.DEVNULL,
                )[:500]
            except Exception:
                info["vulkan"] = "unavailable"
            info["unreal_installed"] = os.path.isdir(UNREAL_PATH)
            info["unreal_path"] = UNREAL_PATH
            self._json(200, info)
            return
        self._json(404, {"error": "not found"})

    def log_message(self, *_args):
        pass


if __name__ == "__main__":
    os.makedirs(os.path.dirname(PING_FILE) or "/tmp", exist_ok=True)
    update_ping()
    port = int(os.environ.get("HEALTH_PORT", "8000"))
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
