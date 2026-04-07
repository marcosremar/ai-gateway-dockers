#!/bin/bash
# Run the main server. If it crashes, expose the error on port 8000
# so the gateway health check can see the traceback instead of "connection refused".

cd /app/HybrIK

# Capture crash output
python /app/server.py 2>/tmp/server_crash.log
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    ERROR=$(cat /tmp/server_crash.log 2>/dev/null | tail -50 | python3 -c "
import sys, json
lines = sys.stdin.read()
print(json.dumps({'status': 'crashed', 'exit_code': $EXIT_CODE, 'detail': lines[:2000]}))
" 2>/dev/null || echo '{"status":"crashed","detail":"unknown error"}')

    echo "[hybrik-x] server.py crashed (exit $EXIT_CODE) — serving error on port 8000"
    cat /tmp/server_crash.log

    # Serve error so gateway can read it via /health
    python3 -c "
import http.server, sys, os
ERROR = open('/tmp/server_crash.log').read()[-2000:] if os.path.exists('/tmp/server_crash.log') else 'unknown'
import json
BODY = json.dumps({'status': 'crashed', 'exit_code': $EXIT_CODE, 'detail': ERROR}).encode()
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(BODY)
    def log_message(self, *a): pass
print('[hybrik-x] Error server listening on :8000', flush=True)
http.server.HTTPServer(('0.0.0.0', 8000), H).serve_forever()
"
fi
