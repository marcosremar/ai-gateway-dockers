#!/bin/bash
# All weights are baked into the Docker image — just start the server.
exec python3 /app/server.py
