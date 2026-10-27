#!/bin/bash
# Start ComfyUI in the background on the Colab VM (localhost only) and wait until it answers.
set -euo pipefail

COMFY=${COMFY_DIR:-/content/ComfyUI}
PORT=${PORT:-8188}

if curl -sf "http://127.0.0.1:$PORT/system_stats" >/dev/null; then
  echo "ComfyUI already running on :$PORT"
  exit 0
fi

cd "$COMFY"
setsid nohup python3 main.py --listen 127.0.0.1 --port "$PORT" ${COMFY_ARGS:-} \
  > /content/comfy.log 2>&1 < /dev/null &

for _ in $(seq 120); do
  curl -sf "http://127.0.0.1:$PORT/system_stats" >/dev/null && { echo "ComfyUI up on :$PORT"; exit 0; }
  sleep 2
done
tail -30 /content/comfy.log
exit 1
