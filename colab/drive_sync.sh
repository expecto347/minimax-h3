#!/bin/bash
# Copy finished videos to Google Drive every 20s (run on the VM after `colab drivemount`).
#   setsid nohup bash drive_sync.sh > drive_sync.log 2>&1 &
set -uo pipefail

SRC=${COMFY_DIR:-/content/ComfyUI}/output/h3
DEST=${H3_DRIVE_DIR:-/content/drive/MyDrive/MiniMax-H3}

mkdir -p "$DEST"
while true; do
  for f in "$SRC"/*.mp4; do
    [ -e "$f" ] || continue
    name=$(basename "$f")
    [ -e "$DEST/$name" ] && continue
    # Skip files still being written.
    [ $(( $(date +%s) - $(stat -c %Y "$f") )) -lt 10 ] && continue
    cp "$f" "$DEST/$name.part" && mv "$DEST/$name.part" "$DEST/$name" && echo "$(date +%T) saved $name"
  done
  sleep 20
done
