#!/bin/bash
# Fill the Google Drive model cache from a CPU-only Colab runtime (no GPU billing), then stop it.
# Run on the instabox host:  bash seed_drive.sh [model file names...]   (default: Mystic set)
set -euo pipefail

export PATH=$HOME/.local/bin:$PATH
REPO=${REPO:-$HOME/minimax-h3}
DRIVE=${H3_DRIVE_REMOTE:-gdrive:MiniMax-H3-models}
SESSION=h3seed
HOST=colab-$SESSION

if [ $# -gt 0 ]; then
  files="$*"
else
  files=$(bash "$REPO/colab/setup_vm.sh" --list | awk '$2 !~ /PinkCherry|turbo_8step/ {print $2}' | xargs)
fi

grep -q "^Host $HOST\$" ~/.ssh/config || cat >> ~/.ssh/config <<EOF

Host $HOST
  User root
  ProxyCommand $HOME/.local/bin/colab ssh --proxy-mode -s $SESSION
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
  ServerAliveInterval 30
  LogLevel ERROR
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h
  ControlPersist 4h
EOF

trap 'ssh -O exit $HOST 2>/dev/null; colab stop -s $SESSION' EXIT
colab sessions | grep -qw "$SESSION" || colab new -s "$SESSION"
ssh "$HOST" 'df -h /content | tail -1; mkdir -p ~/.config/rclone && chmod 700 ~/.config/rclone'
scp -q "$REPO/colab/setup_vm.sh" "$HOST":/content/
scp -q ~/.config/rclone/rclone.conf "$HOST":.config/rclone/rclone.conf
ssh "$HOST" "cd /content && H3_FILES='$files' H3_DRIVE_REMOTE='$DRIVE' bash setup_vm.sh all > setup.log 2>&1; grep -E 'drive|have|SETUP_DONE|rror' setup.log"
# Measure a real cold start: drop every local model and fetch them back from Drive.
ssh "$HOST" "rm -f /content/ComfyUI/models/*/*.safetensors && cd /content && H3_FILES='$files' H3_DRIVE_REMOTE='$DRIVE' bash setup_vm.sh all > restore.log 2>&1; grep -E 'from drive|failed' restore.log; du -sh /content/ComfyUI/models"
rclone lsl "$DRIVE"
