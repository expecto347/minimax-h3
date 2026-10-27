#!/bin/bash
# Local driver for the Colab MiniMax-H3 deployment. Needs colab-cli logged in and ssh.
#
#   colab/h3.sh up                      # create GPU session, install ComfyUI + models, start server
#                                       #   H3_PRESET=mystic (default) | pinkcherry | all
#   colab/h3.sh gen "prompt" [opts...]  # generate; opts go to h3_generate.py; mp4 lands in ./outputs
#   colab/h3.sh web                     # prompt-to-video page at http://localhost:7860
#   colab/h3.sh ui                      # forward ComfyUI to http://localhost:8188
#   colab/h3.sh down                    # stop the session (stops compute-unit billing)
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
SESSION=${H3_SESSION:-h3}
GPU=${H3_GPU:-G4}
HOST=colab-$SESSION
OUT=${H3_OUT:-$PWD/outputs}
export PATH="$HOME/.local/bin:$PATH"

ssh_config() {
  mkdir -p ~/.ssh
  [ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519 -q
  grep -q "^Host $HOST\$" ~/.ssh/config 2>/dev/null && return
  # colab ssh allows one connection per runtime, so multiplex everything over it.
  cat >> ~/.ssh/config <<EOF

Host $HOST
  User root
  ProxyCommand $(command -v colab) ssh --proxy-mode -s $SESSION
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
  ServerAliveInterval 30
  LogLevel ERROR
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h
  ControlPersist 4h
EOF
}

case "${1:-}" in
  up)
    ssh_config
    colab sessions 2>/dev/null | grep -qw "$SESSION" || colab new -s "$SESSION" --gpu "$GPU"
    scp -q "$HERE"/setup_vm.sh "$HERE"/start_comfy.sh "$HERE"/h3_generate.py "$HERE"/web_app.py "$HOST":/content/
    ssh "$HOST" "bash /content/setup_vm.sh ${H3_PRESET:-mystic} && bash /content/start_comfy.sh"
    exec "$0" web
    ;;
  gen)
    shift
    prompt=$1; shift
    args=()
    while [ $# -gt 0 ]; do
      case "$1" in
        --ref-image|--first-frame)
          scp -q "$2" "$HOST":/tmp/"$(basename "$2")"
          args+=("$1" "/tmp/$(basename "$2")"); shift 2 ;;
        *) args+=("$1"); shift ;;
      esac
    done
    remote=$(ssh "$HOST" "python3 /content/h3_generate.py $(printf '%q ' "$prompt" "${args[@]}")")
    mkdir -p "$OUT"
    while read -r f; do
      [ -n "$f" ] && scp -q "$HOST:$f" "$OUT/" && echo "$OUT/$(basename "$f")"
    done <<< "$remote"
    ;;
  web)
    # Optional public link: H3_SHARE=1 H3_AUTH=user:password colab/h3.sh web
    ssh "$HOST" "cd /content && { pgrep -f web_app.py >/dev/null || H3_SHARE=${H3_SHARE:-} H3_AUTH=${H3_AUTH:-} setsid nohup python3 web_app.py > web.log 2>&1 < /dev/null & }
      for i in \$(seq 60); do curl -sf -o /dev/null http://127.0.0.1:7860/ && break; sleep 2; done; grep -o 'https://[a-z0-9.-]*gradio.live' web.log | tail -1"
    ssh -O forward -L 7860:127.0.0.1:7860 "$HOST" 2>/dev/null || true
    echo "Web UI: http://localhost:7860"
    ;;
  ui)
    # Ride the existing multiplexed connection; a second colab ssh would be refused.
    ssh "$HOST" true
    ssh -O forward -L 8188:127.0.0.1:8188 "$HOST"
    echo "ComfyUI: http://localhost:8188  (close with: ssh -O cancel -L 8188:127.0.0.1:8188 $HOST)"
    ;;
  down)
    ssh -O exit "$HOST" 2>/dev/null || true
    colab stop -s "$SESSION"
    ;;
  *)
    sed -n '2,8p' "$0"; exit 1 ;;
esac
