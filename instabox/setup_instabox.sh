#!/bin/bash
# Set up the always-on ComfyUI front end on an instabox machine (no GPU).
# Prompts run on Colab through the colab_remote custom node; see README.md.
#
#   H3_AUTH=user:password bash setup_instabox.sh
# Expects this repo at $REPO, ComfyUI at $COMFY with a venv at $VENV (install_comfy step below),
# and a logged-in colab-cli (~/.config/colab-cli/token.json).
set -euo pipefail

REPO=${REPO:-$HOME/minimax-h3}
COMFY=${COMFY:-$HOME/ComfyUI}
VENV=${VENV:-$HOME/venv}
ENV_FILE=$HOME/.config/h3.env
SESSION=${COLAB_SESSION:-h3}

# ComfyUI on CPU; torch from the CPU index keeps the install small.
if [ ! -d "$COMFY" ]; then
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI "$COMFY"
fi
if [ ! -x "$VENV/bin/python" ]; then
  uv venv --python 3.13 "$VENV"
  uv pip install --python "$VENV/bin/python" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi
uv pip install --python "$VENV/bin/python" -q -r "$COMFY/requirements.txt" aiohttp websocket-client
command -v colab >/dev/null || [ -x "$HOME/.local/bin/colab" ] || uv tool install google-colab-cli

ln -sfn "$REPO/instabox/colab_remote" "$COMFY/custom_nodes/colab_remote"
# Prompt enhancer nodes, at the same reviewed commit setup_vm.sh installs on Colab.
guide=$COMFY/custom_nodes/ComfyUI-MiniMax-H3-Guide
guide_ref=$(sed -n 's/^H3_GUIDE_REF=//p' "$REPO/colab/setup_vm.sh")
if [ "$(git -C "$guide" rev-parse HEAD 2>/dev/null)" != "$guide_ref" ]; then
  rm -rf "$guide"
  git clone -q https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide "$guide"
  git -C "$guide" checkout -q "$guide_ref"
fi

# Empty stand-ins so the loader dropdowns list every model the Colab side can fetch.
bash "$REPO/colab/setup_vm.sh" --list | while read -r sub name; do
  mkdir -p "$COMFY/models/$sub"
  [ -e "$COMFY/models/$sub/$name" ] || touch "$COMFY/models/$sub/$name"
done

"$VENV/bin/python" "$REPO/instabox/prepare_workflows.py" "$COMFY/user/default/workflows"

# colab ssh allows one connection per runtime, so multiplex everything over it.
mkdir -p ~/.ssh && chmod 700 ~/.ssh
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519 -q
grep -q "^Host colab-$SESSION\$" ~/.ssh/config 2>/dev/null || cat >> ~/.ssh/config <<EOF

Host colab-$SESSION
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

mkdir -p "$(dirname "$ENV_FILE")"
if [ ! -f "$ENV_FILE" ]; then
  : "${H3_AUTH:?set H3_AUTH=user:password for the first run}"
  umask 077
  cat > "$ENV_FILE" <<EOF
H3_AUTH=$H3_AUTH
H3_COOKIE_SECRET=$(openssl rand -hex 32)
COLAB_SESSION=$SESSION
COLAB_GPU=${COLAB_GPU:-G4}
COLAB_IDLE_STOP_SECONDS=${COLAB_IDLE_STOP_SECONDS:-300}
H3_COLAB_DIR=$REPO/colab
H3_DRIVE_REMOTE=${H3_DRIVE_REMOTE:-gdrive:MiniMax-H3-models}
H3_PUBLIC_URL=${H3_PUBLIC_URL:-}
PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
EOF
fi

sudo tee /etc/systemd/system/h3-comfyui.service >/dev/null <<EOF
[Unit]
Description=ComfyUI front end (prompts run on Colab)
After=network-online.target

[Service]
User=$USER
WorkingDirectory=$COMFY
EnvironmentFile=$ENV_FILE
Environment=PYTHONUNBUFFERED=1
ExecStart=$VENV/bin/python main.py --cpu --listen 127.0.0.1 --port 8188 --disable-auto-launch
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo tee /etc/systemd/system/h3-proxy.service >/dev/null <<EOF
[Unit]
Description=Password login in front of ComfyUI
After=h3-comfyui.service

[Service]
User=$USER
EnvironmentFile=$ENV_FILE
ExecStart=$VENV/bin/python $REPO/instabox/auth_proxy.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now h3-comfyui.service h3-proxy.service
sudo systemctl restart h3-comfyui.service h3-proxy.service
echo SETUP_DONE
