#!/bin/bash
# Set up ComfyUI + MiniMax-H3 (PinkCherry NSFW community fine-tune) on a Colab VM.
# Run on the VM: bash setup_vm.sh   (idempotent; re-running skips finished steps)
set -euo pipefail

ROOT=${ROOT:-/content}
COMFY=$ROOT/ComfyUI
M=$COMFY/models

# repo|path in repo|target subdir under models/
FILES=(
  "SexGod1979/PinkCherry-NSFW_MiniMax-H3|1.8-hybrid/PinkCherry_hybrid_v1.8_int8_minimax_h3_pruned.safetensors|diffusion_models"
  "ethanfel/Qwen3-VL-32B-Ultra-Heretic-H3-ComfyUI-INT8-ConvRot|qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors|text_encoders"
  "Comfy-Org/MiniMax-H3|vae/minimax_h3_video_vae_fp16.safetensors|vae"
  "Comfy-Org/MiniMax-H3|vae/minimax_h3_audio_vae_fp32.safetensors|vae"
  "lightx2v/Minimax-h3-Turbo|minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors|loras"
)

if [ ! -d "$COMFY" ]; then
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI "$COMFY"
fi
# Keep Colab's preinstalled CUDA torch; install everything else.
grep -viE '^(torch|torchvision|torchaudio)([<>=~ ]|$)' "$COMFY/requirements.txt" > /tmp/comfy-reqs.txt
pip install -q -r /tmp/comfy-reqs.txt "huggingface_hub[hf_xet]>=0.34"

for entry in "${FILES[@]}"; do
  IFS='|' read -r repo path sub <<< "$entry"
  dest=$M/$sub/$(basename "$path")
  [ -s "$dest" ] && { echo "have $(basename "$path")"; continue; }
  mkdir -p "$M/$sub" "$ROOT/hf-dl"
  hf download "$repo" "$path" --local-dir "$ROOT/hf-dl"
  mv "$ROOT/hf-dl/$path" "$dest"
done
rm -rf "$ROOT/hf-dl"
ls -la "$M"/{diffusion_models,text_encoders,vae,loras}
echo SETUP_DONE
