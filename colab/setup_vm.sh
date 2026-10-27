#!/bin/bash
# Set up ComfyUI + MiniMax-H3 community NSFW models on a Colab VM.
# Run on the VM: bash setup_vm.sh [mystic|pinkcherry|all]   (default: mystic)
# Idempotent; re-running skips finished steps.
set -euo pipefail

PRESET=${1:-${H3_PRESET:-mystic}}
ROOT=${ROOT:-/content}
COMFY=$ROOT/ComfyUI
M=$COMFY/models

# hf|repo|path in repo|models subdir     or     civitai|download url|file name|models subdir|sha256
COMMON=(
  "hf|ethanfel/Qwen3-VL-32B-Ultra-Heretic-H3-ComfyUI-INT8-ConvRot|qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors|text_encoders"
  "hf|Comfy-Org/MiniMax-H3|vae/minimax_h3_video_vae_fp16.safetensors|vae"
  "hf|Comfy-Org/MiniMax-H3|vae/minimax_h3_audio_vae_fp32.safetensors|vae"
)
# Mystic XXX v5.0 by alcaitiff (civitai.com/models/2856467) on the official pruned INT8 bases,
# with the turbo LoRA from the author's workflow.
MYSTIC=(
  "hf|Comfy-Org/MiniMax-H3|diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors|diffusion_models"
  "hf|Comfy-Org/MiniMax-H3|diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors|diffusion_models"
  "hf|Kijai/MiniMax-H3_comfy|loras/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors|loras"
  "civitai|https://civitai.com/api/download/models/3416478?fileId=3309543|MysticXXX_MMH3-V5.safetensors|loras|4168917daeb6e570a004bc6994f73d091b9f3e887f7e946bd00c0f2f7573e5f3"
  "civitai|https://civitai.com/api/download/models/3416478?fileId=3311427|MysticXXX_MMH3-V5-ref2va.safetensors|loras|f57c76e761f7a5113e541ad58e84c648a5fe4d3cfbc277d0c44ae9763da29afe"
)
PINKCHERRY=(
  "hf|SexGod1979/PinkCherry-NSFW_MiniMax-H3|1.8-hybrid/PinkCherry_hybrid_v1.8_int8_minimax_h3_pruned.safetensors|diffusion_models"
  "hf|lightx2v/Minimax-h3-Turbo|minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors|loras"
)
case "$PRESET" in
  mystic) FILES=("${COMMON[@]}" "${MYSTIC[@]}") ;;
  pinkcherry) FILES=("${COMMON[@]}" "${PINKCHERRY[@]}") ;;
  all) FILES=("${COMMON[@]}" "${MYSTIC[@]}" "${PINKCHERRY[@]}") ;;
  *) echo "unknown preset: $PRESET" >&2; exit 1 ;;
esac

if [ ! -d "$COMFY" ]; then
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI "$COMFY"
fi
# Keep Colab's preinstalled CUDA torch; install everything else.
grep -viE '^(torch|torchvision|torchaudio)([<>=~ ]|$)' "$COMFY/requirements.txt" > /tmp/comfy-reqs.txt
pip install -q -r /tmp/comfy-reqs.txt "huggingface_hub[hf_xet]>=0.34"

for entry in "${FILES[@]}"; do
  IFS='|' read -r src a b sub sha <<< "$entry"
  dest=$M/$sub/$(basename "$b")
  [ -s "$dest" ] && { echo "have $(basename "$b")"; continue; }
  mkdir -p "$M/$sub"
  if [ "$src" = hf ]; then
    mkdir -p "$ROOT/hf-dl"
    hf download "$a" "$b" --local-dir "$ROOT/hf-dl"
    mv "$ROOT/hf-dl/$b" "$dest"
  else
    curl -fsSL ${CIVITAI_TOKEN:+-H "Authorization: Bearer $CIVITAI_TOKEN"} -o "$dest.part" "$a"
    echo "$sha  $dest.part" | sha256sum -c --quiet || { rm -f "$dest.part"; echo "checksum mismatch: $b" >&2; exit 1; }
    mv "$dest.part" "$dest"
  fi
done
rm -rf "$ROOT/hf-dl"
ls -la "$M"/{diffusion_models,text_encoders,vae,loras}
echo SETUP_DONE
