#!/bin/bash
# Set up ComfyUI + MiniMax-H3 community NSFW models on a Colab VM.
# Run on the VM: bash setup_vm.sh [mystic|pinkcherry|all]   (default: mystic)
# Idempotent; re-running skips finished steps.
#   COMFY_REF=<commit>   check out this ComfyUI commit (to match another install)
#   H3_FILES="a b ..."   only fetch these model file names (from any preset)
#   H3_DRIVE_REMOTE=gdrive:dir   rclone remote holding a model cache: files are copied from it
#                        first, and anything fetched from the original source is uploaded to it
#   --list               print "subdir name" for every known model file and exit
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
# Used by the Mystic author's REF2VA workflow; decodes ~2x faster.
EXTRA=(
  "hf|Comfy-Org/MiniMax-H3|vae/minimax_h3_video_vae_int8_convrot.safetensors|vae"
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
  all|--list) FILES=("${COMMON[@]}" "${MYSTIC[@]}" "${PINKCHERRY[@]}" "${EXTRA[@]}") ;;
  *) echo "unknown preset: $PRESET" >&2; exit 1 ;;
esac

if [ "$PRESET" = --list ]; then
  for entry in "${FILES[@]}"; do
    IFS='|' read -r _ _ b sub _ <<< "$entry"
    echo "$sub $(basename "$b")"
  done
  exit 0
fi

if [ -n "${H3_FILES:-}" ]; then
  FILES=("${COMMON[@]}" "${MYSTIC[@]}" "${PINKCHERRY[@]}" "${EXTRA[@]}")
  wanted=()
  for entry in "${FILES[@]}"; do
    IFS='|' read -r _ _ b _ _ <<< "$entry"
    [[ " $H3_FILES " == *" $(basename "$b") "* ]] && wanted+=("$entry")
  done
  FILES=("${wanted[@]}")
fi

if [ ! -d "$COMFY" ]; then
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI "$COMFY"
fi
if [ -n "${COMFY_REF:-}" ] && [ "$(git -C "$COMFY" rev-parse HEAD)" != "$COMFY_REF" ]; then
  git -C "$COMFY" fetch -q --depth 1 origin "$COMFY_REF"
  git -C "$COMFY" checkout -q "$COMFY_REF"
fi
# Keep Colab's preinstalled CUDA torch; install everything else.
grep -viE '^(torch|torchvision|torchaudio)([<>=~ ]|$)' "$COMFY/requirements.txt" > /tmp/comfy-reqs.txt
pip install -q -r /tmp/comfy-reqs.txt "huggingface_hub[hf_xet]>=0.34" gradio

DRIVE=${H3_DRIVE_REMOTE:-}
if [ -n "$DRIVE" ] && ! command -v rclone >/dev/null; then
  rm -rf /tmp/rc && mkdir /tmp/rc
  curl -fsSL -o /tmp/rc/rclone.zip https://downloads.rclone.org/rclone-current-linux-amd64.zip
  unzip -q /tmp/rc/rclone.zip -d /tmp/rc && install -m 755 /tmp/rc/rclone-*-linux-amd64/rclone /usr/local/bin/rclone
fi
cached=""
[ -n "$DRIVE" ] && cached=$(rclone lsf -R --files-only "$DRIVE" 2>/dev/null || true)
uploads=()

for entry in ${FILES[@]+"${FILES[@]}"}; do
  IFS='|' read -r src a b sub sha <<< "$entry"
  name=$(basename "$b")
  dest=$M/$sub/$name
  [ -s "$dest" ] && { echo "have $name"; continue; }
  mkdir -p "$M/$sub"
  if grep -qxF "$sub/$name" <<< "$cached"; then
    t0=$SECONDS
    rclone copyto "$DRIVE/$sub/$name" "$dest" --multi-thread-streams 8 && {
      echo "from drive $name in $((SECONDS - t0))s"; continue; }
    rm -f "$dest"
  fi
  [ -n "$DRIVE" ] && uploads+=("$sub/$name")
  if [ "$src" = hf ]; then
    mkdir -p "$ROOT/hf-dl"
    # Unauthenticated Hub downloads get rate limited (429); back off, then skip the xet CDN.
    for try in 1 2 3 4 5; do
      if [ "$try" -eq 1 ]; then hf download "$a" "$b" --local-dir "$ROOT/hf-dl" && break
      else HF_HUB_DISABLE_XET=1 hf download "$a" "$b" --local-dir "$ROOT/hf-dl" && break; fi
      [ "$try" -eq 5 ] && exit 1
      echo "download failed (try $try), retrying in $((try * 20))s" >&2
      sleep $((try * 20))
    done
    mv "$ROOT/hf-dl/$b" "$dest"
  else
    curl -fsSL ${CIVITAI_TOKEN:+-H "Authorization: Bearer $CIVITAI_TOKEN"} -o "$dest.part" "$a"
    echo "$sha  $dest.part" | sha256sum -c --quiet || { rm -f "$dest.part"; echo "checksum mismatch: $b" >&2; exit 1; }
    mv "$dest.part" "$dest"
  fi
done
rm -rf "$ROOT/hf-dl"
for rel in ${uploads[@]+"${uploads[@]}"}; do
  t0=$SECONDS
  rclone copyto "$M/$rel" "$DRIVE/$rel" --drive-chunk-size 256M && echo "to drive $rel in $((SECONDS - t0))s"
done
ls -la "$M"/{diffusion_models,text_encoders,vae,loras}
echo SETUP_DONE
