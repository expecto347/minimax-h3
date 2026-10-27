"""Generate a MiniMax-H3 video through a running ComfyUI server (run on the Colab VM).

Follows the PinkCherry v1.8 recommended recipe: 8-step turbo LoRA, euler/simple,
12 steps, reference-to-video conditioning.

  python3 h3_generate.py "prompt text" --seconds 5 --aspect 9:16 --mp 0.8
  python3 h3_generate.py "..." --ref-image a.png --ref-image b.png   # <Picture 1>, <Picture 2>
  python3 h3_generate.py "..." --first-frame start.png               # image-to-video
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time
import urllib.request
import uuid

COMFY = os.environ.get("COMFY_DIR", "/content/ComfyUI")
SERVER = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")

UNET = "PinkCherry_hybrid_v1.8_int8_minimax_h3_pruned.safetensors"
CLIP = "qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
TURBO_LORA = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"


def frames_for(seconds):
    # Same snapping as the official template: 24 fps, frame count on the 17k + 5 grid.
    n = max(5, round(seconds * 24))
    return n + (5 - n % 17) % 17


def size_for(aspect, megapixels):
    aw, ah = (float(x) for x in aspect.split(":"))
    w = math.sqrt(megapixels * 1e6 * aw / ah)
    h = w * ah / aw
    return int(round(w / 32) * 32), int(round(h / 32) * 32)


def stage_input(path):
    """Copy a local file into ComfyUI/input and return its name there."""
    name = f"{uuid.uuid4().hex[:8]}_{os.path.basename(path)}"
    shutil.copy(path, os.path.join(COMFY, "input", name))
    return name


def build(args, width, height, length):
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "model": {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": ["unet", 0], "lora_name": TURBO_LORA, "strength_model": args.lora_strength}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "minimax", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
    }
    common = {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": args.prompt,
              "width": width, "height": height, "length": length}
    if args.first_frame:
        g["ff"] = {"class_type": "LoadImage", "inputs": {"image": stage_input(args.first_frame)}}
        g["cond"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": {**common, "first_frame": ["ff", 0]}}
    else:
        inputs = {**common, "audio_vae": ["avae", 0], "ref_image_size": "match"}
        for i, path in enumerate(args.ref_image):
            g[f"ref{i}"] = {"class_type": "LoadImage", "inputs": {"image": stage_input(path)}}
            inputs[f"ref_images.ref_image_{i}"] = [f"ref{i}", 0]
        g["cond"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": inputs}
    g.update({
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": args.seed}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": args.sampler}},
        "sigmas": {"class_type": "BasicScheduler",
                   "inputs": {"model": ["model", 0], "scheduler": args.scheduler, "steps": args.steps, "denoise": 1.0}},
        "guider": {"class_type": "BasicGuider", "inputs": {"model": ["model", 0], "conditioning": ["cond", 0]}},
        "sample": {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                              "sigmas": ["sigmas", 0], "latent_image": ["cond", 1]}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "adecode": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sample", 0], "vae": ["avae", 0]}},
        "video": {"class_type": "CreateVideo", "inputs": {"images": ["decode", 0], "audio": ["adecode", 0], "fps": 24}},
        "save": {"class_type": "SaveVideo",
                 "inputs": {"video": ["video", 0], "filename_prefix": args.prefix, "format": "auto", "codec": "auto"}},
    })
    return g


def call(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(SERVER + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prompt")
    p.add_argument("--seconds", type=float, default=5)
    p.add_argument("--aspect", default="9:16")
    p.add_argument("--mp", type=float, default=0.8, help="output megapixels")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--sampler", default="euler")
    p.add_argument("--scheduler", default="simple")
    p.add_argument("--lora-strength", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--ref-image", action="append", default=[], help="reference image (repeatable, max 9)")
    p.add_argument("--first-frame", help="first frame image (image-to-video)")
    p.add_argument("--prefix", default="h3/H3")
    args = p.parse_args()
    if args.seed is None:
        args.seed = random.randrange(2**48)

    width, height = size_for(args.aspect, args.mp)
    length = frames_for(args.seconds)
    graph = build(args, width, height, length)
    print(f"{width}x{height}, {length} frames, {args.steps} steps, seed {args.seed}", file=sys.stderr)

    pid = call("/prompt", {"prompt": graph, "client_id": uuid.uuid4().hex})["prompt_id"]
    t0 = time.time()
    while True:
        hist = call(f"/history/{pid}").get(pid)
        if hist and hist.get("status", {}).get("completed") is not None:
            break
        time.sleep(2)
    status = hist["status"]
    if status.get("status_str") != "success":
        sys.exit(f"generation failed: {json.dumps(status.get('messages'))[:2000]}")
    for out in hist["outputs"].values():
        for item in out.get("images", []) + out.get("videos", []):
            path = os.path.join(COMFY, "output", item.get("subfolder", ""), item["filename"])
            print(f"done in {time.time() - t0:.0f}s", file=sys.stderr)
            print(path)


if __name__ == "__main__":
    main()
