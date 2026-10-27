"""Generate a MiniMax-H3 video through a running ComfyUI server (run on the Colab VM).

Presets follow each community model author's published workflow:
  mystic      Mystic XXX v5.0 LoRA on the official pruned INT8 bases (default)
  pinkcherry  PinkCherry v1.8-hybrid NSFW fine-tune

  python3 h3_generate.py "prompt text" --seconds 5 --aspect 9:16
  python3 h3_generate.py "..." --ref-image a.png --ref-image b.png   # <Picture 1>, <Picture 2>
  python3 h3_generate.py "..." --first-frame start.png               # image-to-video
  python3 h3_generate.py "..." --preset pinkcherry
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

CLIP = "qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
MYSTIC_TURBO = "minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors"
PINKCHERRY = "PinkCherry_hybrid_v1.8_int8_minimax_h3_pruned.safetensors"

# unet/loras are keyed by variant: "fl2va" (text / first-frame) or "ref2va" (reference images).
PRESETS = {
    "mystic": {
        "unet": {"fl2va": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                 "ref2va": "minimax_h3_ref2va_pruned_int8_convrot.safetensors"},
        "loras": {"fl2va": [("MysticXXX_MMH3-V5.safetensors", 1.0), (MYSTIC_TURBO, 0.75)],
                  "ref2va": [("MysticXXX_MMH3-V5-ref2va.safetensors", 1.0), (MYSTIC_TURBO, 0.75)]},
        "text_variant": "fl2va",
        "sampler": "res_multistep", "scheduler": "simple", "steps": 8, "mp": 1.0, "ref_max": 512,
    },
    "pinkcherry": {
        "unet": {"fl2va": PINKCHERRY, "ref2va": PINKCHERRY},
        "loras": {v: [("minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", 1.0)] for v in ("fl2va", "ref2va")},
        "text_variant": "ref2va",
        "sampler": "euler", "scheduler": "simple", "steps": 12, "mp": 0.8, "ref_max": None,
    },
}


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


def build(args, preset, width, height, length):
    if args.ref_image:
        variant = "ref2va"
    elif args.first_frame:
        variant = "fl2va"
    else:
        variant = preset["text_variant"]

    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": preset["unet"][variant], "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "minimax", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
    }
    model = ["unet", 0]
    for i, (lora, strength) in enumerate(preset["loras"][variant]):
        if i == 0 and args.lora_strength is not None:
            strength = args.lora_strength
        g[f"lora{i}"] = {"class_type": "LoraLoaderModelOnly",
                         "inputs": {"model": model, "lora_name": lora, "strength_model": strength}}
        model = [f"lora{i}", 0]

    common = {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": args.prompt,
              "width": width, "height": height, "length": length}
    if variant == "ref2va":
        inputs = {**common, "audio_vae": ["avae", 0], "ref_image_size": "match"}
        for i, path in enumerate(args.ref_image):
            g[f"ref{i}"] = {"class_type": "LoadImage", "inputs": {"image": stage_input(path)}}
            image = [f"ref{i}", 0]
            if preset["ref_max"]:
                g[f"refscale{i}"] = {"class_type": "ImageScaleToMaxDimension",
                                     "inputs": {"image": image, "upscale_method": "area", "largest_size": preset["ref_max"]}}
                image = [f"refscale{i}", 0]
            inputs[f"ref_images.ref_image_{i}"] = image
        g["cond"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": inputs}
    else:
        inputs = dict(common)
        if args.first_frame:
            g["ff"] = {"class_type": "LoadImage", "inputs": {"image": stage_input(args.first_frame)}}
            inputs["first_frame"] = ["ff", 0]
        g["cond"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": inputs}

    g.update({
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": args.seed}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": args.sampler or preset["sampler"]}},
        "sigmas": {"class_type": "BasicScheduler",
                   "inputs": {"model": model, "scheduler": args.scheduler or preset["scheduler"],
                              "steps": args.steps or preset["steps"], "denoise": 1.0}},
        "guider": {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": ["cond", 0]}},
        "sample": {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                              "sigmas": ["sigmas", 0], "latent_image": ["cond", 1]}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "adecode": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sample", 0], "vae": ["avae", 0]}},
        "video": {"class_type": "CreateVideo", "inputs": {"images": ["decode", 0], "audio": ["adecode", 0], "fps": 24}},
        "save": {"class_type": "SaveVideo",
                 "inputs": {"video": ["video", 0], "filename_prefix": args.prefix or f"h3/{args.preset}",
                            "format": "auto", "codec": "auto"}},
    })
    return g, variant


def call(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(SERVER + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def generate(prompt, preset="mystic", seconds=5, aspect="9:16", mp=None, steps=None, sampler=None,
             scheduler=None, lora_strength=None, seed=None, ref_image=(), first_frame=None, prefix=None,
             log=lambda msg: print(msg, file=sys.stderr)):
    """Queue one generation on ComfyUI, wait for it, and return the output file paths."""
    args = argparse.Namespace(prompt=prompt, preset=preset, steps=steps, sampler=sampler, scheduler=scheduler,
                              lora_strength=lora_strength, ref_image=list(ref_image), first_frame=first_frame,
                              prefix=prefix, seed=random.randrange(2**48) if seed is None or seed < 0 else seed)
    cfg = PRESETS[preset]
    width, height = size_for(aspect, mp or cfg["mp"])
    length = frames_for(seconds)
    graph, variant = build(args, cfg, width, height, length)
    log(f"{preset}/{variant}: {width}x{height}, {length} frames, seed {args.seed}")

    pid = call("/prompt", {"prompt": graph, "client_id": uuid.uuid4().hex})["prompt_id"]
    t0 = time.time()
    while True:
        hist = call(f"/history/{pid}").get(pid)
        if hist and hist.get("status", {}).get("completed") is not None:
            break
        time.sleep(2)
    status = hist["status"]
    if status.get("status_str") != "success":
        raise RuntimeError(f"generation failed: {json.dumps(status.get('messages'))[:2000]}")
    log(f"done in {time.time() - t0:.0f}s")
    return [os.path.join(COMFY, "output", item.get("subfolder", ""), item["filename"])
            for out in hist["outputs"].values()
            for item in out.get("images", []) + out.get("videos", [])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prompt")
    p.add_argument("--preset", choices=sorted(PRESETS), default="mystic")
    p.add_argument("--seconds", type=float, default=5)
    p.add_argument("--aspect", default="9:16")
    p.add_argument("--mp", type=float, help="output megapixels (preset default)")
    p.add_argument("--steps", type=int)
    p.add_argument("--sampler")
    p.add_argument("--scheduler")
    p.add_argument("--lora-strength", type=float, help="strength of the preset's main LoRA")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--ref-image", action="append", default=[], help="reference image (repeatable, max 9)")
    p.add_argument("--first-frame", help="first frame image (image-to-video)")
    p.add_argument("--prefix")
    try:
        paths = generate(**vars(p.parse_args()))
    except RuntimeError as e:
        sys.exit(str(e))
    print("\n".join(paths))


if __name__ == "__main__":
    main()
