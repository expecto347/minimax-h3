"""Download the Mystic XXX v5 author workflows from Civitai and point them at our model files.

  python3 prepare_workflows.py <ComfyUI user workflows dir>
"""
import json
import os
import sys
import urllib.request

WORKFLOWS = {
    "Mystic V5 - 文生视频 (FL2VA).json": "https://civitai.com/api/download/models/3416478?fileId=3309562",
    "Mystic V5 - 参考图 (REF2VA).json": "https://civitai.com/api/download/models/3416478?fileId=3311449",
}
MODEL_EXTS = (".safetensors", ".gguf", ".ckpt", ".pt", ".pth", ".bin")
# The author uses the stock NVFP4 encoder; ours is the uncensored drop-in replacement.
RENAMES = {"qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors": "qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors"}
# This host validates prompts on CPU, where Comfy Kitchen attention is not an option (and it
# cannot run the INT8 ConvRot checkpoints anyway).
WIDGET_FIXES = {"ModelAttentionBackend": {"comfy kitchen attention": "pytorch attention"}}
# ImageConcatMulti (KJNodes, not installed) only builds the REF2VA side-by-side "Grid" video;
# core ImageStitch does the same and repeats the single reference image across the frame batch.
# The author's sample inputs are not shipped; point at ComfyUI's bundled image until one is uploaded.
PLACEHOLDER_IMAGE = "example.png"


def fix(value):
    if isinstance(value, str) and value.lower().endswith(MODEL_EXTS):
        name = value.replace("\\", "/").rsplit("/", 1)[-1]
        return RENAMES.get(name, name)
    return value


def fix_nodes(nodes):
    for node in nodes:
        if isinstance(node.get("widgets_values"), list):
            fixes = WIDGET_FIXES.get(node["type"], {})
            node["widgets_values"] = [fixes.get(v, fix(v)) for v in node["widgets_values"]]
            if node["type"] == "LoadImage" and node["widgets_values"]:
                node["widgets_values"][0] = PLACEHOLDER_IMAGE


def replace_concat(nodes):
    for node in nodes:
        if node["type"] != "ImageConcatMulti":
            continue
        links = {i["name"]: i.get("link") for i in node.get("inputs") or []}
        direction = node["widgets_values"][1] if len(node.get("widgets_values") or []) > 1 else "right"
        node["type"] = "ImageStitch"
        node["inputs"] = [{"name": "image1", "type": "IMAGE", "link": links.get("image_1")},
                          {"name": "image2", "type": "IMAGE", "link": links.get("image_2")}]
        node["widgets_values"] = [direction, False, 0, "black"]


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name, url in WORKFLOWS.items():
        # Civitai's CDN rejects urllib's default User-Agent.
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            wf = json.load(r)
        for nodes in [wf.get("nodes", [])] + [sg.get("nodes", []) for sg in wf.get("definitions", {}).get("subgraphs", [])]:
            replace_concat(nodes)
            fix_nodes(nodes)
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            json.dump(wf, f, ensure_ascii=False)
        print("wrote", name)


if __name__ == "__main__":
    main(sys.argv[1])
