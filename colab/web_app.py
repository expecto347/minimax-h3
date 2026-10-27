"""Prompt-to-video web page for the Colab MiniMax-H3 deployment (run on the VM).

  python3 web_app.py                       # http://127.0.0.1:7860, reach it via `colab/h3.sh web`
  H3_SHARE=1 H3_AUTH=user:pass python3 web_app.py   # also open a public gradio.live link
"""
import glob
import os

import gradio as gr

import h3_generate as h3

ASPECTS = ["9:16", "16:9", "1:1", "3:4", "4:3", "21:9"]


def installed_presets():
    def have(name, sub):
        return os.path.exists(os.path.join(h3.COMFY, "models", sub, name))
    out = []
    for name, cfg in h3.PRESETS.items():
        files = [(u, "diffusion_models") for u in cfg["unet"].values()]
        files += [(lora, "loras") for stack in cfg["loras"].values() for lora, _ in stack]
        if all(have(f, sub) for f, sub in files):
            out.append(name)
    return out or list(h3.PRESETS)


def history():
    files = glob.glob(os.path.join(h3.COMFY, "output", "h3", "*.mp4"))
    return sorted(files, key=os.path.getmtime, reverse=True)[:30]


def run(prompt, preset, seconds, aspect, mp, steps, lora_strength, seed, ref_images, first_frame,
        progress=gr.Progress()):
    if not prompt.strip():
        raise gr.Error("请输入提示词")
    if ref_images and first_frame:
        raise gr.Error("参考图和首帧图只能二选一")
    lines = []

    def log(msg):
        lines.append(msg)
        progress(0.5, desc=msg)

    progress(0.05, desc="排队中…")
    try:
        paths = h3.generate(prompt, preset=preset, seconds=seconds, aspect=aspect, mp=mp or None, steps=int(steps) or None,
                            lora_strength=lora_strength, seed=int(seed), ref_image=ref_images or [],
                            first_frame=first_frame, log=log)
    except RuntimeError as e:
        raise gr.Error(str(e)[:500])
    video = next((p for p in paths if p.endswith(".mp4")), paths[0])
    return video, "\n".join(lines), history()


def main():
    presets = installed_presets()
    with gr.Blocks(title="MiniMax-H3") as app:
        gr.Markdown("## MiniMax-H3 视频生成")
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="提示词", lines=8,
                                    placeholder="描述画面、动作、镜头和声音。用参考图时用 <Picture 1>、<Picture 2> 指代。")
                with gr.Row():
                    preset = gr.Dropdown(presets, value=presets[0], label="模型")
                    aspect = gr.Radio(ASPECTS, value="9:16", label="画幅")
                seconds = gr.Slider(4, 15, value=5, step=1, label="时长（秒）")
                go = gr.Button("生成", variant="primary")
                with gr.Accordion("参考图 / 首帧图", open=False):
                    ref_images = gr.File(label="参考图（最多 9 张，自动用 Ref2VA）", file_count="multiple",
                                         file_types=["image"], type="filepath")
                    first_frame = gr.Image(label="首帧图（图生视频）", type="filepath")
                with gr.Accordion("高级参数", open=False):
                    mp = gr.Slider(0, 1.5, value=0, step=0.1, label="分辨率（百万像素，0 = 模型默认）")
                    steps = gr.Slider(0, 30, value=0, step=1, label="步数（0 = 模型默认）")
                    lora_strength = gr.Slider(0, 1.5, value=1.0, step=0.05, label="主 LoRA 强度")
                    seed = gr.Number(value=-1, precision=0, label="种子（-1 = 随机）")
            with gr.Column(scale=1):
                video = gr.Video(label="结果", autoplay=True)
                info = gr.Textbox(label="信息", lines=2)
                gallery = gr.Files(value=history(), label="历史视频（点击下载）")
        go.click(run, [prompt, preset, seconds, aspect, mp, steps, lora_strength, seed, ref_images, first_frame],
                 [video, info, gallery])

    auth = tuple(os.environ["H3_AUTH"].split(":", 1)) if os.environ.get("H3_AUTH") else None
    share = os.environ.get("H3_SHARE") == "1"
    if share and not auth:
        raise SystemExit("H3_SHARE=1 needs H3_AUTH=user:password")
    app.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=int(os.environ.get("WEB_PORT", 7860)), share=share, auth=auth,
        allowed_paths=[os.path.join(h3.COMFY, "output")])


if __name__ == "__main__":
    main()
