# MiniMax-H3（PinkCherry NSFW 社区版）· Colab 部署

用 [colab-cli](https://github.com/googlecolab/google-colab-cli) 在 Colab GPU 上跑 ComfyUI + MiniMax-H3。

## 用到的模型

| 组件 | 来源 | 文件 | 大小 |
|---|---|---|---|
| 视频模型（NSFW 微调） | [SexGod1979/PinkCherry-NSFW_MiniMax-H3](https://huggingface.co/SexGod1979/PinkCherry-NSFW_MiniMax-H3) | `1.8-hybrid/PinkCherry_hybrid_v1.8_int8_minimax_h3_pruned.safetensors` | 19.6 GB |
| 文本编码器（去审查） | [ethanfel/Qwen3-VL-32B-Ultra-Heretic-H3-ComfyUI-INT8-ConvRot](https://huggingface.co/ethanfel/Qwen3-VL-32B-Ultra-Heretic-H3-ComfyUI-INT8-ConvRot) | `qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors` | 24.6 GB |
| 视频 / 音频 VAE | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | `minimax_h3_video_vae_fp16` / `minimax_h3_audio_vae_fp32` | 5.4 GB |
| 8 步 Turbo LoRA | [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) | `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | 1.8 GB |

采样参数沿用 PinkCherry v1.8 作者的工作流：euler / simple，12 步，Turbo LoRA 强度 1.0，Reference-to-Video 条件。

## 用法

前提：本机已安装并登录 `colab`（`uv tool install google-colab-cli`），有 `ssh`。

```bash
colab/h3.sh up      # 开 G4 (RTX PRO 6000 96GB)、装 ComfyUI、下模型、启动服务，约 10 分钟
colab/h3.sh gen "提示词" --seconds 5                      # 生成，mp4 下载到 ./outputs
colab/h3.sh gen "..." --aspect 16:9 --mp 0.5 --seed 42
colab/h3.sh gen "<Picture 1> ..." --ref-image a.png      # 参考图（可重复，最多 9 张）
colab/h3.sh gen "..." --first-frame start.png            # 首帧图生视频
colab/h3.sh ui      # 把 ComfyUI 网页转发到 http://localhost:8188
colab/h3.sh down    # 用完务必停机
```

`gen` 的其它参数：`--steps`、`--sampler`、`--scheduler`、`--lora-strength`、`--prefix`，见 `h3_generate.py --help`。
环境变量：`H3_GPU`（默认 `G4`，也可 `H100`/`A100`）、`H3_SESSION`、`H3_OUT`。

## 实测（G4，RTX PRO 6000 Blackwell 96GB）

- 费率 8.9 compute units/小时
- 安装下载约 10 分钟（约 51 GB）
- 672×1184、5 秒：首次 80 秒（含加载模型）；928×544、5 秒：36 秒

## 注意

- Colab 机器是临时的，停机后模型要重新下载。
- Colab 禁止制作 deepfake；H3 许可证禁止涉及未成年人、未经同意冒充他人。不要用真实人物的照片或肖像生成色情内容。
- PinkCherry 是 MiniMax-H3 的衍生模型。Colab 机房可能在美国或欧盟，这两个地区在 H3 社区许可证的排除范围内，是否适用请自行判断。
