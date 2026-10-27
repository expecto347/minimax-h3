# MiniMax-H3 NSFW 社区模型 · Colab 部署

用 [colab-cli](https://github.com/googlecolab/google-colab-cli) 在 Colab GPU 上跑 ComfyUI + MiniMax-H3 社区 NSFW 模型。
支持两套预设，采样参数都沿用模型作者发布的工作流。

## 预设

### `mystic`（默认）：Mystic XXX v5.0

[Civitai: [MMH3] Mystic XXX](https://civitai.com/models/2856467)，作者 alcaitiff。这是 LoRA，加载在官方剪枝 INT8 底模上。

| 组件 | 来源 | 文件 |
|---|---|---|
| 底模（文生 / 首帧） | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | `minimax_h3_fl2va_pruned_int8_convrot.safetensors`（21 GB） |
| 底模（参考图） | 同上 | `minimax_h3_ref2va_pruned_int8_convrot.safetensors`（21 GB） |
| Mystic LoRA，强度 1.0 | Civitai，下载后做 SHA256 校验 | `MysticXXX_MMH3-V5.safetensors` / `MysticXXX_MMH3-V5-ref2va.safetensors` |
| 4 步 Turbo LoRA，强度 0.75 | [Kijai/MiniMax-H3_comfy](https://huggingface.co/Kijai/MiniMax-H3_comfy) | `minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors` |

采样：res_multistep / simple，8 步，默认约 1MP，参考图缩到最长边 512。

### `pinkcherry`：PinkCherry v1.8-hybrid

[SexGod1979/PinkCherry-NSFW_MiniMax-H3](https://huggingface.co/SexGod1979/PinkCherry-NSFW_MiniMax-H3) 完整微调模型（19.6 GB），配 [lightx2v](https://huggingface.co/lightx2v/Minimax-h3-Turbo) 8 步 Turbo LoRA。
采样：euler / simple，12 步，默认 0.8MP。

### 共用组件

- 文本编码器（去审查）：[ethanfel/Qwen3-VL-32B-Ultra-Heretic](https://huggingface.co/ethanfel/Qwen3-VL-32B-Ultra-Heretic-H3-ComfyUI-INT8-ConvRot)，24.6 GB
- VAE：Comfy-Org 官方视频 / 音频 VAE

## 网页界面（最简单）

`colab/h3.sh up` 部署完会自动启动网页，也可以单独运行 `colab/h3.sh web`。然后打开 http://localhost:7860：

- 输入提示词，选模型和画幅，点「生成」，视频直接显示在页面上
- 「参考图 / 首帧图」：上传参考图会自动切到 Ref2VA，提示词里用 `<Picture 1>` 指代
- 「高级参数」：分辨率、步数、LoRA 强度、种子
- 「历史视频」：以前生成的视频，点击即可下载

没有电脑、只用手机的话，可以开带密码的公网链接（Gradio share）：
`H3_SHARE=1 H3_AUTH=用户名:密码 colab/h3.sh web`，会打印一个 `https://xxx.gradio.live` 地址，72 小时有效。
注意：这个链接任何人拿到都能访问（有密码保护），Colab 条款对隧道类连接也有限制，请自行权衡。

## 命令行用法

前提：本机已安装并登录 `colab`（`uv tool install google-colab-cli`），有 `ssh`。

```bash
colab/h3.sh up                                   # 开 G4、装 ComfyUI、下 mystic 模型、启动服务
H3_PRESET=all colab/h3.sh up                     # 两套模型都下
colab/h3.sh gen "提示词" --seconds 5               # 文生视频，mp4 下载到 ./outputs
colab/h3.sh gen "..." --aspect 16:9 --seed 42
colab/h3.sh gen "<Picture 1> ..." --ref-image a.png      # 参考图（可重复，最多 9 张）
colab/h3.sh gen "..." --first-frame start.png            # 首帧图生视频
colab/h3.sh gen "..." --preset pinkcherry                # 换用 PinkCherry
colab/h3.sh gen "..." --lora-strength 0.8                # 调 Mystic LoRA 强度（作者建议 0.5–1.0）
colab/h3.sh ui                                   # 把 ComfyUI 网页转发到 http://localhost:8188
colab/h3.sh down                                 # 用完务必停机
```

`gen` 的其它参数：`--mp`、`--steps`、`--sampler`、`--scheduler`、`--prefix`，见 `h3_generate.py --help`。
环境变量：`H3_GPU`（默认 `G4`，也可 `H100`/`A100`）、`H3_PRESET`、`H3_SESSION`、`H3_OUT`。
Civitai 如果要求登录才能下载，设置 `CIVITAI_TOKEN` 后在 VM 上重跑 `setup_vm.sh`。

## 实测（G4，RTX PRO 6000 Blackwell 96GB）

- 费率 8.9 compute units/小时
- 安装下载：mystic 约 70 GB，约 10 分钟
- mystic，736×1344、5 秒：文生视频 70 秒，参考图生视频 74 秒
- pinkcherry，928×544、5 秒：36 秒

## 注意

- Colab 机器是临时的，停机后模型要重新下载。
- Colab 禁止制作 deepfake；H3 许可证禁止涉及未成年人、未经同意冒充他人。不要用真实人物的照片或肖像生成色情内容。
- 两个模型都是 MiniMax-H3 的衍生模型。Colab 机房可能在美国或欧盟，这两个地区在 H3 社区许可证的排除范围内，是否适用请自行判断。
