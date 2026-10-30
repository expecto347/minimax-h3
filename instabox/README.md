# instabox 常驻 ComfyUI + Colab GPU 按需执行

instabox 机器（无 GPU）常驻运行 ComfyUI 界面；点「运行」时 `colab_remote` 插件自动开 Colab GPU 执行，
空闲后自动关闭 Colab。

```
浏览器 ──HTTPS──► instabox: auth_proxy.py (登录) ──► ComfyUI (CPU, 只做界面/队列/历史)
                                                        │ colab_remote 插件
                                                        ▼ colab-cli + ssh 端口转发
                                               Colab G4: ComfyUI (GPU) 实际执行
```

## 文件

- `colab_remote/`：ComfyUI 插件。替换 `PromptExecutor.execute_async`：按需 `colab new`，用
  `colab/setup_vm.sh` 在 Colab 装同一版本 ComfyUI、只下载工作流用到的模型；上传输入文件；
  转发进度事件；把输出下载回 instabox；队列空闲 `COLAB_IDLE_STOP_SECONDS`（默认 300）秒后 `colab stop`。
- `auth_proxy.py`：ComfyUI 前面的登录页 + HTTP/WebSocket 反向代理（端口 8080）。
- `prepare_workflows.py`：下载 Mystic V5 作者的两个工作流，把模型路径改成本仓库的文件名。
- `setup_instabox.sh`：装 ComfyUI(CPU)、模型占位文件、工作流、ssh 配置、systemd 服务。

## 部署

```bash
instabox cp -r colab agent:/data/home/minimax-h3/      # 本仓库代码放到 ~/minimax-h3
H3_AUTH=用户名:密码 bash ~/minimax-h3/instabox/setup_instabox.sh
colab ...                                               # 在 instabox 上登录 colab-cli（一次）
instabox port publish agent 8080                        # 得到 https://8080-<id>.instabox.tech
```

配置在 `~/.config/h3.env`（`COLAB_GPU`、`COLAB_IDLE_STOP_SECONDS`、可选的 `HF_TOKEN` 只读令牌等），改完
`sudo systemctl restart h3-comfyui`。日志：`sudo journalctl -u h3-comfyui -f`。

## Google Drive 模型缓存

`~/.config/rclone/rclone.conf` 里的 `gdrive` 远端（`drive.file` 权限，只能看到 rclone 自己上传的文件）
加上 `H3_DRIVE_REMOTE=gdrive:MiniMax-H3-models`：模型照常从 Hugging Face / Civitai 下载（实测更快），
下载失败时改从 Drive 拷贝；Drive 里没有的新模型下载后自动传进 Drive。`H3_DRIVE_MODE=first` 可改为 Drive 优先。
实测（CPU Colab）：Drive 冷启动约 61MB/s（73GB 用 20 分钟），Hugging Face 约 95–150MB/s。`seed_drive.sh` 用便宜的 CPU Colab（约 0.08 units/小时）
预先把 Mystic 全套传进 Drive。

## 运行时能看到的信息

- **节点上的状态文字**（带已用时间）：开机、连接、下载模型（第几个、已下 / 总大小、速度、剩余时间，配进度条）、
  启动远程 ComfyUI、当前执行的节点（加载模型 / 扩写 / 采样 / 解码 / 保存），采样每一步的耗时和剩余时间，最后的总用时。
- **左侧栏 Console**：Colab 上的完整日志（`[Colab 准备]` 是部署阶段，`[Colab]` 是执行阶段），出错时看这里。

## 注意

- 首次运行（或 Colab 关机后再运行）要先开机并下载模型，约 5–10 分钟；界面上会显示进度文字。
- 模型下拉框里的文件在 instabox 上只是空占位，真正的文件在 Colab 上按需下载。
- 生成的视频保存在 instabox 的 `~/ComfyUI/output/`，Colab 关机后不会丢。
- instabox 免费版单次最多运行 480 分钟，机器重启后服务会自动启动。
