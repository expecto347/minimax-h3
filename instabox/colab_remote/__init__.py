"""Run every queued ComfyUI prompt on a Colab GPU instead of this (CPU-only) machine.

Installed as a ComfyUI custom node on the instabox host. It replaces
PromptExecutor.execute_async, so queueing, history, progress and output previews
keep working through ComfyUI's own machinery while the work happens remotely:

  1. start the Colab GPU session if it is not up (colab-cli), install ComfyUI at
     this host's commit and fetch only the model files the prompt uses
  2. upload the prompt's input files, queue it on the remote ComfyUI over an ssh
     port forward and relay its websocket events to this server's clients
  3. download outputs into this host's output folder as they are produced
  4. stop the Colab session once the queue has been empty for IDLE_STOP_SECONDS

Configuration (environment):
  COLAB_SESSION (h3)  COLAB_GPU (G4)  COLAB_REMOTE_PORT (9188)
  COLAB_IDLE_STOP_SECONDS (300)  H3_COLAB_DIR (dir with setup_vm.sh, start_comfy.sh)
  H3_PUBLIC_URL: this host's published URL; while Colab may be up a websocket is held open
    through it so instabox's 30-minute idle pause cannot freeze this plugin mid-job
  HF_TOKEN: Hugging Face read token, written to the VM's token file (avoids 429s)
  H3_DRIVE_REMOTE (e.g. gdrive:MiniMax-H3-models): rclone model cache; this host's
    ~/.config/rclone/rclone.conf is copied to the Colab VM to reach it
"""
import collections
import hashlib
import hmac
import json
import logging
import os
import shlex
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import uuid

import websocket

import comfy.model_management
import execution
import folder_paths
import server

log = logging.getLogger("colab_remote")

SESSION = os.environ.get("COLAB_SESSION", "h3")
GPU = os.environ.get("COLAB_GPU", "G4")
PORT = int(os.environ.get("COLAB_REMOTE_PORT", "9188"))
IDLE_STOP = int(os.environ.get("COLAB_IDLE_STOP_SECONDS", "300"))
COLAB_DIR = os.environ.get("H3_COLAB_DIR", os.path.expanduser("~/minimax-h3/colab"))
DRIVE_REMOTE = os.environ.get("H3_DRIVE_REMOTE", "")
RCLONE_CONF = os.path.expanduser("~/.config/rclone/rclone.conf")
PUBLIC_URL = os.environ.get("H3_PUBLIC_URL", "").rstrip("/")
COLAB = os.environ.get("COLAB_BIN", os.path.expanduser("~/.local/bin/colab"))
HOST = f"colab-{SESSION}"
REMOTE_COMFY = "/content/ComfyUI"
BASE = f"http://127.0.0.1:{PORT}"
MODEL_EXTS = (".safetensors", ".gguf", ".ckpt", ".pt", ".pth", ".bin", ".sft")

_lock = threading.Lock()
_last_activity = time.time()
_busy = False
_maybe_running = True  # a session may survive a restart of this server; the first idle check finds out


class RemoteError(Exception):
    pass


def sh(cmd, timeout=900):
    """Run a shell command locally; raise RemoteError with its tail on failure."""
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, errors="replace", timeout=timeout)
    if p.returncode != 0:
        raise RemoteError(f"{cmd.split()[0]} failed ({p.returncode}): {(p.stdout + p.stderr)[-1500:]}")
    return p.stdout


def ssh(remote_cmd, timeout=900):
    return sh(f"ssh {HOST} {shlex.quote(remote_cmd)}", timeout=timeout)


def http(path, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def remote_up():
    try:
        http("/system_stats", timeout=3)
        return True
    except Exception:
        return False


def session_exists():
    try:
        return SESSION in sh(f"{COLAB} sessions", timeout=60)
    except RemoteError:
        return False


def comfy_ref():
    return sh(f"git -C {shlex.quote(os.path.dirname(folder_paths.__file__))} rev-parse HEAD", timeout=30).strip()


def model_files(prompt):
    names = set()
    for node in prompt.values():
        for value in node.get("inputs", {}).values():
            if isinstance(value, str) and value.lower().endswith(MODEL_EXTS):
                names.add(os.path.basename(value))
    return sorted(names)


def fmt_bytes(n):
    return f"{n / 2**30:.1f} GB" if n >= 2**30 else f"{n / 2**20:.0f} MB"


def fmt_secs(sec):
    sec = int(sec)
    return f"{sec // 60} 分 {sec % 60} 秒" if sec >= 60 else f"{sec} 秒"


# Friendly names for the nodes whose execution is worth announcing.
NODE_LABELS = {
    "UNETLoader": "加载视频模型", "CLIPLoader": "加载文本编码器", "VAELoader": "加载 VAE",
    "LoraLoaderModelOnly": "加载 LoRA", "MiniMaxH3PromptGuide": "整理想法",
    "MiniMaxH3PromptEnhancer": "扩写提示词（去审查 Qwen3-VL-32B）",
    "MiniMaxH3ImageToVideo": "编码提示词", "MiniMaxH3ReferenceToVideo": "编码提示词和参考图",
    "SamplerCustomAdvanced": "采样生成视频", "KSampler": "采样生成视频",
    "VAEDecode": "解码视频画面", "VAEDecodeAudio": "解码音频", "CreateVideo": "合成视频", "SaveVideo": "保存视频",
}
LOG_NOISE = ("Hint:", "hf-cli", "unauthenticated requests", "hf update", "hf skills")


class Reporter:
    """Status text and progress bars on the UI's nodes, details in the server Console."""

    def __init__(self, srv, sid, prompt_id, prompt):
        self.srv, self.sid, self.prompt_id, self.prompt = srv, sid, prompt_id, prompt
        self.first = next(iter(prompt))
        self.t0 = time.time()
        self._last_text = {}
        self._last_progress_line = 0.0

    def elapsed(self):
        return fmt_secs(time.time() - self.t0)

    def say(self, text, node=None, console=True):
        node = node or self.first
        if self._last_text.get(node) == text:
            return
        self._last_text[node] = text
        if console:
            log.info(text.replace("\n", " · "))
        self.srv.send_progress_text(f"[{self.elapsed()}] {text}", node, self.sid)

    def bar(self, value, maximum, node=None):
        self.srv.send_sync("progress", {"value": int(value), "max": max(int(maximum), 1),
                                        "prompt_id": self.prompt_id, "node": node or self.first}, self.sid)

    def console(self, text, prefix="[Colab]"):
        """Mirror remote log lines into this server's log (the UI's Console panel)."""
        for line in text.replace("\r\n", "\n").split("\n"):
            line = line.split("\r")[-1].rstrip()
            if not line.strip() or any(n in line for n in LOG_NOISE):
                continue
            if "%|" in line or "it/s]" in line or "s/it]" in line:  # tqdm bars: at most one per 10 s
                if time.time() - self._last_progress_line < 10:
                    continue
                self._last_progress_line = time.time()
            print(f"{prefix} {line}", flush=True)


SETUP_POLL = r"""cd /content; printf 'EXIT=%s\n' "$(cat setup.exit 2>/dev/null)"
printf 'STATUS=%s\n' "$(cat h3-status.json 2>/dev/null)"
w=$(sed -n 's/.*"watch":"\([^"]*\)".*/\1/p' h3-status.json 2>/dev/null)
printf 'BYTES=%s\n' "$([ -n "$w" ] && du -sbc $w 2>/dev/null | tail -1 | cut -f1)"
printf 'SIZE=%s\n' "$(stat -c %s setup.log 2>/dev/null || echo 0)"
echo ---LOG---; tail -c +$((OFFSET + 1)) setup.log 2>/dev/null | head -c 20000"""


def run_setup(env, rep):
    """Run setup_vm.sh detached on the VM and report its progress until it exits."""
    env_file = "".join(f"{k}={shlex.quote(v)}\n" for k, v in env.items())
    subprocess.run(["ssh", HOST, "cat > /content/setup.env"], input=env_file, text=True, check=True, timeout=60)
    ssh("cd /content && rm -f setup.exit h3-status.json setup.log && "
        "(setsid nohup bash -c 'set -a; . /content/setup.env; set +a; cd /content; "
        "bash setup_vm.sh all > setup.log 2>&1; echo $? > setup.exit' > /dev/null 2>&1 < /dev/null &)", timeout=60)
    offset, current, samples = 0, None, collections.deque(maxlen=8)
    deadline = time.time() + 3600
    while time.time() < deadline:
        time.sleep(3)
        try:
            head, _, chunk = ssh(SETUP_POLL.replace("OFFSET", str(offset)), timeout=60).partition("---LOG---\n")
        except (RemoteError, subprocess.TimeoutExpired) as e:
            log.warning("setup poll failed: %s", e)
            continue
        fields = dict(line.split("=", 1) for line in head.splitlines() if "=" in line)
        size = int(fields.get("SIZE") or 0)
        offset = size if len(chunk.encode()) < 20000 else offset + 20000
        rep.console(chunk, "[Colab 准备]")
        try:
            st = json.loads(fields.get("STATUS") or "{}")
        except json.JSONDecodeError:
            st = {}
        stage = st.get("stage")
        if stage == "comfy":
            rep.say("安装 / 更新远程 ComfyUI 和节点包…")
        elif stage == "drive":
            rep.say(f"从 Google Drive 拷贝 {st.get('count')} 个模型…")
        elif stage == "upload":
            rep.say(f"把 {st.get('count')} 个新模型备份到 Google Drive…")
        elif stage == "download":
            name, total = st.get("file", ""), int(st.get("total") or 0)
            got = min(int(fields.get("BYTES") or 0), total) if total else int(fields.get("BYTES") or 0)
            if name != current:
                current = name
                samples.clear()
            samples.append((time.time(), got))
            speed = (samples[-1][1] - samples[0][1]) / max(samples[-1][0] - samples[0][0], 1e-6) if len(samples) > 1 else 0
            text = f"下载模型 {st.get('index')}/{st.get('count')}：{name}\n{fmt_bytes(got)}"
            if total:
                text += f" / {fmt_bytes(total)}"
            if speed > 0:
                text += f" · {fmt_bytes(speed)}/s"
                if total and got < total:
                    text += f" · 剩余约 {fmt_secs((total - got) / speed)}"
            rep.say(text, console=False)
            if total:
                rep.bar(got // 2**20, total // 2**20)
        if fields.get("EXIT"):
            if fields["EXIT"].strip() != "0":
                tail = ssh("tail -c 1500 /content/setup.log", timeout=60)
                raise RemoteError(f"setup_vm.sh failed ({fields['EXIT'].strip()}): {tail}")
            return
    raise RemoteError("setup_vm.sh did not finish within an hour")


def ensure_remote(prompt, rep):
    """Bring the Colab GPU up with ComfyUI running and the prompt's models present."""
    global _maybe_running
    _maybe_running = True
    say = rep.say
    if not remote_up():
        if not session_exists():
            say(f"正在启动 Colab GPU（{GPU}），通常需要 1–3 分钟…")
            sh(f"{COLAB} new -s {SESSION} --gpu {GPU}", timeout=900)
            say(f"Colab GPU 已分配")
        say("正在连接 Colab…")
        sh(f"ssh -O exit {HOST} 2>/dev/null; true", timeout=30)
        for _ in range(5):
            try:
                ssh("true", timeout=120)
                break
            except (RemoteError, subprocess.TimeoutExpired):
                time.sleep(5)
        sh(f"scp -q {COLAB_DIR}/setup_vm.sh {COLAB_DIR}/start_comfy.sh {HOST}:/content/", timeout=120)
    files = model_files(prompt)
    say(f"检查 ComfyUI 和 {len(files)} 个模型文件…")
    env = {"COMFY_REF": comfy_ref(), "H3_FILES": " ".join(files)}
    if os.environ.get("HF_TOKEN"):
        # Via stdin into hf's token file, so the token never shows on a command line.
        subprocess.run(["ssh", HOST, "mkdir -p ~/.cache/huggingface && umask 077 && cat > ~/.cache/huggingface/token"],
                       input=os.environ["HF_TOKEN"], text=True, check=True, timeout=60)
    if DRIVE_REMOTE and os.path.isfile(RCLONE_CONF):
        ssh("mkdir -p ~/.config/rclone && chmod 700 ~/.config/rclone", timeout=60)
        sh(f"scp -q {RCLONE_CONF} {HOST}:.config/rclone/rclone.conf", timeout=60)
        env["H3_DRIVE_REMOTE"] = DRIVE_REMOTE
    run_setup(env, rep)
    if not remote_up():
        say("正在启动远程 ComfyUI（加载节点）…")
        ssh("bash /content/start_comfy.sh", timeout=600)
        sh(f"ssh -O forward -L {PORT}:127.0.0.1:8188 {HOST}", timeout=60)
        if not remote_up():
            raise RemoteError("remote ComfyUI is not reachable through the ssh forward")


def upload_inputs(prompt):
    input_dir = folder_paths.get_input_directory()
    for node in prompt.values():
        for value in node.get("inputs", {}).values():
            if not isinstance(value, str) or not value or len(value) > 512:
                continue
            name = value.split(" [")[0]
            local = os.path.join(input_dir, name)
            if os.path.isfile(local):
                remote = f"{REMOTE_COMFY}/input/{name}"
                ssh(f"mkdir -p {shlex.quote(os.path.dirname(remote))}", timeout=60)
                sh(f"scp -q {shlex.quote(local)} {HOST}:{shlex.quote(remote)}", timeout=600)


def download_outputs(output):
    for items in output.values():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict) or "filename" not in item:
                continue
            kind = item.get("type", "output")
            root = folder_paths.get_output_directory() if kind == "output" else folder_paths.get_temp_directory()
            remote_name = item["filename"]
            dest = os.path.join(root, item.get("subfolder", ""), remote_name)
            # Every new Colab VM numbers outputs from 00001 again; never overwrite an earlier result
            # here. Rename instead and point the UI (this dict is relayed after) at the new name.
            stem, ext = os.path.splitext(remote_name)
            n = 1
            while os.path.exists(dest):
                n += 1
                item["filename"] = f"{stem}{n}{ext}"
                dest = os.path.join(root, item.get("subfolder", ""), item["filename"])
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            q = urllib.parse.urlencode({"filename": remote_name, "subfolder": item.get("subfolder", ""), "type": kind})
            with urllib.request.urlopen(f"{BASE}/view?{q}", timeout=600) as r, open(dest + ".part", "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            os.replace(dest + ".part", dest)


def remote_logs_since(last_t):
    """New entries of the remote ComfyUI's log buffer after timestamp last_t."""
    entries = http("/internal/logs/raw", timeout=10).get("entries", [])
    fresh = [e for e in entries if e.get("t", "") > last_t]
    return fresh, (fresh[-1]["t"] if fresh else last_t)


def run_remote(self, prompt, prompt_id, extra_data):
    srv = self.server
    sid = srv.client_id
    rep = Reporter(srv, sid, prompt_id, prompt)
    say = rep.say

    ensure_remote(prompt, rep)
    say("上传输入文件…")
    upload_inputs(prompt)
    try:
        _, log_t = remote_logs_since("")
    except Exception:
        log_t = ""

    client_id = uuid.uuid4().hex
    ws = websocket.create_connection(f"ws://127.0.0.1:{PORT}/ws?clientId={client_id}", timeout=1)
    try:
        say("已提交到 Colab GPU，开始执行…")
        resp = http("/prompt", {"prompt": prompt, "client_id": client_id, "prompt_id": prompt_id,
                                "extra_data": {k: v for k, v in extra_data.items() if k == "extra_pnginfo"}})
        if resp.get("node_errors"):
            raise RemoteError(f"remote validation failed: {json.dumps(resp['node_errors'])[:1500]}")
        ui_outputs, meta = {}, {}
        interrupted = False
        steps = {}  # node -> [(time, value)] for per-step timing
        last_log_poll = 0.0
        while True:
            if not interrupted and comfy.model_management.processing_interrupted():
                interrupted = True
                http("/interrupt", {})
            if time.time() - last_log_poll > 2:
                last_log_poll = time.time()
                try:
                    fresh, log_t = remote_logs_since(log_t)
                    rep.console("".join(e.get("m", "") for e in fresh))
                except Exception as e:
                    log.debug("remote log poll failed: %s", e)
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            if not isinstance(raw, str):
                continue  # binary latent previews are not relayed
            msg = json.loads(raw)
            kind, data = msg.get("type"), msg.get("data", {})
            if data.get("prompt_id") != prompt_id:
                continue
            if kind == "executed":
                download_outputs(data.get("output") or {})
                node = data["node"]
                ui_outputs[node] = data.get("output")
                meta[node] = {"node_id": node, "display_node": data.get("display_node", node),
                              "parent_node": None, "real_node_id": node}
                srv.send_sync(kind, data, sid)
            elif kind == "executing":
                node = data.get("node")
                srv.last_node_id = node
                if node is not None:
                    srv.send_sync(kind, data, sid)
                    label = NODE_LABELS.get(prompt.get(node, {}).get("class_type"))
                    if label:
                        say(f"{label}…", node=data.get("display_node", node))
            elif kind in ("execution_success", "execution_error", "execution_interrupted"):
                self.success = kind == "execution_success"
                self.add_message(kind, data, broadcast=kind != "execution_success")
                break
            elif kind == "progress":
                srv.send_sync(kind, data, sid)
                node, value, total = data.get("node"), data.get("value", 0), data.get("max", 0)
                hist = steps.setdefault(node, [])
                hist.append((time.time(), value))
                text = f"第 {value}/{total} 步"
                if len(hist) > 1 and value > hist[0][1]:
                    per = (hist[-1][0] - hist[0][0]) / (value - hist[0][1])
                    text += f" · 每步 {per:.1f} 秒 · 剩余约 {fmt_secs(per * (total - value))}"
                if total > 1:
                    say(text, node=node, console=False)
            elif kind in ("progress_state", "execution_cached"):
                srv.send_sync(kind, data, sid)
        self.history_result = {"outputs": ui_outputs, "meta": meta}
        try:
            fresh, log_t = remote_logs_since(log_t)
            rep.console("".join(e.get("m", "") for e in fresh))
        except Exception:
            pass
        say(f"完成，总用时 {rep.elapsed()}" if self.success else f"失败（用时 {rep.elapsed()}），详情见 Console")
    finally:
        ws.close()


async def remote_execute_async(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
    global _busy, _last_activity
    self.server.client_id = extra_data.get("client_id")
    self.status_messages = []
    self.success = True
    self.history_result = {"outputs": {}, "meta": {}}
    self.add_message("execution_start", {"prompt_id": prompt_id}, broadcast=False)
    with _lock:
        _busy = True
    try:
        run_remote(self, prompt, prompt_id, extra_data)
    except Exception as e:  # report any failure on the job instead of killing the worker
        log.exception("remote execution failed")
        self.success = False
        node = next(iter(prompt))
        self.add_message("execution_error", {
            "prompt_id": prompt_id, "node_id": node, "node_type": prompt[node]["class_type"], "executed": [],
            "exception_message": f"Colab 远程执行失败：{e}", "exception_type": type(e).__name__,
            "traceback": [], "current_inputs": {}, "current_outputs": {}}, broadcast=False)
    finally:
        with _lock:
            _busy = False
            _last_activity = time.time()
        self.server.last_node_id = None


def remote_busy():
    """True while the Colab ComfyUI still has work queued, e.g. a job this server lost track of
    after a restart. Asked over ssh (an existing session is required, so this never creates one)."""
    if not session_exists():
        return False
    try:
        out = ssh("curl -s -m 10 http://127.0.0.1:8188/queue", timeout=60)
        q = json.loads(out)
        return bool(q.get("queue_running") or q.get("queue_pending"))
    except Exception:
        return False


def idle_stopper():
    """Stop the Colab session once nothing has been queued for IDLE_STOP seconds."""
    global _maybe_running, _last_activity
    while True:
        time.sleep(30)
        try:
            queue = server.PromptServer.instance.prompt_queue
            with _lock:
                idle = not _busy and queue.get_tasks_remaining() == 0 and time.time() - _last_activity > IDLE_STOP
            if not (idle and _maybe_running):
                continue
            if remote_busy():
                with _lock:
                    _last_activity = time.time()
                continue
            if remote_up() or session_exists():
                log.info("queue idle for %ss, stopping Colab session %s", IDLE_STOP, SESSION)
                sh(f"ssh -O exit {HOST} 2>/dev/null; {COLAB} stop -s {SESSION}", timeout=300)
            with _lock:
                if not _busy:
                    _maybe_running = False
        except Exception:
            log.exception("idle check failed")


execution.PromptExecutor.execute_async = remote_execute_async
def keep_awake():
    """Hold a request open on the public URL while Colab may be running.

    instabox pauses an idle machine after 30 minutes, which would freeze this plugin with the
    Colab GPU still billing. An in-progress public-port request counts as activity, so keep a
    websocket through auth_proxy.py (logged in with the same cookie it issues) and ping it.
    """
    user, _ = os.environ.get("H3_AUTH", ":").split(":", 1)
    secret = os.environ.get("H3_COOKIE_SECRET", "")
    if not (PUBLIC_URL and user and secret):
        log.warning("keep_awake disabled: H3_PUBLIC_URL, H3_AUTH or H3_COOKIE_SECRET missing")
        return
    cookie = "h3_session=" + hmac.new(secret.encode(), user.encode(), hashlib.sha256).hexdigest()
    url = PUBLIC_URL.replace("https://", "wss://", 1) + f"/ws?clientId=keepawake-{uuid.uuid4().hex[:8]}"
    while True:
        if not _maybe_running:
            time.sleep(15)
            continue
        try:
            ws = websocket.create_connection(url, cookie=cookie, timeout=30)
            ws.settimeout(5)
            last_ping = 0.0
            try:
                while _maybe_running:
                    if time.time() - last_ping > 30:
                        ws.ping()
                        last_ping = time.time()
                    try:
                        ws.recv()  # drain ComfyUI's broadcasts so this socket never backs up
                    except websocket.WebSocketTimeoutException:
                        pass
            finally:
                ws.close()
        except Exception as e:
            log.warning("keep_awake connection dropped: %s", e)
            time.sleep(30)


threading.Thread(target=idle_stopper, name="colab-idle-stopper", daemon=True).start()
threading.Thread(target=keep_awake, name="colab-keep-awake", daemon=True).start()
log.info("colab_remote: prompts run on Colab session %s (%s); idle stop after %ss", SESSION, GPU, IDLE_STOP)

NODE_CLASS_MAPPINGS = {}
