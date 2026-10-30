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
  H3_DRIVE_REMOTE (e.g. gdrive:MiniMax-H3-models): rclone model cache; this host's
    ~/.config/rclone/rclone.conf is copied to the Colab VM to reach it
"""
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
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
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


def ensure_remote(prompt, say):
    """Bring the Colab GPU up with ComfyUI running and the prompt's models present."""
    global _maybe_running
    _maybe_running = True
    if not remote_up():
        if not session_exists():
            say(f"正在启动 Colab GPU（{GPU}）…")
            sh(f"{COLAB} new -s {SESSION} --gpu {GPU}", timeout=900)
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
    say(f"正在准备 ComfyUI 和模型（{len(files)} 个文件，首次约 5–10 分钟）…")
    env = f"COMFY_REF={comfy_ref()} H3_FILES={shlex.quote(' '.join(files))}"
    if DRIVE_REMOTE and os.path.isfile(RCLONE_CONF):
        ssh("mkdir -p ~/.config/rclone && chmod 700 ~/.config/rclone", timeout=60)
        sh(f"scp -q {RCLONE_CONF} {HOST}:.config/rclone/rclone.conf", timeout=60)
        env += f" H3_DRIVE_REMOTE={shlex.quote(DRIVE_REMOTE)}"
    ssh(f"cd /content && {env} bash setup_vm.sh all > setup.log 2>&1 || (tail -20 setup.log; exit 1)", timeout=3600)
    if not remote_up():
        say("正在启动远程 ComfyUI…")
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
            dest = os.path.join(root, item.get("subfolder", ""), item["filename"])
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            q = urllib.parse.urlencode({"filename": item["filename"], "subfolder": item.get("subfolder", ""), "type": kind})
            with urllib.request.urlopen(f"{BASE}/view?{q}", timeout=600) as r, open(dest + ".part", "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            os.replace(dest + ".part", dest)


def run_remote(self, prompt, prompt_id, extra_data):
    srv = self.server
    sid = srv.client_id
    first_node = next(iter(prompt))

    def say(text):
        log.info(text)
        srv.send_progress_text(text, first_node, sid)

    ensure_remote(prompt, say)
    upload_inputs(prompt)

    client_id = uuid.uuid4().hex
    ws = websocket.create_connection(f"ws://127.0.0.1:{PORT}/ws?clientId={client_id}", timeout=1)
    try:
        say("已提交到 Colab GPU，等待执行…")
        resp = http("/prompt", {"prompt": prompt, "client_id": client_id, "prompt_id": prompt_id,
                                "extra_data": {k: v for k, v in extra_data.items() if k == "extra_pnginfo"}})
        if resp.get("node_errors"):
            raise RemoteError(f"remote validation failed: {json.dumps(resp['node_errors'])[:1500]}")
        ui_outputs, meta = {}, {}
        interrupted = False
        while True:
            if not interrupted and comfy.model_management.processing_interrupted():
                interrupted = True
                http("/interrupt", {})
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
                srv.last_node_id = data.get("node")
                if data.get("node") is not None:
                    srv.send_sync(kind, data, sid)
            elif kind in ("execution_success", "execution_error", "execution_interrupted"):
                self.success = kind == "execution_success"
                self.add_message(kind, data, broadcast=kind != "execution_success")
                break
            elif kind in ("progress", "progress_state", "execution_cached"):
                srv.send_sync(kind, data, sid)
        self.history_result = {"outputs": ui_outputs, "meta": meta}
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


def idle_stopper():
    """Stop the Colab session once nothing has been queued for IDLE_STOP seconds."""
    global _maybe_running
    while True:
        time.sleep(30)
        try:
            queue = server.PromptServer.instance.prompt_queue
            with _lock:
                idle = not _busy and queue.get_tasks_remaining() == 0 and time.time() - _last_activity > IDLE_STOP
            if not (idle and _maybe_running):
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
threading.Thread(target=idle_stopper, name="colab-idle-stopper", daemon=True).start()
log.info("colab_remote: prompts run on Colab session %s (%s); idle stop after %ss", SESSION, GPU, IDLE_STOP)

NODE_CLASS_MAPPINGS = {}
