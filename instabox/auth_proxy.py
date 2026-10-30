"""Password login in front of a local ComfyUI (HTTP + websocket reverse proxy).

  H3_AUTH=user:password python3 auth_proxy.py   # listens on 0.0.0.0:8080 -> 127.0.0.1:8188
"""
import asyncio
import hashlib
import hmac
import html
import os
import re
import secrets

import aiohttp
from aiohttp import web
from yarl import URL

UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:8188")
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "8080"))
USER, PASSWORD = os.environ["H3_AUTH"].split(":", 1)
SECRET = os.environ.get("H3_COOKIE_SECRET") or secrets.token_hex(32)
COOKIE = "h3_session"
HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "upgrade", "host", "content-length",
               "proxy-authenticate", "proxy-authorization", "te", "trailer"}

LOGIN_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ComfyUI 登录</title>
<style>body{{font-family:system-ui,sans-serif;background:#1e1e1e;color:#eee;display:flex;min-height:100vh;
align-items:center;justify-content:center;margin:0}}form{{background:#2b2b2b;padding:28px;border-radius:10px;
width:min(320px,90vw)}}input{{width:100%;box-sizing:border-box;margin:8px 0;padding:10px;border-radius:6px;
border:1px solid #555;background:#1e1e1e;color:#eee}}button{{width:100%;padding:10px;margin-top:8px;border:0;
border-radius:6px;background:#e8590c;color:#fff;font-size:15px}}.err{{color:#ff8787}}</style></head>
<body><form method="post" action="/login"><h3>MiniMax-H3 ComfyUI</h3>{error}
<input name="user" placeholder="用户名" autocomplete="username" required>
<input name="password" type="password" placeholder="密码" autocomplete="current-password" required>
<button>登录</button></form></body></html>"""


USERDATA = re.compile(r"^(/api)?/userdata/([^?]+)(\?.*)?$")


def upstream_url(request):
    # ComfyUI's /userdata/{file} and /userdata/{file}/move/{dest} take a whole file path as one
    # %2F-encoded segment. The instabox edge decodes %2F before we see it, so re-encode the
    # slashes there; everything else is forwarded as the raw request target.
    target = request.raw_path
    m = USERDATA.match(target)
    if m:
        prefix, rest, query = m.group(1) or "", m.group(2), m.group(3) or ""
        parts = rest.split("/move/", 1)
        rest = "/move/".join(p.replace("/", "%2F") for p in parts)
        target = f"{prefix}/userdata/{rest}{query}"
    return URL(UPSTREAM + target, encoded=True)


def token():
    return hmac.new(SECRET.encode(), USER.encode(), hashlib.sha256).hexdigest()


def authed(request):
    return hmac.compare_digest(request.cookies.get(COOKIE, ""), token())


async def login(request):
    if request.method == "POST":
        form = await request.post()
        ok_user = hmac.compare_digest(form.get("user", ""), USER)
        ok_pass = hmac.compare_digest(form.get("password", ""), PASSWORD)
        if ok_user and ok_pass:
            resp = web.HTTPFound("/")
            resp.set_cookie(COOKIE, token(), max_age=30 * 86400, httponly=True, secure=True, samesite="Lax")
            raise resp
        return web.Response(text=LOGIN_PAGE.format(error='<p class="err">用户名或密码错误</p>'),
                            content_type="text/html", status=401)
    return web.Response(text=LOGIN_PAGE.format(error=""), content_type="text/html")


async def proxy_ws(request):
    client = web.WebSocketResponse(max_msg_size=0)
    await client.prepare(request)
    session = request.app["session"]
    async with session.ws_connect(upstream_url(request), max_msg_size=0) as upstream:
        async def pump(src, dst):
            async for msg in src:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await dst.send_str(msg.data)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    await dst.send_bytes(msg.data)
                else:
                    break

        tasks = [asyncio.create_task(pump(client, upstream)), asyncio.create_task(pump(upstream, client))]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in tasks:
            t.cancel()
    return client


async def handler(request):
    if request.path == "/login":
        return await login(request)
    if not authed(request):
        if request.method == "GET" and "text/html" in request.headers.get("Accept", ""):
            raise web.HTTPFound("/login")
        raise web.HTTPUnauthorized()
    if request.headers.get("Upgrade", "").lower() == "websocket":
        return await proxy_ws(request)

    # ComfyUI 403s when Origin and Host differ, and the instabox edge rewrites Host. Logins are
    # already enforced here with a SameSite cookie, so Origin is not forwarded.
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS | {"origin"}}
    async with request.app["session"].request(request.method, upstream_url(request), headers=headers,
                                              data=request.content, allow_redirects=False) as up:
        resp = web.StreamResponse(status=up.status, headers={
            k: v for k, v in up.headers.items() if k.lower() not in HOP_HEADERS})
        await resp.prepare(request)
        async for chunk in up.content.iter_chunked(1 << 16):
            await resp.write(chunk)
        await resp.write_eof()
        return resp


async def on_startup(app):
    app["session"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None),
                                           auto_decompress=False)


async def on_cleanup(app):
    await app["session"].close()


def main():
    app = web.Application(client_max_size=1 << 31)
    app.router.add_route("*", "/{tail:.*}", handler)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    # Open websockets would otherwise hold a restart for aiohttp's default 60 s.
    web.run_app(app, host="0.0.0.0", port=LISTEN_PORT, access_log=None, shutdown_timeout=3)


if __name__ == "__main__":
    main()
