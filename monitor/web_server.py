"""本地监控 HTTP 服务"""

from __future__ import annotations

import json
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from monitor.event_bus import MonitorEventBus
from monitor.frame_hub import FrameHub
from monitor.models import MonitorConfig

# import logging
# logger = logging.getLogger(__name__)


class MonitorWebServer:
    """提供 MJPEG、事件轮询和静态产物文件"""

    def __init__(
        self,
        config: MonitorConfig,
        *,
        frame_hub: FrameHub,
        event_bus: MonitorEventBus,
        artifact_root: Path,
        status_provider,
    ):
        self._config = config
        self._frame_hub = frame_hub
        self._event_bus = event_bus
        self._artifact_root = artifact_root.resolve()
        self._status_provider = status_provider
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str | None:
        if not self._server:
            return None
        host = self._config.host
        if host == "0.0.0.0":
            host = "127.0.0.1"
        return f"http://{host}:{self._server.server_port}"

    def start(self) -> None:
        if self._server is not None:
            return

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    self._write_html(outer._render_index())
                    return
                if parsed.path == "/api/events":
                    params = parse_qs(parsed.query)
                    cursor = int(params.get("cursor", ["0"])[0])
                    events, next_cursor = outer._event_bus.list_since(cursor)
                    self._write_json({"events": events, "cursor": next_cursor})
                    return
                if parsed.path == "/api/status":
                    self._write_json(outer._status_provider())
                    return
                if parsed.path == "/stream.mjpg":
                    self._stream_mjpeg()
                    return
                if parsed.path.startswith("/artifacts/"):
                    rel = parsed.path.removeprefix("/artifacts/")
                    self._serve_artifact(rel)
                    return
                self.send_error(HTTPStatus.NOT_FOUND)

            def log_message(self, format: str, *args) -> None:  # noqa: A003
                # logger.debug("monitor http: " + format, *args)
                return None

            def _write_html(self, html: str) -> None:
                body = html.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _write_json(self, payload) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _stream_mjpeg(self) -> None:
                boundary = "frame"
                self.send_response(HTTPStatus.OK)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
                self.end_headers()

                last_frame_id = -1
                try:
                    while outer._server is not None:
                        frame_id, jpeg_data, _source = outer._frame_hub.wait_for_frame(last_frame_id, timeout=1.0)
                        if frame_id == last_frame_id:
                            frame_id, jpeg_data, _source = outer._frame_hub.snapshot()
                        last_frame_id = frame_id
                        self.wfile.write(f"--{boundary}\r\n".encode("ascii"))
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg_data)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(jpeg_data)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    return

            def _serve_artifact(self, rel_path: str) -> None:
                target = (outer._artifact_root / rel_path).resolve()
                if not target.is_relative_to(outer._artifact_root) or not target.exists():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                mime, _ = mimetypes.guess_type(str(target))
                mime = mime or "application/octet-stream"
                data = target.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer((self._config.host, self._config.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, name="monitor-http", daemon=True)
        self._thread.start()
        # logger.info("监控页已启动: %s", self.url)

    def stop(self) -> None:
        server = self._server
        if server is None:
            return
        server.shutdown()
        server.server_close()
        self._server = None

    def _render_index(self) -> str:
        return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>AutoQA Monitor</title>
  <style>
    :root { color-scheme: light; --bg: #f4efe7; --panel: #fffaf2; --line: #d9cdbd; --text: #221d19; --accent: #e85d2a; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: "SF Mono", "Menlo", monospace; background: linear-gradient(135deg, #efe6d6, #f7f1ea); color: var(--text); }
    .layout { display: grid; grid-template-columns: minmax(380px, 1fr) 360px; gap: 18px; min-height: 100vh; padding: 18px; }
    .panel { background: rgba(255, 250, 242, 0.9); border: 1px solid var(--line); border-radius: 18px; box-shadow: 0 18px 60px rgba(90, 62, 39, 0.08); overflow: hidden; }
    .viewer { position: relative; display: flex; align-items: center; justify-content: center; padding: 18px; min-height: 70vh; }
    .device-shell { position: relative; width: min(420px, 100%); aspect-ratio: 9 / 19.5; border-radius: 28px; overflow: hidden; border: 10px solid #1e1e1e; background: #0d0d0d; box-shadow: 0 22px 50px rgba(0,0,0,0.18); }
    .device-shell img, .device-shell canvas { position: absolute; inset: 0; width: 100%; height: 100%; }
    .device-shell canvas { pointer-events: none; }
    .status { padding: 16px 18px; border-top: 1px solid var(--line); display: flex; gap: 16px; flex-wrap: wrap; font-size: 12px; }
    .sidebar { display: grid; grid-template-rows: auto 1fr auto; }
    .sidebar h2 { margin: 0; padding: 18px; border-bottom: 1px solid var(--line); font-size: 15px; letter-spacing: 0.06em; }
    .timeline { overflow: auto; padding: 12px 14px 18px; }
    .event { padding: 10px 12px; border: 1px solid var(--line); border-radius: 12px; background: rgba(255,255,255,0.7); margin-bottom: 10px; font-size: 12px; line-height: 1.5; }
    .event strong { display: block; font-size: 13px; margin-bottom: 4px; }
    .event a { color: var(--accent); text-decoration: none; }
    .gallery { border-top: 1px solid var(--line); padding: 14px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .gallery img { width: 100%; border-radius: 10px; border: 1px solid var(--line); background: #fff; }
    .muted { color: #7f766d; }
  </style>
</head>
<body>
  <div class="layout">
    <section class="panel">
      <div class="viewer">
        <div class="device-shell">
          <img id="stream" src="/stream.mjpg" alt="live stream" />
          <canvas id="overlay"></canvas>
        </div>
      </div>
      <div class="status" id="status">Monitor booting...</div>
    </section>
    <aside class="panel sidebar">
      <h2>Timeline</h2>
      <div class="timeline" id="timeline"></div>
      <div class="gallery" id="gallery">
        <div><img id="before" alt="before" /></div>
        <div><img id="after" alt="after" /></div>
      </div>
    </aside>
  </div>
  <script>
    const canvas = document.getElementById("overlay");
    const ctx = canvas.getContext("2d");
    const stream = document.getElementById("stream");
    const timeline = document.getElementById("timeline");
    const statusEl = document.getElementById("status");
    const beforeImg = document.getElementById("before");
    const afterImg = document.getElementById("after");
    let cursor = 0;
    let overlay = null;
    let overlayUntil = 0;

    function resizeCanvas() {
      canvas.width = stream.clientWidth;
      canvas.height = stream.clientHeight;
    }

    function drawOverlay() {
      resizeCanvas();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (!overlay || Date.now() > overlayUntil) return;
      ctx.strokeStyle = "#ff5a36";
      ctx.fillStyle = "#ff5a36";
      ctx.lineWidth = 5;
      const x = overlay.x_norm == null ? null : overlay.x_norm * canvas.width;
      const y = overlay.y_norm == null ? null : overlay.y_norm * canvas.height;
      const endX = overlay.end_x_norm == null ? null : overlay.end_x_norm * canvas.width;
      const endY = overlay.end_y_norm == null ? null : overlay.end_y_norm * canvas.height;
      if (x !== null && y !== null) {
        ctx.beginPath();
        ctx.arc(x, y, 16, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(x, y, 30, 0, Math.PI * 2);
        ctx.stroke();
      }
      if (endX !== null && endY !== null && x !== null && y !== null) {
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(endX, endY);
        ctx.stroke();
      }
      ctx.fillStyle = "rgba(20, 20, 20, 0.7)";
      ctx.fillRect(14, 14, Math.max(140, overlay.label.length * 10), 28);
      ctx.fillStyle = "#ffffff";
      ctx.font = "12px Menlo, monospace";
      ctx.fillText(overlay.label, 22, 33);
      requestAnimationFrame(drawOverlay);
    }

    function pushEvent(event) {
      const el = document.createElement("div");
      el.className = "event";
      const time = new Date(event.timestamp * 1000).toLocaleTimeString();
      const title = event.action ? event.action.label : event.step_title || event.event_type;
      const extra = event.message ? `<div>${event.message}</div>` : "";
      const links = [];
      if (event.artifacts?.before_marked) links.push(`<a href="${event.artifacts.before_marked}" target="_blank">before</a>`);
      if (event.artifacts?.after_marked) links.push(`<a href="${event.artifacts.after_marked}" target="_blank">after</a>`);
      el.innerHTML = `<strong>${title}</strong><div class="muted">${time} · ${event.event_type}</div>${extra}<div>${links.join(" / ")}</div>`;
      if (event.artifacts?.before_marked) beforeImg.src = event.artifacts.before_marked;
      if (event.artifacts?.after_marked) afterImg.src = event.artifacts.after_marked;
      if (event.action) {
        overlay = event.action;
        overlayUntil = Date.now() + 1400;
        requestAnimationFrame(drawOverlay);
      }
      timeline.prepend(el);
    }

    async function refreshStatus() {
      const resp = await fetch("/api/status");
      const status = await resp.json();
      statusEl.innerHTML = `URL: ${status.url || "-"} | scrcpy: ${status.scrcpy_status} | frame: ${status.frame_source} | artifacts: ${status.artifact_root}`;
    }

    async function pollEvents() {
      const resp = await fetch(`/api/events?cursor=${cursor}`);
      const data = await resp.json();
      cursor = data.cursor;
      for (const event of data.events) pushEvent(event);
    }

    stream.addEventListener("load", resizeCanvas);
    window.addEventListener("resize", resizeCanvas);
    setInterval(pollEvents, 500);
    setInterval(refreshStatus, 1500);
    refreshStatus();
  </script>
</body>
</html>"""
