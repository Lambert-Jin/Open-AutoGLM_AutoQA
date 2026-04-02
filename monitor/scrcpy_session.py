"""scrcpy 录屏会话管理 — FIFO 流式解码，实时视频流"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

from monitor.frame_hub import FrameHub
from monitor.models import MonitorConfig

log = logging.getLogger("autoqa:scrcpy")


class ScrcpySession:
    """通过 scrcpy 录制到 FIFO 管道，流式解码实时帧"""

    def __init__(
        self,
        config: MonitorConfig,
        *,
        device_id: str | None,
        frame_hub: FrameHub,
        work_dir: str | Path,
        status_callback: Callable[[str, str | None], None] | None = None,
    ):
        self._config = config
        self._device_id = device_id
        self._frame_hub = frame_hub
        self._work_dir = Path(work_dir)
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._record_path = self._work_dir / f"live.{config.scrcpy_record_format}"
        self._status_callback = status_callback
        self._process: subprocess.Popen[str] | None = None
        self._stop_event = threading.Event()
        self._log_thread: threading.Thread | None = None
        self._decode_thread: threading.Thread | None = None

    @property
    def record_path(self) -> Path:
        return self._record_path

    def start(self) -> None:
        if self._process and self._process.poll() is None:
            return

        scrcpy_path = self._config.scrcpy_path or shutil.which("scrcpy")
        if not scrcpy_path:
            raise RuntimeError("未找到 scrcpy，可通过 --scrcpy-path 指定")

        # 清理旧文件，创建 FIFO 管道
        if self._record_path.exists() or self._record_path.is_fifo():
            self._record_path.unlink()
        os.mkfifo(str(self._record_path))

        cmd = [
            scrcpy_path,
            "--no-control",
            "--no-audio",
            "--no-window",
            "--record",
            str(self._record_path),
            "--record-format",
            self._config.scrcpy_record_format,
            "--video-codec",
            "h264",
            "--video-bit-rate",
            self._config.scrcpy_video_bit_rate,
            "--max-fps",
            str(self._config.scrcpy_max_fps),
        ]
        if self._config.scrcpy_max_size > 0:
            cmd += ["--max-size", str(self._config.scrcpy_max_size)]
        if self._device_id:
            cmd += ["--serial", self._device_id]

        log.info("启动: %s", " ".join(cmd))
        self._notify("starting", "scrcpy 会话启动中")
        self._stop_event.clear()

        # 先启动解码线程（它会阻塞在 FIFO open 上等待写端）
        self._decode_thread = threading.Thread(target=self._decode_loop, name="scrcpy-decode", daemon=True)
        self._decode_thread.start()

        # 再启动 scrcpy 进程（它打开 FIFO 写端，解码线程解除阻塞）
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._log_thread = threading.Thread(target=self._log_loop, name="scrcpy-log", daemon=True)
        self._log_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        process = self._process
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        self._process = None
        # 清理 FIFO
        try:
            if self._record_path.exists():
                self._record_path.unlink()
        except OSError:
            pass
        self._notify("stopped", "scrcpy 会话已停止")

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _log_loop(self) -> None:
        assert self._process is not None
        stdout = self._process.stdout
        if stdout is None:
            return

        for raw_line in stdout:
            line = raw_line.strip()
            if not line:
                continue
            log.debug("[scrcpy] %s", line)
            lower = line.lower()
            if "error" in lower or "failed" in lower:
                self._notify("error", line)
            elif "device:" in lower or "recording" in lower:
                self._notify("running", line)

        returncode = self._process.wait()
        level = "stopped" if self._stop_event.is_set() else "error"
        self._notify(level, f"scrcpy 退出，code={returncode}")

    def _decode_loop(self) -> None:
        try:
            import av
        except ImportError:
            self._notify("degraded", "未安装 PyAV，监控页将使用动作截图兜底刷新")
            return

        self._notify("starting", "等待 scrcpy 视频流...")

        try:
            # av.open 会阻塞直到 FIFO 写端被 scrcpy 打开
            container = av.open(str(self._record_path), format="matroska")
        except Exception as e:
            if not self._stop_event.is_set():
                self._notify("error", f"无法打开视频流: {e}")
            return

        self._notify("running", "视频流已连接，开始实时解码")
        log.info("FIFO 流式解码启动")

        frame_count = 0
        fps_limit = self._config.scrcpy_max_fps
        min_interval = 1.0 / fps_limit if fps_limit > 0 else 0
        last_publish = 0.0

        try:
            for frame in container.decode(video=0):
                if self._stop_event.is_set():
                    break

                # 限帧：跳过多余帧，避免推送速度超过浏览器消费速度
                now = time.monotonic()
                if now - last_publish < min_interval:
                    continue

                self._frame_hub.publish_image(frame.to_image(), source="scrcpy")
                last_publish = now
                frame_count += 1
        except av.error.EOFError:
            log.info("视频流结束 (EOF)")
        except Exception as e:
            if not self._stop_event.is_set():
                log.warning("解码异常: %s", e)
        finally:
            container.close()
            log.info("解码结束，共处理 %d 帧", frame_count)

    def _notify(self, status: str, message: str | None) -> None:
        if self._status_callback:
            self._status_callback(status, message)
