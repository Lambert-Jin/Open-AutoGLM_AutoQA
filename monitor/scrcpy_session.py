"""scrcpy 录屏会话管理"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

from monitor.frame_hub import FrameHub
from monitor.models import MonitorConfig

# import logging
# logger = logging.getLogger(__name__)


class ScrcpySession:
    """通过 scrcpy 录制到 MKV，并从录制文件解出最新帧"""

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

        if self._record_path.exists():
            self._record_path.unlink()

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

        # logger.info("启动 scrcpy: %s", " ".join(cmd))
        self._notify("starting", "scrcpy 会话启动中")
        self._stop_event.clear()
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._log_thread = threading.Thread(target=self._log_loop, name="scrcpy-log", daemon=True)
        self._log_thread.start()
        self._decode_thread = threading.Thread(target=self._decode_loop, name="scrcpy-decode", daemon=True)
        self._decode_thread.start()

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
            # logger.info("[scrcpy] %s", line)
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

        last_size = -1
        while not self._stop_event.is_set():
            if self._record_path.exists():
                try:
                    current_size = self._record_path.stat().st_size
                except FileNotFoundError:
                    current_size = 0
                if current_size > 0 and current_size != last_size:
                    try:
                        with av.open(str(self._record_path)) as container:
                            frame = None
                            for decoded in container.decode(video=0):
                                frame = decoded
                            if frame is not None:
                                self._frame_hub.publish_image(frame.to_image(), source="scrcpy")
                                last_size = current_size
                    except Exception as e:
                        # logger.debug("scrcpy 帧解码失败，稍后重试: %s", e)
                        pass

            process = self._process
            if process and process.poll() is not None and not self._record_path.exists():
                break
            time.sleep(self._config.scrcpy_poll_interval)

    def _notify(self, status: str, message: str | None) -> None:
        if self._status_callback:
            self._status_callback(status, message)
