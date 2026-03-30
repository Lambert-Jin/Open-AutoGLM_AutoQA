"""Android ADB 设备实现"""

import base64
import logging
import struct
import time

from device.base import DeviceScreenshot
from device.command import CommandRunner
from device.apps import get_package_name
from device.timing import TIMING
from device.errors import ScreenshotError, ScreenshotSensitiveError

logger = logging.getLogger(__name__)


class ADBDevice:
    """Android ADB 设备实现，满足 Device Protocol"""

    def __init__(self, device_id: str | None = None):
        self.device_id = device_id
        prefix = ["adb"]
        if device_id:
            prefix += ["-s", device_id]
        self._cmd = CommandRunner(prefix=prefix)

    # ── 截图 ──

    def screenshot(self, timeout: int = 10) -> DeviceScreenshot:
        """
        截图优化：用 exec-out 直接输出到 stdout，省去文件 IO。
        失败时先重试（最多 2 次），敏感屏幕抛专用异常。
        """
        last_error = None
        for attempt in range(3):
            try:
                png_bytes = self._cmd.run_bytes(
                    ["exec-out", "screencap", "-p"], timeout=timeout
                )
                if len(png_bytes) < 100:
                    raise RuntimeError("截图数据过小，可能为空")
                b64 = base64.b64encode(png_bytes).decode()
                width, height = self._parse_png_size(png_bytes)
                return DeviceScreenshot(base64_data=b64, width=width, height=height)
            except RuntimeError as e:
                if attempt == 0:
                    logger.debug("exec-out 截图失败，降级到文件截图: %s", e)
                    try:
                        return self._screenshot_via_file(timeout)
                    except ScreenshotSensitiveError:
                        raise
                    except Exception as file_err:
                        last_error = file_err
                else:
                    last_error = e
                if attempt < 2:
                    logger.warning("截图失败 (attempt %d/3)，0.5s 后重试", attempt + 1)
                    time.sleep(0.5)

        raise ScreenshotError(f"截图连续失败: {last_error}")

    def _screenshot_via_file(self, timeout: int) -> DeviceScreenshot:
        """降级方案：通过文件截图"""
        remote = "/sdcard/autoqa_tmp.png"
        result = self._cmd.run(["shell", "screencap", "-p", remote], timeout=timeout)

        output = result.stdout + result.stderr
        if not result.success or "Status: -1" in output or "Failed" in output:
            logger.warning("截图受限（敏感屏幕），返回兜底黑屏图")
            raise ScreenshotSensitiveError("敏感屏幕截图受限")

        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            local = f.name
        try:
            self._cmd.run(["pull", remote, local], timeout=timeout)
            with open(local, "rb") as f:
                png_bytes = f.read()
            b64 = base64.b64encode(png_bytes).decode()
            width, height = self._parse_png_size(png_bytes)
            return DeviceScreenshot(base64_data=b64, width=width, height=height)
        finally:
            os.unlink(local)
            self._cmd.run(["shell", "rm", "-f", remote], timeout=5)

    @staticmethod
    def create_fallback_screenshot() -> DeviceScreenshot:
        """生成黑色兜底截图（仅敏感屏幕场景使用），带 is_sensitive 标记"""
        import io
        from PIL import Image
        img = Image.new("RGB", (1080, 2400), color=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return DeviceScreenshot(base64_data=b64, width=1080, height=2400, is_sensitive=True)

    @staticmethod
    def _parse_png_size(data: bytes) -> tuple[int, int]:
        """从 PNG 头解析宽高"""
        if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
            width, height = struct.unpack(">II", data[16:24])
            return width, height
        return 1080, 2400  # 兜底

    # ── App 管理 ──

    def current_app(self) -> str:
        result = self._cmd.run(
            ["shell", "dumpsys", "window", "displays"], timeout=5
        )
        for line in result.stdout.splitlines():
            if "mCurrentFocus" in line or "mFocusedApp" in line:
                parts = line.split()
                for part in parts:
                    if "/" in part and "." in part:
                        return part.split("/")[0].rstrip("}")
        return "unknown"

    def current_activity(self) -> str:
        """获取当前 Activity 类名"""
        result = self._cmd.run(
            ["shell", "dumpsys", "window", "displays"], timeout=5
        )
        for line in result.stdout.splitlines():
            if "mCurrentFocus" in line or "mFocusedApp" in line:
                parts = line.split()
                for part in parts:
                    if "/" in part and "." in part:
                        activity = part.split("/")[1].rstrip("}")
                        return activity.split(".")[-1]  # 只取类名
        return "unknown"

    def launch_app(self, app_name: str) -> None:
        """启动 App，优先 am start，降级到 monkey"""
        package = get_package_name(app_name)
        if not package:
            package = app_name

        result = self._cmd.run(
            ["shell", "monkey", "-p", package,
             "-c", "android.intent.category.LAUNCHER", "1"],
            timeout=10,
        )
        if not result.success:
            logger.warning("monkey 启动失败，尝试 am start: %s", result.stderr)
            self._cmd.run(
                ["shell", "am", "start", "-n", f"{package}/.MainActivity"],
                timeout=10,
            )
        time.sleep(TIMING.device.launch_delay)

    # ── 导航 ──

    def back(self) -> None:
        self._cmd.run(["shell", "input", "keyevent", "4"], timeout=5)
        time.sleep(TIMING.device.back_delay)

    def home(self) -> None:
        self._cmd.run(["shell", "input", "keyevent", "KEYCODE_HOME"], timeout=5)
        time.sleep(TIMING.device.home_delay)

    # ── 触控 ──

    def tap(self, x: int, y: int) -> None:
        self._cmd.run(["shell", "input", "tap", str(x), str(y)], timeout=5)
        time.sleep(TIMING.device.tap_delay)

    def double_tap(self, x: int, y: int) -> None:
        self._cmd.run(["shell", "input", "tap", str(x), str(y)], timeout=5)
        time.sleep(TIMING.device.double_tap_interval)
        self._cmd.run(["shell", "input", "tap", str(x), str(y)], timeout=5)
        time.sleep(TIMING.device.double_tap_delay)

    def long_press(self, x: int, y: int, duration_ms: int = 1000) -> None:
        self._cmd.run(
            ["shell", "input", "swipe", str(x), str(y), str(x), str(y), str(duration_ms)],
            timeout=10,
        )
        time.sleep(TIMING.device.long_press_delay)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 500) -> None:
        self._cmd.run(
            ["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms)],
            timeout=10,
        )
        time.sleep(TIMING.device.swipe_delay)

    # ── 文字输入（纯输入，键盘切换由 ActionExecutor 层管理）──

    def type_text(self, text: str) -> None:
        """输入文字，不管理键盘切换"""
        if self._has_adb_keyboard():
            b64_text = base64.b64encode(text.encode("utf-8")).decode()
            self._cmd.run(
                ["shell", "am", "broadcast", "-a", "ADB_INPUT_B64",
                 "--es", "msg", b64_text],
                timeout=10,
            )
        else:
            safe_text = text.replace(" ", "%s").replace("&", "\\&")
            self._cmd.run(
                ["shell", "input", "text", safe_text],
                timeout=10,
            )

    def clear_text(self) -> None:
        if self._has_adb_keyboard():
            self._cmd.run(
                ["shell", "am", "broadcast", "-a", "ADB_CLEAR_TEXT"],
                timeout=5,
            )
        else:
            self._cmd.run(["shell", "input", "keyevent", "KEYCODE_MOVE_HOME"], timeout=5)
            self._cmd.run(
                ["shell", "input", "keyevent", "--longpress"]
                + ["KEYCODE_DEL"] * 50,
                timeout=10,
            )

    # ── 键盘管理（分离，由上层调用）──

    def switch_to_adb_keyboard(self) -> str | None:
        """切换到 ADB Keyboard，返回原始输入法名称。切换后回读验证。"""
        if not self._has_adb_keyboard():
            return None

        result = self._cmd.run(
            ["shell", "settings", "get", "secure", "default_input_method"],
            timeout=5,
        )
        original_ime = result.stdout.strip()

        ADB_IME = "com.android.adbkeyboard/.AdbIME"
        self._cmd.run(["shell", "ime", "set", ADB_IME], timeout=5)

        # 验证切换成功
        verify = self._cmd.run(
            ["shell", "settings", "get", "secure", "default_input_method"],
            timeout=5,
        )
        if ADB_IME not in verify.stdout:
            logger.warning("ADB Keyboard 切换失败，当前 IME: %s", verify.stdout.strip())
            return None

        return original_ime

    def restore_keyboard(self, original_ime: str | None) -> None:
        """恢复原始输入法"""
        if original_ime and original_ime != "null":
            self._cmd.run(["shell", "ime", "set", original_ime], timeout=5)

    def _has_adb_keyboard(self) -> bool:
        """检查 ADB Keyboard 是否已安装"""
        result = self._cmd.run(["shell", "ime", "list", "-s"], timeout=5)
        return "com.android.adbkeyboard" in result.stdout

    # ── 设备信息 ──

    def get_screen_size(self) -> tuple[int, int]:
        result = self._cmd.run(["shell", "wm", "size"], timeout=5)
        # 输出格式: "Physical size: 1080x2400"
        for line in result.stdout.splitlines():
            if "size" in line.lower():
                parts = line.split(":")[-1].strip().split("x")
                if len(parts) == 2:
                    try:
                        return int(parts[0]), int(parts[1])
                    except ValueError:
                        pass
        return 1080, 2400  # 兜底

    # ── 连接 ──

    def is_connected(self) -> bool:
        result = self._cmd.run(["get-state"], timeout=5)
        return result.success and "device" in result.stdout

    def reconnect(self) -> None:
        """重连设备"""
        if self.device_id and ":" in self.device_id:
            # 远程设备（云手机）
            self._cmd.run(["disconnect", self.device_id], timeout=5)
            time.sleep(1)
            self._cmd.run(["connect", self.device_id], timeout=10, retries=3)
        else:
            # USB 设备
            self._cmd.run(["reconnect"], timeout=10)

    # ── 清理 ──

    def kill_all_apps(self) -> None:
        """强制停止所有第三方 app"""
        result = self._cmd.run(["shell", "pm", "list", "packages", "-3"], timeout=10)
        if not result.success:
            logger.warning("获取第三方包名失败: %s", result.stderr)
            self._cmd.run(["shell", "am", "kill-all"], timeout=5)
            return

        for line in result.stdout.splitlines():
            if not line.startswith("package:"):
                continue
            pkg = line[8:]
            self._cmd.run(["shell", "am", "force-stop", pkg], timeout=5)