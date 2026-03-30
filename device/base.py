"""设备抽象层：Protocol 定义 + 数据类"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class DeviceType(Enum):
    ADB = "adb"
    # 扩展点：未来支持其他平台时在此添加
    # HDC = "hdc"    # HarmonyOS
    # IOS = "ios"    # iOS (WebDriverAgent)


@dataclass
class DeviceScreenshot:
    """设备截图"""
    base64_data: str
    width: int
    height: int
    is_sensitive: bool = False  # 是否为敏感屏幕（截图受限）


@runtime_checkable
class Device(Protocol):
    """设备操作协议：所有设备实现必须满足的接口"""

    device_id: str | None

    # 截图
    def screenshot(self, timeout: int = 10) -> DeviceScreenshot: ...

    # App 管理
    def current_app(self) -> str: ...
    def current_activity(self) -> str: ...
    def launch_app(self, app_name: str) -> None: ...

    # 导航
    def back(self) -> None: ...
    def home(self) -> None: ...

    # 触控
    def tap(self, x: int, y: int) -> None: ...
    def double_tap(self, x: int, y: int) -> None: ...
    def long_press(self, x: int, y: int, duration_ms: int = 1000) -> None: ...
    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 500) -> None: ...

    # 文字输入
    def type_text(self, text: str) -> None: ...
    def clear_text(self) -> None: ...

    # 键盘管理（从 type_text 中分离，由 ActionExecutor 层调用）
    def switch_to_adb_keyboard(self) -> str | None: ...
    def restore_keyboard(self, original_ime: str | None) -> None: ...

    # 设备信息
    def get_screen_size(self) -> tuple[int, int]: ...

    # 连接
    def is_connected(self) -> bool: ...
    def reconnect(self) -> None: ...

    # 清理
    def kill_all_apps(self) -> None: ...