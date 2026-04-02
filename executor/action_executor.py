"""统一动作执行器：接收 UnifiedAction，调用 Device 执行"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from device import Device
from device.timing import TIMING
from executor.actions import ActionType, UnifiedAction

log_action = logging.getLogger("autoqa:action")


@dataclass
class ActionExecuteResult:
    """动作执行结果"""

    success: bool
    should_finish: bool = False
    message: str | None = None


class ActionExecutor:
    """
    替代 phone_agent.ActionHandler。

    改进：
    - 输入是 UnifiedAction（坐标已转换），不做坐标转换
    - 通过 Device 实例操作设备，不依赖全局状态
    - 键盘管理提升到此层（type_text 时自动切换/恢复键盘）
    - 支持确认/接管回调
    """

    def __init__(
        self,
        device: Device,
        confirmation_callback: Callable[[str], bool] | None = None,
        takeover_callback: Callable[[str], None] | None = None,
    ):
        self._device = device
        self._confirmation_callback = confirmation_callback or self._default_confirmation
        self._takeover_callback = takeover_callback or self._default_takeover

    def execute(self, action: UnifiedAction) -> ActionExecuteResult:
        """执行一个 UnifiedAction"""
        if action.is_finish:
            return ActionExecuteResult(
                success=True, should_finish=True, message=action.text
            )

        handler = self._HANDLERS.get(action.type)
        if not handler:
            log_action.warning("未知动作类型: %s", action.type)
            return ActionExecuteResult(
                success=False, message="Unknown action: " + str(action.type)
            )

        try:
            handler(self, action)
            return ActionExecuteResult(success=True)
        except Exception as e:
            log_action.error("动作执行失败 [%s]: %s", action.type.value, e)
            return ActionExecuteResult(success=False, message=str(e))

    # ── 动作处理器 ──

    def _tap(self, a: UnifiedAction):
        # 带 message 的 Tap 需要确认
        if a.text and self._confirmation_callback:
            if not self._confirmation_callback(a.text):
                log_action.info("用户取消操作: %s", a.text)
                return
        self._device.tap(a.x, a.y)

    def _double_tap(self, a: UnifiedAction):
        self._device.double_tap(a.x, a.y)

    def _long_press(self, a: UnifiedAction):
        self._device.long_press(a.x, a.y, a.duration_ms or 1000)

    def _swipe(self, a: UnifiedAction):
        self._device.swipe(a.x, a.y, a.end_x, a.end_y, a.duration_ms or 500)

    def _scroll(self, a: UnifiedAction):
        cx = a.x or 540
        cy = a.y or 1200
        dist = 600
        offsets = {
            "up": (0, -dist),
            "down": (0, dist),
            "left": (-dist, 0),
            "right": (dist, 0),
        }
        dx, dy = offsets.get(a.direction or "down", (0, dist))
        self._device.swipe(cx, cy, cx + dx, cy + dy, 500)

    def _type_text(self, a: UnifiedAction):
        """文字输入（含键盘管理）"""
        device = self._device

        # 1. 切换到 ADB Keyboard
        original_ime = device.switch_to_adb_keyboard()
        time.sleep(TIMING.action.keyboard_switch_delay)

        # 2. 清空已有文字
        device.clear_text()
        time.sleep(TIMING.action.text_clear_delay)

        # 3. 输入新文字
        device.type_text(a.text)
        time.sleep(TIMING.action.text_input_delay)

        # 4. 恢复原始键盘
        device.restore_keyboard(original_ime)
        time.sleep(TIMING.action.keyboard_restore_delay)

    def _back(self, _: UnifiedAction):
        self._device.back()

    def _home(self, _: UnifiedAction):
        self._device.home()

    def _launch(self, a: UnifiedAction):
        self._device.launch_app(a.text)

    def _wait(self, a: UnifiedAction):
        duration_s = (a.duration_ms or 2000) / 1000
        time.sleep(duration_s)

    def _take_over(self, a: UnifiedAction):
        """用户接管：暂停自动化，等待用户手动完成"""
        self._takeover_callback(a.text or "请完成需要手动操作的步骤")

    def _note(self, a: UnifiedAction):
        log_action.info("Note: %s", a.text)

    def _call_api(self, a: UnifiedAction):
        log_action.info("Call_API: %s (未实现)", a.text)

    # ── 动作分发表 ──

    _HANDLERS: dict[ActionType, Callable[[ActionExecutor, UnifiedAction], None]] = {
        ActionType.TAP: _tap,
        ActionType.DOUBLE_TAP: _double_tap,
        ActionType.LONG_PRESS: _long_press,
        ActionType.SWIPE: _swipe,
        ActionType.SCROLL: _scroll,
        ActionType.TYPE: _type_text,
        ActionType.BACK: _back,
        ActionType.HOME: _home,
        ActionType.LAUNCH: _launch,
        ActionType.WAIT: _wait,
        ActionType.TAKE_OVER: _take_over,
        ActionType.NOTE: _note,
        ActionType.CALL_API: _call_api,
    }

    # ── 默认回调 ──

    @staticmethod
    def _default_confirmation(message: str) -> bool:
        response = input("  确认操作: " + message + " (Y/n): ").strip().lower()
        return response != "n"

    @staticmethod
    def _default_takeover(message: str) -> None:
        input("  " + message + "\n  完成后按 Enter 继续...")