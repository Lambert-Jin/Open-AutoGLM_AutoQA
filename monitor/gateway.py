"""监控网关协议"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from monitor.models import ActionTraceContext

if TYPE_CHECKING:
    from device.base import DeviceScreenshot
    from executor.actions import UnifiedAction


class MonitorGateway(Protocol):
    """执行链路与监控模块之间的协议"""

    url: str | None

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def on_run_start(self, run_id: str, suite_name: str) -> None: ...
    def on_run_end(self, run_id: str, success: bool) -> None: ...
    def on_case_start(self, run_id: str, case_index: int, case_name: str) -> None: ...
    def on_case_end(self, run_id: str, case_index: int, case_name: str, success: bool) -> None: ...
    def on_step_start(
        self,
        run_id: str,
        case_index: int,
        step_index: int,
        step_type: str,
        title: str,
    ) -> None: ...
    def on_step_end(
        self,
        run_id: str,
        case_index: int,
        step_index: int,
        step_type: str,
        title: str,
        success: bool,
        message: str | None = None,
    ) -> None: ...
    def on_action_before(
        self,
        trace: ActionTraceContext,
        action: UnifiedAction,
        screenshot: DeviceScreenshot,
    ) -> None: ...
    def on_action_after(
        self,
        trace: ActionTraceContext,
        action: UnifiedAction,
        screenshot: DeviceScreenshot | None,
        *,
        success: bool,
        message: str | None = None,
    ) -> None: ...


class NoopMonitorGateway:
    """默认空实现，避免调用方频繁判空"""

    url: str | None = None

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def on_run_start(self, run_id: str, suite_name: str) -> None:
        return None

    def on_run_end(self, run_id: str, success: bool) -> None:
        return None

    def on_case_start(self, run_id: str, case_index: int, case_name: str) -> None:
        return None

    def on_case_end(self, run_id: str, case_index: int, case_name: str, success: bool) -> None:
        return None

    def on_step_start(
        self,
        run_id: str,
        case_index: int,
        step_index: int,
        step_type: str,
        title: str,
    ) -> None:
        return None

    def on_step_end(
        self,
        run_id: str,
        case_index: int,
        step_index: int,
        step_type: str,
        title: str,
        success: bool,
        message: str | None = None,
    ) -> None:
        return None

    def on_action_before(self, trace: ActionTraceContext, action, screenshot) -> None:
        return None

    def on_action_after(
        self,
        trace: ActionTraceContext,
        action,
        screenshot,
        *,
        success: bool,
        message: str | None = None,
    ) -> None:
        return None
