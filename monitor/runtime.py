"""监控运行时"""

from __future__ import annotations

import time
from pathlib import Path

from monitor.artifact_recorder import ArtifactRecorder
from monitor.event_bus import MonitorEventBus
from monitor.frame_hub import FrameHub
from monitor.gateway import NoopMonitorGateway
from monitor.models import ActionTraceContext, MonitorConfig, MonitorEvent, OverlayAction
from monitor.scrcpy_session import ScrcpySession
from monitor.web_server import MonitorWebServer

# import logging
# logger = logging.getLogger(__name__)


class MonitorRuntime(NoopMonitorGateway):
    """监控模块总入口"""

    def __init__(self, config: MonitorConfig, *, device_id: str | None):
        self._config = config
        self._device_id = device_id
        self._bus = MonitorEventBus(config.event_buffer_size)
        self._frame_hub = FrameHub()
        self._recorder = ArtifactRecorder(config.artifact_root)
        self._scrcpy_status = "idle"
        self._scrcpy_message: str | None = None
        self._web = MonitorWebServer(
            config,
            frame_hub=self._frame_hub,
            event_bus=self._bus,
            artifact_root=config.artifact_root,
            status_provider=self.status,
        )
        self._scrcpy = ScrcpySession(
            config,
            device_id=device_id,
            frame_hub=self._frame_hub,
            work_dir=config.artifact_root / "_live",
            status_callback=self._on_scrcpy_status,
        )
        self.url: str | None = None

    def start(self) -> None:
        self._web.start()
        self.url = self._web.url
        self._publish(
            MonitorEvent(
                event_type="monitor.ready",
                run_id="bootstrap",
                timestamp=time.time(),
                message=f"监控页已启动: {self.url}",
                status="ready",
            )
        )
        try:
            self._scrcpy.start()
        except Exception as e:
            # logger.warning("scrcpy 启动失败，将退化为动作截图刷新: %s", e)
            self._on_scrcpy_status("degraded", str(e))

    def stop(self) -> None:
        self._scrcpy.stop()
        self._web.stop()

    def status(self) -> dict:
        frame_id, _data, source = self._frame_hub.snapshot()
        return {
            "url": self.url,
            "scrcpy_status": self._scrcpy_status,
            "scrcpy_message": self._scrcpy_message,
            "frame_id": frame_id,
            "frame_source": source,
            "artifact_root": str(Path(self._config.artifact_root).resolve()),
        }

    def on_run_start(self, run_id: str, suite_name: str) -> None:
        self._publish(MonitorEvent("run.start", run_id, time.time(), message=suite_name, status="running"))

    def on_run_end(self, run_id: str, success: bool) -> None:
        status = "passed" if success else "failed"
        self._publish(MonitorEvent("run.end", run_id, time.time(), status=status))

    def on_case_start(self, run_id: str, case_index: int, case_name: str) -> None:
        self._publish(MonitorEvent("case.start", run_id, time.time(), case_index=case_index, message=case_name))

    def on_case_end(self, run_id: str, case_index: int, case_name: str, success: bool) -> None:
        status = "passed" if success else "failed"
        self._publish(
            MonitorEvent(
                "case.end",
                run_id,
                time.time(),
                case_index=case_index,
                message=case_name,
                status=status,
            )
        )

    def on_step_start(
        self,
        run_id: str,
        case_index: int,
        step_index: int,
        step_type: str,
        title: str,
    ) -> None:
        self._publish(
            MonitorEvent(
                "step.start",
                run_id,
                time.time(),
                case_index=case_index,
                step_index=step_index,
                step_title=title,
                step_type=step_type,
                status="running",
            )
        )

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
        status = "passed" if success else "failed"
        self._publish(
            MonitorEvent(
                "step.end",
                run_id,
                time.time(),
                case_index=case_index,
                step_index=step_index,
                step_title=title,
                step_type=step_type,
                status=status,
                message=message,
            )
        )

    def on_action_before(self, trace: ActionTraceContext, action, screenshot) -> None:
        overlay = OverlayAction.from_unified_action(
            action,
            screenshot.width,
            screenshot.height,
            trace.action_id,
            mask_input=self._config.mask_input_text,
        )
        artifacts = self._recorder.record_before(trace, overlay, screenshot)
        self._frame_hub.publish_screenshot(screenshot, source="device-before")
        self._publish(
            MonitorEvent(
                "action.before",
                trace.run_id,
                time.time(),
                case_index=trace.case_index,
                step_index=trace.step_index,
                round_index=trace.round_index,
                action=overlay,
                artifacts=artifacts,
                message=trace.step_description,
                status="from_cache" if trace.from_cache else "pending",
            )
        )

    def on_action_after(
        self,
        trace: ActionTraceContext,
        action,
        screenshot,
        *,
        success: bool,
        message: str | None = None,
    ) -> None:
        screen_width = screenshot.width if screenshot else 1
        screen_height = screenshot.height if screenshot else 1
        overlay = OverlayAction.from_unified_action(
            action,
            screen_width,
            screen_height,
            trace.action_id,
            mask_input=self._config.mask_input_text,
        )
        artifacts = self._recorder.record_after(
            trace,
            overlay,
            screenshot,
            success=success,
            message=message,
        )
        if screenshot is not None:
            self._frame_hub.publish_screenshot(screenshot, source="device-after")
        self._publish(
            MonitorEvent(
                "action.after",
                trace.run_id,
                time.time(),
                case_index=trace.case_index,
                step_index=trace.step_index,
                round_index=trace.round_index,
                action=overlay,
                artifacts=artifacts,
                status="success" if success else "failed",
                message=message,
            )
        )

    def _publish(self, event: MonitorEvent) -> None:
        self._bus.publish(event.to_dict(self._config.artifact_root))

    def _on_scrcpy_status(self, status: str, message: str | None) -> None:
        self._scrcpy_status = status
        self._scrcpy_message = message
        self._publish(
            MonitorEvent(
                "scrcpy.status",
                "bootstrap",
                time.time(),
                status=status,
                message=message,
            )
        )
