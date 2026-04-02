"""动作截图归档"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from device.base import DeviceScreenshot
from monitor.annotator import ScreenshotAnnotator
from monitor.models import ActionTraceContext, ArtifactRef, OverlayAction
from screenshot import ScreenshotManager


class ArtifactRecorder:
    """按 action 维度保存前后截图与元数据"""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._annotator = ScreenshotAnnotator()
        self._screenshot_mgr = ScreenshotManager()
        self._states: dict[str, dict] = {}
        self._lock = threading.Lock()

    def record_before(
        self,
        trace: ActionTraceContext,
        overlay: OverlayAction,
        screenshot: DeviceScreenshot,
    ) -> ArtifactRef:
        with self._lock:
            state = self._ensure_state(trace, overlay)
            artifact: ArtifactRef = state["artifact"]
            action_dir: Path = state["action_dir"]

            artifact.before_raw = action_dir / "before.png"
            artifact.before_marked = action_dir / "before_marked.png"
            self._screenshot_mgr.save(screenshot, str(artifact.before_raw))
            self._annotator.save(screenshot, overlay, "Before", str(artifact.before_marked))

            state["meta"]["before"] = {
                "timestamp": screenshot.timestamp,
                "screenshot_id": screenshot.id,
                "is_sensitive": screenshot.is_sensitive,
            }
            self._write_meta(state)
            return artifact

    def record_after(
        self,
        trace: ActionTraceContext,
        overlay: OverlayAction,
        screenshot: DeviceScreenshot | None,
        *,
        success: bool,
        message: str | None = None,
    ) -> ArtifactRef:
        with self._lock:
            state = self._ensure_state(trace, overlay)
            artifact: ArtifactRef = state["artifact"]
            action_dir: Path = state["action_dir"]

            if screenshot is not None:
                artifact.after_raw = action_dir / "after.png"
                artifact.after_marked = action_dir / "after_marked.png"
                self._screenshot_mgr.save(screenshot, str(artifact.after_raw))
                self._annotator.save(screenshot, overlay, "After", str(artifact.after_marked))
                state["meta"]["after"] = {
                    "timestamp": screenshot.timestamp,
                    "screenshot_id": screenshot.id,
                    "is_sensitive": screenshot.is_sensitive,
                }

            state["meta"]["result"] = {
                "success": success,
                "message": message,
            }
            self._write_meta(state)
            return artifact

    def _ensure_state(self, trace: ActionTraceContext, overlay: OverlayAction) -> dict:
        state = self._states.get(trace.action_id)
        if state is not None:
            return state

        action_dir = (
            self.root_dir
            / trace.run_id
            / trace.case_dir_name
            / trace.step_dir_name
            / trace.action_dir_name
        )
        action_dir.mkdir(parents=True, exist_ok=True)

        artifact = ArtifactRef(meta_json=action_dir / "action.json")
        state = {
            "action_dir": action_dir,
            "artifact": artifact,
            "meta": {
                "context": {
                    "run_id": trace.run_id,
                    "case_index": trace.case_index,
                    "step_index": trace.step_index,
                    "round_index": trace.round_index,
                    "step_description": trace.step_description,
                    "action_id": trace.action_id,
                    "from_cache": trace.from_cache,
                },
                "overlay": overlay.to_dict(),
            },
        }
        self._states[trace.action_id] = state
        self._write_meta(state)
        return state

    def _write_meta(self, state: dict) -> None:
        artifact: ArtifactRef = state["artifact"]
        payload = dict(state["meta"])
        payload["artifacts"] = artifact.to_dict(self.root_dir)
        with open(artifact.meta_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
