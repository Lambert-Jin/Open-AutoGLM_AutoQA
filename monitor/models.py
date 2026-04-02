"""监控模块数据模型"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from executor.actions import UnifiedAction


@dataclass(slots=True)
class MonitorConfig:
    """本地监控页配置"""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8765
    artifact_dir: str = ".artifacts/monitor"
    scrcpy_path: str | None = None
    scrcpy_max_fps: int = 15
    scrcpy_video_bit_rate: str = "6M"
    scrcpy_max_size: int = 1080
    scrcpy_record_format: str = "mkv"
    scrcpy_poll_interval: float = 0.4
    event_buffer_size: int = 1000
    mask_input_text: bool = True

    @property
    def artifact_root(self) -> Path:
        return Path(self.artifact_dir)


@dataclass(frozen=True, slots=True)
class ActionTraceContext:
    """一次真实动作的上下文"""

    run_id: str
    case_index: int
    step_index: int
    round_index: int
    step_description: str
    action_id: str
    from_cache: bool = False

    @property
    def case_dir_name(self) -> str:
        return f"case_{self.case_index:02d}"

    @property
    def step_dir_name(self) -> str:
        return f"step_{self.step_index:02d}"

    @property
    def action_dir_name(self) -> str:
        return f"action_{self.round_index:02d}_{self.action_id}"


def _norm(value: int | None, total: int) -> float | None:
    if value is None or total <= 0:
        return None
    clamped = max(0, min(value, total))
    return clamped / total


def _action_label(action_type: str, text: str | None = None, direction: str | None = None) -> str:
    labels = {
        "tap": "点击",
        "double_tap": "双击",
        "long_press": "长按",
        "swipe": "滑动",
        "scroll": f"滚动 {direction or ''}".strip(),
        "type": "输入文本",
        "back": "返回",
        "home": "回到桌面",
        "launch": "启动应用",
        "wait": "等待",
        "take_over": "人工接管",
        "note": "记录",
        "call_api": "调用 API",
        "finish": "结束",
    }
    label = labels.get(action_type, action_type)
    if action_type == "launch" and text:
        return f"{label}: {text}"
    return label


def mask_text(text: str | None) -> str | None:
    if not text:
        return text
    if len(text) <= 2:
        return "*" * len(text)
    return text[0] + "*" * (len(text) - 2) + text[-1]


@dataclass(slots=True)
class OverlayAction:
    """前端与截图标注复用的动作描述"""

    action_id: str
    type: str
    label: str
    x_norm: float | None = None
    y_norm: float | None = None
    end_x_norm: float | None = None
    end_y_norm: float | None = None
    text: str | None = None
    duration_ms: int | None = None
    direction: str | None = None
    source_width: int = 0
    source_height: int = 0

    @classmethod
    def from_unified_action(
        cls,
        action: UnifiedAction,
        screen_width: int,
        screen_height: int,
        action_id: str,
        *,
        mask_input: bool = True,
    ) -> "OverlayAction":
        text = action.text
        if mask_input and action.type.value == "type":
            text = mask_text(text)
        return cls(
            action_id=action_id,
            type=action.type.value,
            label=_action_label(action.type.value, text=text, direction=action.direction),
            x_norm=_norm(action.x, screen_width),
            y_norm=_norm(action.y, screen_height),
            end_x_norm=_norm(action.end_x, screen_width),
            end_y_norm=_norm(action.end_y, screen_height),
            text=text,
            duration_ms=action.duration_ms,
            direction=action.direction,
            source_width=screen_width,
            source_height=screen_height,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ArtifactRef:
    """动作前后截图产物"""

    before_raw: Path | None = None
    before_marked: Path | None = None
    after_raw: Path | None = None
    after_marked: Path | None = None
    meta_json: Path | None = None

    def to_dict(self, root_dir: Path | None = None) -> dict[str, str | None]:
        def _convert(path: Path | None) -> str | None:
            if path is None:
                return None
            if root_dir is not None and path.is_relative_to(root_dir):
                rel = path.relative_to(root_dir).as_posix()
                return f"/artifacts/{rel}"
            return str(path)

        return {
            "before_raw": _convert(self.before_raw),
            "before_marked": _convert(self.before_marked),
            "after_raw": _convert(self.after_raw),
            "after_marked": _convert(self.after_marked),
            "meta_json": _convert(self.meta_json),
        }


@dataclass(slots=True)
class MonitorEvent:
    """监控页使用的事件对象"""

    event_type: str
    run_id: str
    timestamp: float
    case_index: int = 0
    step_index: int = 0
    round_index: int = 0
    status: str | None = None
    message: str | None = None
    step_title: str | None = None
    step_type: str | None = None
    action: OverlayAction | None = None
    artifacts: ArtifactRef | None = None

    def to_dict(self, root_dir: Path | None = None) -> dict[str, Any]:
        data: dict[str, Any] = {
            "event_type": self.event_type,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "case_index": self.case_index,
            "step_index": self.step_index,
            "round_index": self.round_index,
            "status": self.status,
            "message": self.message,
            "step_title": self.step_title,
            "step_type": self.step_type,
            "action": self.action.to_dict() if self.action else None,
            "artifacts": self.artifacts.to_dict(root_dir) if self.artifacts else None,
        }
        return data
