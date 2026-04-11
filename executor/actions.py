"""统一动作定义：ActionType, UnifiedAction, ModelOutput"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    """13 种动作类型 + 1 个结束标识"""

    # 核心 10 种
    TAP = "tap"
    DOUBLE_TAP = "double_tap"
    LONG_PRESS = "long_press"
    SWIPE = "swipe"
    SCROLL = "scroll"
    TYPE = "type"
    BACK = "back"
    HOME = "home"
    LAUNCH = "launch"
    WAIT = "wait"

    # 扩展 3 种
    TAKE_OVER = "take_over"     # 用户接管
    NOTE = "note"               # 记录内容（预留）
    CALL_API = "call_api"       # API 调用（预留）

    # 结束标识
    FINISH = "finish"


@dataclass
class UnifiedAction:
    """
    统一动作数据类。

    所有模型适配器的 parse() 输出此结构，ActionExecutor 消费此结构。
    坐标已从模型输出的相对坐标转换为绝对像素坐标。
    """

    type: ActionType
    x: int | None = None
    y: int | None = None
    end_x: int | None = None       # swipe 终点
    end_y: int | None = None       # swipe 终点
    text: str | None = None        # Type / Launch / Take_over / Note
    duration_ms: int | None = None  # Long Press / Swipe / Wait
    direction: str | None = None    # Scroll 方向
    thinking: str = ""
    raw_response: str = ""

    @property
    def is_finish(self) -> bool:
        return self.type == ActionType.FINISH


@dataclass
class ModelOutput:
    """模型调用的原始输出"""

    thinking: str
    action_text: str
    raw_content: str
    time_to_first_token: float | None = None
    total_time: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0