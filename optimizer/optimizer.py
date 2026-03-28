"""ActionOptimizer：根据累积的执行历史，用 LLM 改写当前操作指令"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config.settings import LLMConfig
from optimizer.prompts import OPTIMIZER_SYSTEM_PROMPT
from providers import create_provider

logger = logging.getLogger(__name__)


@dataclass
class StepRecord:
    """单步执行记录"""
    step_num: int
    instruction: str       # 原始操作指令
    page_description: str  # PageDescriber 对该步截图的描述


class ActionOptimizer:
    """根据累积的历史步骤（指令 + 页面描述），用 LLM 改写当前操作指令。"""

    def __init__(self, config: LLMConfig):
        self.provider = create_provider(
            provider=config.provider,
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url,
            temperature=0.1,
            max_tokens=3000,
        )
        self._history: list[StepRecord] = []
        self._step_counter: int = 0

    def optimize(self, instruction: str, current_page_description: str = "") -> str:
        """根据累积的历史上下文 + 当前页面描述，改写当前指令。无历史时原样返回。"""
        if not self._history:
            return instruction

        user_prompt = self._build_prompt(instruction, current_page_description)
        logger.debug("──── 🔧 指令优化器输入 ────\n%s", user_prompt)
        try:
            result = self.provider.chat(
                [{"role": "user", "content": user_prompt}],
                system_prompt=OPTIMIZER_SYSTEM_PROMPT,
            )
            optimized = result.strip()
            if optimized:
                logger.info("指令优化: %s → %s", instruction, optimized)
                return optimized
            return instruction
        except Exception as e:
            logger.warning("指令优化失败: %s，使用原始指令", e)
            return instruction

    def record(self, instruction: str, page_description: str):
        """记录已完成步骤的指令和页面描述"""
        self._step_counter += 1
        self._history.append(StepRecord(
            step_num=self._step_counter,
            instruction=instruction,
            page_description=page_description,
        ))

    def reset(self):
        """清空历史（切换 TestCase 时调用）"""
        self._history.clear()
        self._step_counter = 0
        logger.debug("ActionOptimizer 已重置")

    def _build_prompt(self, current_instruction: str, current_page_description: str = "") -> str:
        """构建带 XML 标签的用户消息"""
        parts = ["<已执行步骤>"]
        for record in self._history:
            parts.append(f"<步骤{record.step_num}>")
            parts.append(f"  <指令>{record.instruction}</指令>")
            if record.page_description:
                parts.append(f"  <页面描述>{record.page_description}</页面描述>")
            parts.append(f"</步骤{record.step_num}>")
        parts.append("</已执行步骤>")
        parts.append("")
        if current_page_description:
            parts.append(f"<当前页面描述>{current_page_description}</当前页面描述>")
            parts.append("")
        parts.append(f"<当前指令>{current_instruction}</当前指令>")
        parts.append("")
        parts.append("请改写当前指令，使其更加具体明确：")
        return "\n".join(parts)
