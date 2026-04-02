"""ActionOptimizer：根据对话历史，用 LLM 改写当前操作指令"""

from __future__ import annotations

import logging

from config.settings import LLMConfig
from optimizer.prompts import OPTIMIZER_SYSTEM_PROMPT
from providers import create_provider

log_optimizer = logging.getLogger("autoqa:optimizer")


class ActionOptimizer:
    """根据 AutoGLM 的对话历史，用 LLM 改写当前操作指令使其更具体。"""

    def __init__(self, config: LLMConfig):
        self.provider = create_provider(
            provider=config.provider,
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url,
            temperature=0.1,
            max_tokens=3000,
        )

    def optimize(self, instruction: str, conversation_history: list[dict]) -> str:
        """
        根据 AutoGLM 的对话历史改写当前指令。

        Args:
            instruction: 当前步骤的原始指令
            conversation_history: 之前步骤的 AutoGLM 对话历史
                                  (包含 system/user/assistant 消息)
        """
        if not conversation_history:
            return instruction

        user_prompt = self._build_prompt(instruction, conversation_history)
        try:
            result = self.provider.chat(
                [{"role": "user", "content": user_prompt}],
                system_prompt=OPTIMIZER_SYSTEM_PROMPT,
            )
            optimized = result.strip()
            if optimized:
                log_optimizer.info("优化: %s → %s", instruction, optimized)
                return optimized
            return instruction
        except Exception as e:
            log_optimizer.warning("优化失败: %s，使用原始指令", e)
            return instruction

    @staticmethod
    def _build_prompt(current_instruction: str, conversation_history: list[dict]) -> str:
        """将对话历史转为文本摘要，构建优化提示"""
        parts = ["<历史对话摘要>"]
        for msg in conversation_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            # 跳过 system prompt（太长且对优化器无用）
            if role == "system":
                continue
            # 提取文本内容，跳过图片
            if isinstance(content, list):
                texts = [p.get("text", "") for p in content if p.get("type") == "text"]
                text = "\n".join(t for t in texts if t)
                if not text:
                    continue
            else:
                text = str(content)
            parts.append(f"<{role}>{text}</{role}>")
        parts.append("</历史对话摘要>")
        parts.append("")
        parts.append(f"<当前指令>{current_instruction}</当前指令>")
        parts.append("")
        parts.append("请改写当前指令，使其更加具体明确：")
        return "\n".join(parts)
