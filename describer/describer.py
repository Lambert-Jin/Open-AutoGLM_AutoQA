"""PageDescriber：用 VLM 分析截图，提取页面关键信息供跨步骤传递"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from config.settings import VLMConfig
from describer.prompts import DESCRIBE_SYSTEM_PROMPT, VERIFY_COMPLETION_PROMPT
from device.base import DeviceScreenshot
from providers import create_provider
from providers._utils import guess_mime_type

logger = logging.getLogger(__name__)


@dataclass
class VerifyResult:
    """VLM 完成度验证结果"""
    completed: bool
    reason: str
    confidence: float


class PageDescriber:
    """用 VLM 分析手机截图，返回页面关键信息摘要（纯文本）"""

    def __init__(self, vlm_config: VLMConfig):
        self.provider = create_provider(
            provider=vlm_config.provider,
            api_key=vlm_config.api_key,
            model=vlm_config.model,
            base_url=vlm_config.base_url,
            temperature=0.1,
            max_tokens=3000,
        )

    def describe(self, screenshot: DeviceScreenshot) -> str:
        """分析截图，返回页面关键信息摘要（纯文本，≤200字）"""
        mime = guess_mime_type(screenshot.base64_data)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{screenshot.base64_data}"}},
                {"type": "text", "text": "分析这个手机截图"},
            ],
        }]
        raw = self.provider.chat(messages, system_prompt=DESCRIBE_SYSTEM_PROMPT)
        return raw.strip()

    def verify_completion(
        self,
        before_screenshot: DeviceScreenshot,
        after_screenshot: DeviceScreenshot,
        action_description: str,
        history_text: str = "",
    ) -> VerifyResult:
        """判断操作是否已完成（前后截图对比 + 历史上下文）"""
        before_mime = guess_mime_type(before_screenshot.base64_data)
        after_mime = guess_mime_type(after_screenshot.base64_data)

        # 构建文本：历史 + 当前指令
        text_parts = []
        if history_text:
            text_parts.append(f"<历史操作>\n{history_text}\n</历史操作>\n")
        text_parts.append(f"<当前操作>{action_description}</当前操作>\n")
        text_parts.append("请判断当前操作是否已完成。")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{before_mime};base64,{before_screenshot.base64_data}"}},
                {"type": "image_url", "image_url": {"url": f"data:{after_mime};base64,{after_screenshot.base64_data}"}},
                {"type": "text", "text": "\n".join(text_parts)},
            ],
        }]
        raw = self.provider.chat(messages, system_prompt=VERIFY_COMPLETION_PROMPT)
        logger.debug("VLM 验证原始响应: %s", raw)
        return self._parse_verify_response(raw)

    def _parse_verify_response(self, raw: str) -> VerifyResult:
        """解析 VLM 验证响应为 VerifyResult"""
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            data = json.loads(cleaned)
            return VerifyResult(
                completed=bool(data.get("completed", False)),
                reason=str(data.get("reason", "")),
                confidence=float(data.get("confidence", 0.0)),
            )
        except (json.JSONDecodeError, ValueError):
            logger.warning("VLM 验证响应解析失败: %s", raw[:200])
            return VerifyResult(completed=False, reason="解析失败", confidence=0.0)
