"""基于 VLM 的视觉断言引擎"""

from __future__ import annotations

import json
import logging
import re

from asserter.prompts import ASSERT_SYSTEM_PROMPT
from config.settings import AssertResult, Screenshot, VLMConfig
from providers import create_provider
from providers._utils import guess_mime_type

logger = logging.getLogger(__name__)


class Asserter:
    """基于 VLM 的视觉断言引擎，通过统一 Provider 层调用模型"""

    def __init__(self, vlm_config: VLMConfig):
        self.provider = create_provider(
            provider=vlm_config.provider,
            api_key=vlm_config.api_key,
            model=vlm_config.model,
            base_url=vlm_config.base_url,
            temperature=vlm_config.temperature,
            max_tokens=vlm_config.max_tokens,
        )

    def verify(self, screenshot: Screenshot, expectation: str) -> AssertResult:
        """对截图执行视觉断言，返回结构化结果"""
        mime = guess_mime_type(screenshot.base64)
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{screenshot.base64}"}},
            {"type": "text", "text": f"请判断以下期望是否成立：\n{expectation}"},
        ]}]

        raw = self.provider.chat(messages, system_prompt=ASSERT_SYSTEM_PROMPT)

        logger.debug("VLM 原始响应: %s", raw)
        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> AssertResult:
        """解析 VLM 响应为 AssertResult，容忍 markdown 代码块包裹"""
        # 去掉可能的 ```json ... ``` 包裹
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("VLM 响应 JSON 解析失败，原始内容: %s", raw)
            return AssertResult(
                passed=False,
                reason=f"VLM 响应解析失败: {raw[:200]}",
                confidence=0.0,
            )

        return AssertResult(
            passed=bool(data.get("passed", False)),
            reason=str(data.get("reason", "")),
            confidence=float(data.get("confidence", 0.0)),
        )
