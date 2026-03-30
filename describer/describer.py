"""PageDescriber：用 VLM 分析截图，提取页面关键信息供跨步骤传递"""

from __future__ import annotations

import logging

from config.settings import VLMConfig
from describer.prompts import DESCRIBE_SYSTEM_PROMPT
from device.base import DeviceScreenshot
from providers import create_provider
from providers._utils import guess_mime_type

logger = logging.getLogger(__name__)


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
        logger
        return raw.strip()
