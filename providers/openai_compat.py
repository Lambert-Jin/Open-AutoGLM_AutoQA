"""OpenAI 兼容 API Provider（Qwen、GPT 等通用）"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class OpenAICompatProvider:
    """OpenAI 兼容 API Provider，支持流式和非流式调用，内置重试"""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "",
        temperature: float = 0.1,
        max_tokens: int = 2000,
        timeout: int = 60,
        max_retries: int = 2,
        retry_delay: float = 2.0,
    ):
        from openai import OpenAI

        self.client = OpenAI(
            base_url=base_url or None,
            api_key=api_key,
            timeout=timeout,
        )
        self.model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def chat(
        self,
        messages: list[dict],
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> str:
        """发送消息，返回模型原始文本响应。内置重试。"""
        # system_prompt 非空时插入到 messages 头部
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.default_temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
            "stream": stream,
        }

        for attempt in range(self.max_retries + 1):
            try:
                if stream:
                    return self._call_stream(params)
                else:
                    return self._call_sync(params)
            except Exception as e:
                if attempt < self.max_retries:
                    wait = self.retry_delay * (attempt + 1)
                    logger.warning(
                        "Provider 调用失败 (attempt %d/%d), %.1fs 后重试: %s",
                        attempt + 1,
                        self.max_retries + 1,
                        wait,
                        e,
                    )
                    time.sleep(wait)
                else:
                    raise

    def _call_sync(self, params: dict) -> str:
        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content or ""

    def _call_stream(self, params: dict) -> str:
        response = self.client.chat.completions.create(**params)
        content = ""
        for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                content += delta
        return content
