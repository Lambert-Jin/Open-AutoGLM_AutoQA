"""通用 OpenAI 兼容客户端，所有模型适配器共用"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from openai import OpenAI

from executor.actions import ModelOutput

logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    """客户端配置"""

    base_url: str
    api_key: str
    model: str
    max_tokens: int = 3000
    temperature: float = 0.0
    top_p: float = 0.85
    frequency_penalty: float = 0.2
    timeout: int = 60
    max_retries: int = 2
    retry_delay: float = 2.0
    extra_body: dict = field(default_factory=dict)  # 模型专属参数


class BaseModelClient:
    """
    通用 OpenAI 兼容客户端。

    改进点（对比 phone_agent.ModelClient）：
    - 不 print 任何内容，全部走 logging
    - 支持流式/非流式切换
    - 内置重试（429、超时）
    - 支持 extra_body 传递模型专属参数
    """

    def __init__(self, config: ClientConfig):
        self._config = config
        self._client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
        )

    def request(
        self,
        messages: list[dict],
        stream: bool = True,
    ) -> ModelOutput:
        """调用模型，返回 ModelOutput。内置重试。"""
        for attempt in range(self._config.max_retries + 1):
            try:
                if stream:
                    return self._request_stream(messages)
                else:
                    return self._request_sync(messages)
            except Exception as e:
                if attempt < self._config.max_retries:
                    wait = self._config.retry_delay * (attempt + 1)
                    logger.warning(
                        "模型调用失败 (attempt %d/%d), %.1fs 后重试: %s",
                        attempt + 1,
                        self._config.max_retries + 1,
                        wait,
                        e,
                    )
                    time.sleep(wait)
                else:
                    raise

    def _build_params(self, messages: list[dict], stream: bool) -> dict:
        """构建 API 调用参数"""
        params = {
            "model": self._config.model,
            "messages": messages,
            "max_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
            "frequency_penalty": self._config.frequency_penalty,
            "stream": stream,
        }
        if self._config.extra_body:
            params["extra_body"] = self._config.extra_body
        return params

    def _request_stream(self, messages: list[dict]) -> ModelOutput:
        start = time.time()
        first_token_time = None

        response = self._client.chat.completions.create(
            **self._build_params(messages, stream=True)
        )

        content = ""
        for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                if first_token_time is None:
                    first_token_time = time.time() - start
                content += delta
                # logger.debug("stream token: %s", delta)

        return ModelOutput(
            thinking="",
            action_text="",
            raw_content=content,
            time_to_first_token=first_token_time,
            total_time=time.time() - start,
        )

    def _request_sync(self, messages: list[dict]) -> ModelOutput:
        start = time.time()

        response = self._client.chat.completions.create(
            **self._build_params(messages, stream=False)
        )

        content = response.choices[0].message.content or ""
        return ModelOutput(
            thinking="",
            action_text="",
            raw_content=content,
            total_time=time.time() - start,
        )