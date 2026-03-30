"""统一模型 Provider 层：所有 LLM/VLM 调用的底层接口"""

from __future__ import annotations

from typing import Protocol


class ModelProvider(Protocol):
    """统一模型 Provider：所有 LLM/VLM 调用的底层接口"""

    def chat(
        self,
        messages: list[dict],
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> str:
        """发送消息，返回模型原始文本响应"""
        ...


def create_provider(
    provider: str,
    api_key: str,
    model: str,
    base_url: str = "",
    temperature: float = 0.1,
    max_tokens: int = 2000,
    **kwargs,
) -> ModelProvider:
    """工厂函数：根据 provider 名称创建实例"""
    if provider == "gemini":
        from providers.gemini import GeminiProvider
        return GeminiProvider(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider in ("qwen", "openai"):
        from providers.openai_compat import OpenAICompatProvider
        return OpenAICompatProvider(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown provider: {provider}, "
            f"supported: ['gemini', 'qwen', 'openai']"
        )
