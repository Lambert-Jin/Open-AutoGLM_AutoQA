"""Gemini Provider：通过 Google GenAI SDK 调用"""

from __future__ import annotations

import base64
import logging

from providers._utils import guess_mime_type

logger = logging.getLogger(__name__)


class GeminiProvider:
    """Google Gemini Provider，支持纯文本和图文混合输入"""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ):
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens

    def chat(
        self,
        messages: list[dict],
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> str:
        """
        发送消息，返回模型原始文本响应。

        messages 使用 OpenAI 兼容格式，内部转换为 Gemini SDK 格式。
        """
        from google.genai import types

        # 从 messages 中提取 system prompt（如果有）
        system = system_prompt
        contents = []

        for msg in messages:
            role = msg["role"]

            if role == "system":
                # system message 合并到 system_prompt
                if not system:
                    system = msg["content"] if isinstance(msg["content"], str) else ""
                continue

            # 转换 content 为 Gemini Parts
            parts = self._convert_content(msg["content"])

            # Gemini 只支持 "user" 和 "model" 角色
            gemini_role = "model" if role == "assistant" else "user"
            contents.append(types.Content(role=gemini_role, parts=parts))

        config = types.GenerateContentConfig(
            temperature=temperature if temperature is not None else self.default_temperature,
            max_output_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
        )
        if system:
            config.system_instruction = system

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        return response.text

    def _convert_content(self, content) -> list:
        """将 OpenAI 格式的 content 转为 Gemini Parts 列表"""
        from google.genai import types

        if isinstance(content, str):
            return [types.Part.from_text(text=content)]

        # content 是列表（图文混合）
        parts = []
        for item in content:
            if item["type"] == "text":
                parts.append(types.Part.from_text(text=item["text"]))
            elif item["type"] == "image_url":
                url = item["image_url"]["url"]
                if url.startswith("data:"):
                    # data:image/png;base64,xxx
                    header, b64_data = url.split(",", 1)
                    mime = header.split(":")[1].split(";")[0]
                    image_bytes = base64.b64decode(b64_data)
                    parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime))
                else:
                    # 直接 URL（Gemini 也支持）
                    parts.append(types.Part.from_uri(file_uri=url, mime_type="image/png"))
        return parts
