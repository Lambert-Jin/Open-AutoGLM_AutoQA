"""公共文本工具函数"""
from __future__ import annotations

import os
import re


def resolve_env_vars(value: str, strict: bool = True) -> str:
    """将 ${VAR} 替换为环境变量值。

    Args:
        strict: True 时缺失变量抛 ValueError，False 时保留原文
    """
    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        env_val = os.environ.get(var_name)
        if env_val is None:
            if strict:
                raise ValueError(f"Environment variable '{var_name}' is not set")
            return match.group(0)
        return env_val

    return re.sub(r"\$\{(\w+)\}", _replace, value)


def resolve_dict(data: dict, strict: bool = True) -> dict:
    """递归解析字典中所有字符串的环境变量"""
    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = resolve_env_vars(value, strict=strict)
        elif isinstance(value, dict):
            result[key] = resolve_dict(value, strict=strict)
        elif isinstance(value, list):
            result[key] = [
                resolve_dict(v, strict=strict) if isinstance(v, dict) else v
                for v in value
            ]
        else:
            result[key] = value
    return result


def strip_markdown_json(text: str) -> str:
    """去除 LLM 响应中的 markdown 代码块包裹，提取纯 JSON 文本。

    支持: ```json ... ```, ``` ... ```, 以及无包裹的纯文本。
    """
    text = text.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text
