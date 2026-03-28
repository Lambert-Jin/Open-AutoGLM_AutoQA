"""全局配置文件加载器"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import yaml

from config.settings import (
    ActionModelConfig,
    CacheConfig,
    DeviceConfig,
    LLMConfig,
    VLMConfig,
)

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _resolve_env_vars(value: str) -> str:
    """将 ${VAR} 替换为环境变量值，缺失时保留原文"""
    def _replace(match):
        return os.environ.get(match.group(1), match.group(0))
    return re.sub(r"\$\{(\w+)\}", _replace, value)


def _resolve_dict(data: dict) -> dict:
    """递归解析字典中所有字符串的环境变量"""
    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = _resolve_env_vars(value)
        elif isinstance(value, dict):
            result[key] = _resolve_dict(value)
        elif isinstance(value, list):
            result[key] = [_resolve_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            result[key] = value
    return result


def load_global_config(
    config_path: str | Path | None = None,
) -> tuple[DeviceConfig, ActionModelConfig, VLMConfig, LLMConfig, CacheConfig]:
    """
    加载全局配置文件。

    查找顺序：config_path 参数 > auto_qa/config.yaml
    文件不存在时返回代码默认值。

    Returns:
        (DeviceConfig, ActionModelConfig, VLMConfig, LLMConfig, CacheConfig)
    """
    path = Path(config_path) if config_path else _CONFIG_PATH

    if not path.exists():
        logger.debug("全局配置文件不存在: %s，使用代码默认值", path)
        return DeviceConfig(), ActionModelConfig(), VLMConfig(), LLMConfig(), CacheConfig()

    logger.info("加载全局配置: %s", path)

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    raw = _resolve_dict(raw)

    # device
    device_raw = raw.get("device", {})
    raw_type = device_raw.get("type", "adb")
    device_type_map = {"android": "adb", "harmony": "hdc", "ios": "ios"}
    device_config = DeviceConfig(
        device_type=device_type_map.get(raw_type, raw_type),
        device_id=device_raw.get("id"),
    )

    config_raw = raw.get("config", {})

    # action_model
    am = config_raw.get("action_model", {})
    action_model_config = ActionModelConfig(
        provider=am.get("provider", "autoglm"),
        base_url=am.get("base_url", "${AUTOGLM_BASE_URL}"),
        api_key=am.get("api_key", "${AUTOGLM_API_KEY}"),
        model=am.get("model", "autoglm-phone"),
        max_tokens=am.get("max_tokens", 3000),
        temperature=am.get("temperature", 0.1),
        lang=am.get("lang", "cn"),
        custom_rules=am.get("custom_rules", []),
    )

    # vlm
    vm = config_raw.get("vlm", {})
    vlm_config = VLMConfig(
        provider=vm.get("provider", "gemini"),
        base_url=vm.get("base_url", ""),
        api_key=vm.get("api_key", "${GEMINI_API_KEY}"),
        model=vm.get("model", "gemini-3-pro-image-preview"),
        temperature=vm.get("temperature", 0.1),
        max_tokens=vm.get("max_tokens", 1000),
    )

    # llm（Planner + Optimizer 共用）
    lm = config_raw.get("llm", {})
    llm_config = LLMConfig(
        provider=lm.get("provider", "gemini"),
        base_url=lm.get("base_url", ""),
        api_key=lm.get("api_key", "${GEMINI_API_KEY}"),
        model=lm.get("model", "gemini-3.1-pro-preview"),
        temperature=lm.get("temperature", 0.3),
        max_tokens=lm.get("max_tokens", 2000),
    )

    # cache
    ca = config_raw.get("cache", {})
    cache_config = CacheConfig(
        enabled=ca.get("enabled", True),
        similarity_threshold=ca.get("similarity_threshold", 0.85),
        region_similarity_threshold=ca.get("region_similarity_threshold", 0.8),
        ttl_days=ca.get("ttl_days", 30),
        db_path=ca.get("db_path", ".cache/action_cache.db"),
    )

    return device_config, action_model_config, vlm_config, llm_config, cache_config