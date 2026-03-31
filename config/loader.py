"""全局配置文件加载器"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

from config.settings import (
    ActionModelConfig,
    CacheConfig,
    DeviceConfig,
    LLMConfig,
    VLMConfig,
)
from utils.text import resolve_dict as _resolve_dict

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


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

    raw = _resolve_dict(raw, strict=False)

    # device
    device_raw = raw.get("device", {})
    raw_type = device_raw.get("type", "adb")
    device_config = DeviceConfig(
        device_type=_map_device_type(raw_type),
        device_id=device_raw.get("id"),
    )

    config_raw = raw.get("config", {})

    # action_model
    action_model_config = _config_from_dict(
        ActionModelConfig, config_raw.get("action_model", {}),
    )

    # vlm
    vlm_config = _config_from_dict(
        VLMConfig, config_raw.get("vlm", {}),
    )

    # llm
    llm_config = _config_from_dict(
        LLMConfig, config_raw.get("llm", {}),
    )

    # cache
    cache_config = _config_from_dict(
        CacheConfig, config_raw.get("cache", {}),
    )

    return device_config, action_model_config, vlm_config, llm_config, cache_config


def _config_from_dict(cls, raw: dict, defaults=None):
    """从 dict 构造 dataclass 实例，仅取 dataclass 字段名对应的 key。

    Args:
        cls: dataclass 类
        raw: 原始 dict
        defaults: 可选的 fallback dataclass 实例（YAML > defaults > field default）
    """
    import dataclasses
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in raw:
            kwargs[f.name] = raw[f.name]
        elif defaults is not None:
            kwargs[f.name] = getattr(defaults, f.name)
    return cls(**kwargs)


def _map_device_type(device_type: str) -> str:
    """映射用户友好的设备类型到内部值"""
    mapping = {
        "android": "adb",
        "harmony": "hdc",
        "ios": "ios",
        "adb": "adb",
        "hdc": "hdc",
    }
    return mapping.get(device_type, "adb")