"""YAML 测试用例解析器"""

from __future__ import annotations

import os

import yaml

from config.settings import ActionModelConfig, DeviceConfig, LLMConfig, VLMConfig
from suite import ActionStep, AssertStep, Step, TestCase, TestSuite
from utils.text import resolve_dict as _resolve_dict


def parse_yaml(path: str) -> tuple[TestSuite, DeviceConfig, ActionModelConfig, VLMConfig, LLMConfig]:
    """
    解析 YAML 测试用例文件。

    优先级：YAML > 全局配置 > 代码默认值

    Returns:
        (TestSuite, DeviceConfig, ActionModelConfig, VLMConfig, LLMConfig)
    """
    from config.loader import load_global_config
    global_device, global_model, global_vlm, global_llm, _ = load_global_config()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    raw = _resolve_dict(raw, strict=False)

    # device: YAML > 全局配置
    device_raw = raw.get("device", {})
    device_config = DeviceConfig(
        device_type=_map_device_type(device_raw.get("type")) if device_raw.get("type") else global_device.device_type,
        device_id=device_raw.get("id", global_device.device_id),
    )

    # config
    config_raw = raw.get("config", {})

    # action_model: YAML > 全局配置
    model_raw = config_raw.get("action_model", {})
    model_config = ActionModelConfig(
        provider=model_raw.get("provider", global_model.provider),
        base_url=model_raw.get("base_url", global_model.base_url),
        api_key=model_raw.get("api_key", global_model.api_key),
        model=model_raw.get("model", global_model.model),
        max_tokens=model_raw.get("max_tokens", global_model.max_tokens),
        temperature=model_raw.get("temperature", global_model.temperature),
        lang=model_raw.get("lang", global_model.lang),
        custom_rules=model_raw.get("custom_rules", global_model.custom_rules),
    )

    # vlm: YAML > 全局配置
    vlm_raw = config_raw.get("vlm", {})
    vlm_config = VLMConfig(
        provider=vlm_raw.get("provider", global_vlm.provider),
        base_url=vlm_raw.get("base_url", global_vlm.base_url),
        api_key=vlm_raw.get("api_key", global_vlm.api_key),
        model=vlm_raw.get("model", global_vlm.model),
        temperature=vlm_raw.get("temperature", global_vlm.temperature),
        max_tokens=vlm_raw.get("max_tokens", global_vlm.max_tokens),
    )

    # llm: YAML > 全局配置
    llm_raw = config_raw.get("llm", {})
    llm_config = LLMConfig(
        provider=llm_raw.get("provider", global_llm.provider),
        base_url=llm_raw.get("base_url", global_llm.base_url),
        api_key=llm_raw.get("api_key", global_llm.api_key),
        model=llm_raw.get("model", global_llm.model),
        temperature=llm_raw.get("temperature", global_llm.temperature),
        max_tokens=llm_raw.get("max_tokens", global_llm.max_tokens),
    )

    # tasks
    test_cases = [_parse_test_case(t) for t in raw.get("tasks", [])]
    suite = TestSuite(name=raw.get("name", os.path.basename(path)), test_cases=test_cases)

    return suite, device_config, model_config, vlm_config, llm_config


def _parse_test_case(raw: dict) -> TestCase:
    """解析单个 TestCase"""
    steps: list[Step] = []
    for step_raw in raw.get("flow", []):
        if "action" in step_raw:
            steps.append(ActionStep(
                description=step_raw["action"],
                timeout=step_raw.get("timeout", 30),
                cache_key=step_raw.get("cache_key", ""),
            ))
        elif "assert" in step_raw:
            steps.append(AssertStep(
                expectation=step_raw["assert"],
                severity=step_raw.get("severity", "critical"),
                retry_on_fail=step_raw.get("retryOnFail", False),
                retry_cleanup=step_raw.get("retryCleanup", "关闭当前弹窗或广告"),
            ))

    return TestCase(
        name=raw.get("name", "unnamed"),
        steps=steps,
        continue_on_error=raw.get("continueOnError", False),
        description=raw.get("description", ""),
    )


def _map_device_type(device_type: str) -> str:
    """映射 YAML 中的设备类型到 DeviceConfig 的值"""
    mapping = {
        "android": "adb",
        "harmony": "hdc",
        "ios": "ios",
        "adb": "adb",
        "hdc": "hdc",
    }
    return mapping.get(device_type, "adb")