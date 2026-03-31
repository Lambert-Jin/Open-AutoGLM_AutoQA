"""YAML 测试用例解析器"""

from __future__ import annotations

import os

import yaml

from config.loader import _config_from_dict, _map_device_type
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
    model_config = _config_from_dict(
        ActionModelConfig, config_raw.get("action_model", {}), defaults=global_model,
    )

    # vlm: YAML > 全局配置
    vlm_config = _config_from_dict(
        VLMConfig, config_raw.get("vlm", {}), defaults=global_vlm,
    )

    # llm: YAML > 全局配置
    llm_config = _config_from_dict(
        LLMConfig, config_raw.get("llm", {}), defaults=global_llm,
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


