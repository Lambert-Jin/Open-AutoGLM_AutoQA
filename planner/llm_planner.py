"""LLM 规划器：自然语言 → 测试步骤"""

from __future__ import annotations

import json
import logging

import yaml

from config.settings import LLMConfig
from planner.prompts import PLANNER_SYSTEM_PROMPT
from suite import ActionStep, AssertStep, Step, TestCase, TestSuite
from utils.text import strip_markdown_json

logger = logging.getLogger(__name__)


def plan_test_case(description: str, llm_config: LLMConfig) -> TestCase:
    """
    将自然语言描述转换为 TestCase。

    Args:
        description: 自然语言测试描述
        llm_config: LLM 配置

    Returns:
        解析后的 TestCase
    """
    response = _call_llm(PLANNER_SYSTEM_PROMPT, description, llm_config)
    return _parse_plan_response(response)


def generate_yaml_content(
    test_case: TestCase,
    suite_name: str = "",
    device_type: str = "android",
) -> str:
    """
    将 TestCase 转换为完整的 YAML 文件内容。

    使用环境变量占位符，用户可在生成后编辑配置。
    """
    suite_name = suite_name or test_case.name

    data = {
        "name": suite_name,
        "device": {"type": device_type},
        "config": {
            "action_model": {
                "provider": "autoglm",
                "base_url": "${AUTOGLM_BASE_URL}",
                "api_key": "${AUTOGLM_API_KEY}",
                "model": "autoglm-phone",
            },
            "vlm": {
                "provider": "gemini",
                "api_key": "${GEMINI_API_KEY}",
                "model": "gemini-2.5-flash",
            },
        },
        "tasks": [_test_case_to_dict(test_case)],
    }

    return yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)


def append_to_yaml(path: str, test_case: TestCase):
    """向已有 YAML 文件追加一个 task。

    Args:
        path: 已有 YAML 文件路径
        test_case: 要追加的 TestCase
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    if "tasks" not in data:
        data["tasks"] = []
    data["tasks"].append(_test_case_to_dict(test_case))

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def _test_case_to_dict(case: TestCase) -> dict:
    """将 TestCase 转为 YAML 兼容的字典"""
    flow = []
    for step in case.steps:
        if isinstance(step, ActionStep):
            item = {"action": step.description}
            if step.cache_key:
                item["cache_key"] = step.cache_key
            if step.timeout != 30:
                item["timeout"] = step.timeout
        elif isinstance(step, AssertStep):
            item = {"assert": step.expectation}
            if step.severity != "critical":
                item["severity"] = step.severity
            if step.retry_on_fail:
                item["retryOnFail"] = True
                item["retryCleanup"] = step.retry_cleanup
        else:
            continue
        flow.append(item)

    return {"name": case.name, "flow": flow}


def _call_llm(system_prompt: str, user_prompt: str, config: LLMConfig) -> str:
    """通过统一 Provider 层调用 LLM"""
    from providers import create_provider

    provider = create_provider(
        provider=config.provider,
        api_key=config.api_key,
        model=config.model,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    messages = [{"role": "user", "content": user_prompt}]
    return provider.chat(messages, system_prompt=system_prompt)


def _parse_plan_response(response: str) -> TestCase:
    """解析 LLM 返回的 JSON 为 TestCase"""
    text = strip_markdown_json(response)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("LLM 返回的 JSON 解析失败: %s\n原文: %s", e, response)
        raise ValueError(f"LLM 返回格式错误，无法解析为 JSON: {e}") from e

    name = data.get("name", "unnamed")
    steps: list[Step] = []

    for item in data.get("flow", []):
        if "action" in item:
            steps.append(ActionStep(
                description=item["action"],
                timeout=item.get("timeout", 30),
                cache_key=item.get("cache_key", ""),
            ))
        elif "assert" in item:
            steps.append(AssertStep(
                expectation=item["assert"],
                severity=item.get("severity", "critical"),
                retry_on_fail=item.get("retryOnFail", False),
                retry_cleanup=item.get("retryCleanup", "关闭当前弹窗或广告"),
            ))

    return TestCase(name=name, steps=steps, description="")