"""AsserterEval: 独立验证断言结果的置信度"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from eval.models import AssertStepData, _StepWithContext
from eval.prompts import ASSERTER_EVAL_SYSTEM_PROMPT, ASSERTER_EVAL_USER_TEMPLATE
from eval._utils import load_image
from providers import ModelProvider

logger = logging.getLogger(__name__)

AMBIGUOUS_THRESHOLD = 0.6


class AsserterEval:
    def __init__(self, provider: ModelProvider):
        self.provider = provider

    def evaluate(self, assert_steps: list[_StepWithContext]) -> dict[str, Any]:
        details = []
        for item in assert_steps:
            step: AssertStepData = item.step
            independent = self._verify_independently(step)
            verdict = self._determine_verdict(step.result, independent)
            details.append({
                "case_name": item.case_name,
                "step_index": step.step_index,
                "expectation": step.expectation,
                "original": step.result,
                "independent": independent,
                "verdict": verdict,
                "risk_factors": independent.get("risk_factors", []),
            })

        summary = self._compute_summary(details)
        return {"summary": summary, "details": details}

    def _verify_independently(self, step: AssertStepData) -> dict:
        messages = [{"role": "user", "content": [
            {"type": "text", "text": ASSERTER_EVAL_USER_TEMPLATE.format(
                expectation=step.expectation,
            )},
        ]}]

        # 如果有截图路径，加载并附加图片
        if step.screenshot:
            img_block = load_image(step.screenshot)
            if img_block:
                messages[0]["content"].insert(0, img_block)

        raw = self.provider.chat(messages, system_prompt=ASSERTER_EVAL_SYSTEM_PROMPT)
        return self._parse_response(raw)

    def _determine_verdict(
        self, original: dict, independent: dict,
    ) -> str:
        if independent.get("confidence", 0) < AMBIGUOUS_THRESHOLD:
            return "ambiguous"

        orig_passed = original.get("passed", False)
        indep_passed = independent.get("passed", False)

        if orig_passed and indep_passed:
            return "confirmed_pass"
        if not orig_passed and not indep_passed:
            return "confirmed_fail"
        if orig_passed and not indep_passed:
            return "false_pass"
        return "false_fail"

    def _compute_summary(self, details: list[dict]) -> dict:
        counts = {
            "total_assertions": len(details),
            "confirmed_pass": 0,
            "confirmed_fail": 0,
            "false_pass": 0,
            "false_fail": 0,
            "ambiguous": 0,
        }
        for d in details:
            counts[d["verdict"]] += 1

        reliable = counts["confirmed_pass"] + counts["confirmed_fail"]
        total = counts["total_assertions"]
        counts["reliability_score"] = reliable / total if total > 0 else 0.0
        return counts

    def _parse_response(self, raw: str) -> dict:
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("评测 VLM 响应解析失败: %s", raw[:200])
            return {"passed": False, "confidence": 0.0, "reason": f"解析失败: {raw[:100]}", "risk_factors": []}
