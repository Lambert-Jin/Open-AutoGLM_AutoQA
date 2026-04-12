"""PlannerEval: LLM-as-Judge 评测规划质量"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from eval.models import (
    ActionStepData, AssertStepData, EvalCaseData,
)
from eval.prompts import PLANNER_EVAL_SYSTEM_PROMPT, PLANNER_EVAL_USER_TEMPLATE
from providers import ModelProvider

logger = logging.getLogger(__name__)

SCORE_DIMENSIONS = ["completeness", "correctness", "ordering", "assertion_quality", "granularity"]


class PlannerEval:
    def __init__(self, provider: ModelProvider):
        self.provider = provider

    def evaluate(
        self,
        cases: list[EvalCaseData],
        asserter_result: dict[str, Any],
        executor_result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        results = []
        for case in cases:
            if not case.description:
                continue
            eval_result = self._evaluate_case(case, asserter_result, executor_result)
            results.append(eval_result)
        return results

    def _evaluate_case(
        self,
        case: EvalCaseData,
        asserter_result: dict,
        executor_result: dict,
    ) -> dict:
        steps_text = self._format_steps(case.steps)
        diagnostics = self._format_diagnostics(case.case_name, asserter_result, executor_result)

        user_text = PLANNER_EVAL_USER_TEMPLATE.format(
            description=case.description,
            steps=steps_text,
            diagnostics=diagnostics,
        )

        messages = [{"role": "user", "content": user_text}]
        raw = self.provider.chat(messages, system_prompt=PLANNER_EVAL_SYSTEM_PROMPT)
        parsed = self._parse_response(raw)

        scores = {dim: parsed.get(dim, {"score": 0, "reason": ""}) for dim in SCORE_DIMENSIONS}
        score_values = [s["score"] for s in scores.values() if isinstance(s.get("score"), (int, float))]
        overall = sum(score_values) / len(score_values) if score_values else 0

        return {
            "case_name": case.case_name,
            "description": case.description,
            "scores": scores,
            "overall_score": round(overall, 1),
            "execution_correlated_issues": parsed.get("execution_correlated_issues", []),
        }

    def _format_steps(self, steps: list) -> str:
        lines = []
        for step in steps:
            if isinstance(step, ActionStepData):
                lines.append(f"  - action: {step.instruction.original}")
            elif isinstance(step, AssertStepData):
                lines.append(f"  - assert: {step.expectation}")
        return "\n".join(lines)

    def _format_diagnostics(
        self, case_name: str,
        asserter_result: dict, executor_result: dict,
    ) -> str:
        issues = []
        for d in asserter_result.get("details", []):
            if d.get("case_name") == case_name and d.get("verdict") in ("false_pass", "false_fail"):
                issues.append(f"断言步骤 {d['step_index']}: {d['verdict']} — {d.get('independent', {}).get('reason', '')}")

        for d in executor_result.get("details", []):
            if d.get("case_name") == case_name and d.get("issue_type"):
                issues.append(f"操作步骤 {d['step_index']}: {d['issue_type']} — {d.get('issue_detail', '')}")

        return "\n".join(issues) if issues else "无异常"

    def _parse_response(self, raw: str) -> dict:
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("PlannerEval LLM 响应解析失败: %s", raw[:200])
            return {}
