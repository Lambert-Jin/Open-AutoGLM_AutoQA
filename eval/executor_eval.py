"""ExecutorEval: 逐步评测 action 执行质量"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from eval.models import ActionStepData, _StepWithContext
from eval.prompts import EXECUTOR_EVAL_SYSTEM_PROMPT, EXECUTOR_EVAL_USER_TEMPLATE
from eval._utils import load_image
from providers import ModelProvider

logger = logging.getLogger(__name__)

ISSUE_TYPES = [
    "action_wrong", "action_missed", "state_incomplete",
    "popup_interference", "unrelated_change",
]


class ExecutorEval:
    def __init__(self, provider: ModelProvider):
        self.provider = provider

    def evaluate(self, action_steps: list[_StepWithContext]) -> dict[str, Any]:
        details = []
        for item in action_steps:
            step: ActionStepData = item.step
            eval_result = self._evaluate_step(step)
            details.append({
                "case_name": item.case_name,
                "step_index": step.step_index,
                "description": step.instruction.original,
                "actions_taken": step.result.get("actions_taken", []),
                "rounds": step.result.get("rounds", 0),
                **eval_result,
            })

        summary = self._compute_summary(details)
        return {"summary": summary, "details": details}

    def _evaluate_step(self, step: ActionStepData) -> dict:
        user_text = EXECUTOR_EVAL_USER_TEMPLATE.format(
            original_instruction=step.instruction.original,
            optimized_instruction=step.instruction.optimized,
            actions_taken=json.dumps(step.result.get("actions_taken", []), ensure_ascii=False),
            rounds=step.result.get("rounds", 0),
        )

        content: list[dict] = [{"type": "text", "text": user_text}]

        # 附加操作前后截图：第一轮 before + 最后一个有 after 截图的轮次
        if step.rounds:
            before_img = load_image(step.rounds[0].screenshot_before)
            if before_img:
                content.insert(0, before_img)

            # 找最后一个有 after 截图的轮次（finish 轮 after 为空）
            after_img = None
            for rd in reversed(step.rounds):
                if rd.screenshot_after:
                    after_img = load_image(rd.screenshot_after)
                    break
            if after_img:
                content.append(after_img)

        messages = [{"role": "user", "content": content}]

        raw = self.provider.chat(messages, system_prompt=EXECUTOR_EVAL_SYSTEM_PROMPT)
        return self._parse_response(raw)

    def _compute_summary(self, details: list[dict]) -> dict:
        total = len(details)
        action_correct = sum(1 for d in details if d.get("action_correct"))
        state_correct = sum(1 for d in details if d.get("state_correct"))

        total_rounds = sum(d.get("rounds", 0) for d in details)
        avg_rounds = total_rounds / total if total > 0 else 0

        issue_dist = {t: 0 for t in ISSUE_TYPES}
        for d in details:
            issue = d.get("issue_type")
            if issue and issue in issue_dist:
                issue_dist[issue] += 1

        return {
            "total_actions": total,
            "action_correct": action_correct,
            "state_correct": state_correct,
            "avg_rounds": round(avg_rounds, 1),
            "issue_distribution": issue_dist,
        }

    def _parse_response(self, raw: str) -> dict:
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("ExecutorEval VLM 响应解析失败: %s", raw[:200])
            return {
                "action_correct": False, "state_correct": False,
                "issue_type": None, "issue_detail": f"解析失败: {raw[:100]}",
                "efficiency_note": "",
            }
