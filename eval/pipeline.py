"""EvalPipeline: 编排评测器执行"""
from __future__ import annotations

import logging
from typing import Any

from eval.asserter_eval import AsserterEval
from eval.executor_eval import ExecutorEval
from eval.models import EvalManifest
from eval.planner_eval import PlannerEval
from providers import ModelProvider

logger = logging.getLogger(__name__)


class EvalPipeline:
    def __init__(self, provider: ModelProvider):
        self.asserter_eval = AsserterEval(provider)
        self.executor_eval = ExecutorEval(provider)
        self.planner_eval = PlannerEval(provider)

    def run(self, manifest: EvalManifest) -> dict[str, Any]:
        logger.info("开始评测: %s (%s)", manifest.suite_name, manifest.run_id)

        # 1. AsserterEval + ExecutorEval（互不依赖）
        asserter_result = self.asserter_eval.evaluate(manifest.get_assert_steps())
        executor_result = self.executor_eval.evaluate(manifest.get_action_steps())

        # 2. PlannerEval（依赖前两者）
        planner_result = None
        if manifest.has_planner_cases():
            planner_result = self.planner_eval.evaluate(
                cases=manifest.cases,
                asserter_result=asserter_result,
                executor_result=executor_result,
            )

        # 3. 计算综合可靠性评分
        reliability = self._compute_reliability(
            asserter_result, executor_result, planner_result,
        )

        logger.info("评测完成: reliability_score=%.1f", reliability)

        return {
            "asserter_eval": asserter_result,
            "executor_eval": executor_result,
            "planner_eval": planner_result,
            "token_usage": {
                "autoglm": {"prompt": manifest.token_usage.autoglm.prompt, "completion": manifest.token_usage.autoglm.completion},
                "vlm": {"prompt": manifest.token_usage.vlm.prompt, "completion": manifest.token_usage.vlm.completion},
                "llm": {"prompt": manifest.token_usage.llm.prompt, "completion": manifest.token_usage.llm.completion},
            },
            "reliability_score": reliability,
        }

    def _compute_reliability(
        self,
        asserter_result: dict,
        executor_result: dict,
        planner_result: list | None,
    ) -> float:
        # 断言可靠性 (权重 0.5)
        a_summary = asserter_result.get("summary", {})
        a_total = a_summary.get("total_assertions", 0)
        a_reliable = a_summary.get("confirmed_pass", 0) + a_summary.get("confirmed_fail", 0)
        asserter_reliability = a_reliable / a_total if a_total > 0 else 1.0

        # false_pass 额外惩罚
        false_pass_count = a_summary.get("false_pass", 0)
        if false_pass_count > 0:
            asserter_reliability *= max(0.5, 1.0 - false_pass_count * 0.15)

        # 执行准确性 (权重 0.3)
        e_summary = executor_result.get("summary", {})
        e_total = e_summary.get("total_actions", 0)
        e_correct = min(e_summary.get("action_correct", 0), e_summary.get("state_correct", 0))
        executor_accuracy = e_correct / e_total if e_total > 0 else 1.0

        # 规划质量 (权重 0.2)
        planner_score = 1.0
        if planner_result:
            scores = [r.get("overall_score", 0) for r in planner_result]
            planner_score = (sum(scores) / len(scores) / 5.0) if scores else 1.0

        reliability = (
            asserter_reliability * 0.5
            + executor_accuracy * 0.3
            + planner_score * 0.2
        ) * 100

        return round(min(100, max(0, reliability)), 1)
