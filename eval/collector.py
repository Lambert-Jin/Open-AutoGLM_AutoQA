"""ScreenshotCollector: 包装 TestRunner 采集完整评测数据"""
from __future__ import annotations

import base64
import logging
import os
import time
from typing import Any

from config.settings import AssertResult
from eval.models import (
    ActionStepData,
    AssertStepData,
    EvalCaseData,
    EvalManifest,
    InstructionData,
    RoundData,
    TokenCount,
    TokenUsage,
)
from executor import ExecutorActionResult
from runner import TestRunner
from suite import (
    ActionStep,
    AssertStep,
    StepResult,
    TestCase,
    TestCaseResult,
    TestSuite,
    TestSuiteResult,
)

logger = logging.getLogger(__name__)


def _read_provider_tokens(obj, *attr_path) -> tuple[int, int]:
    """安全地从嵌套属性链中读取 provider 的 token 累计值"""
    try:
        current = obj
        for attr in attr_path:
            current = getattr(current, attr)
        provider = getattr(current, "provider")
        prompt = provider.total_prompt_tokens
        completion = provider.total_completion_tokens
        if isinstance(prompt, int) and isinstance(completion, int):
            return prompt, completion
    except (AttributeError, TypeError):
        pass
    return 0, 0


class ScreenshotCollector:
    """
    包装 TestRunner，在测试运行时采集完整数据用于评测。

    工作方式：
    1. 调用 runner.run_suite() 执行测试
    2. 从 TestSuiteResult 提取结果数据（ExecutorActionResult 已包含过程数据）
    3. 将截图保存到磁盘，构建 EvalManifest
    """

    def __init__(
        self,
        runner: TestRunner,
        output_dir: str,
        yaml_path: str = "",
    ):
        self.runner = runner
        self.output_dir = output_dir
        self.yaml_path = yaml_path

    def collect(self, suite: TestSuite) -> EvalManifest:
        """执行测试并采集评测数据"""
        run_id = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.output_dir, run_id)
        screenshots_dir = os.path.join(run_dir, "screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)

        # 执行测试
        suite_result = self.runner.run_suite(suite)

        # 从结果构建 manifest
        manifest = self._build_manifest(
            run_id, suite, suite_result, screenshots_dir,
        )

        # 持久化
        manifest.save(os.path.join(run_dir, "eval_manifest.json"))
        logger.info("评测数据已保存: %s", run_dir)

        return manifest

    def _build_manifest(
        self,
        run_id: str,
        suite: TestSuite,
        suite_result: TestSuiteResult,
        screenshots_dir: str,
    ) -> EvalManifest:
        cases = []
        for case_idx, case_result in enumerate(suite_result.cases):
            original_case = suite.test_cases[case_idx] if case_idx < len(suite.test_cases) else None
            description = original_case.description if original_case else ""

            steps = []
            for step_idx, step_result in enumerate(case_result.steps):
                step_data = self._build_step_data(
                    step_result, step_idx, case_idx, screenshots_dir,
                )
                steps.append(step_data)

            cases.append(EvalCaseData(
                case_name=case_result.case_name,
                status=case_result.status,
                description=description,
                steps=steps,
            ))

        # 累加 token 用量：AutoGLM 从 round_outputs，VLM/LLM 从 provider
        token_usage = self._aggregate_token_usage(cases, self.runner)

        return EvalManifest(
            run_id=run_id,
            suite_name=suite_result.suite_name,
            yaml_path=self.yaml_path,
            token_usage=token_usage,
            cases=cases,
        )

    @staticmethod
    def _aggregate_token_usage(cases: list[EvalCaseData], runner) -> TokenUsage:
        # AutoGLM: 从 round_outputs 累加
        autoglm_prompt = 0
        autoglm_completion = 0
        for case in cases:
            for step in case.steps:
                if isinstance(step, ActionStepData):
                    for rd in step.rounds:
                        out = rd.model_output
                        autoglm_prompt += out.get("prompt_tokens", 0)
                        autoglm_completion += out.get("completion_tokens", 0)

        # VLM: 从 asserter.provider 读取累加值
        vlm_prompt, vlm_completion = _read_provider_tokens(runner, "asserter")

        # LLM: 从 executor.action_optimizer.provider 读取累加值
        llm_prompt, llm_completion = _read_provider_tokens(runner, "executor", "action_optimizer")

        return TokenUsage(
            autoglm=TokenCount(prompt=autoglm_prompt, completion=autoglm_completion),
            vlm=TokenCount(prompt=vlm_prompt, completion=vlm_completion),
            llm=TokenCount(prompt=llm_prompt, completion=llm_completion),
        )

    def _build_step_data(
        self,
        step_result: StepResult,
        step_idx: int,
        case_idx: int,
        screenshots_dir: str,
    ) -> ActionStepData | AssertStepData:
        step = step_result.step

        if isinstance(step, ActionStep):
            detail: ExecutorActionResult = step_result.detail
            rounds = []
            for i, (before_b64, after_b64) in enumerate(detail.round_screenshots):
                before_path = os.path.join(screenshots_dir, f"c{case_idx}_s{step_idx}_r{i}_before.png")
                self._save_screenshot(before_b64, before_path)
                # finish/should_finish 轮无 after 截图，路径设为空
                after_path = ""
                if after_b64:
                    after_path = os.path.join(screenshots_dir, f"c{case_idx}_s{step_idx}_r{i}_after.png")
                    self._save_screenshot(after_b64, after_path)
                output = detail.round_outputs[i] if i < len(detail.round_outputs) else {}
                rounds.append(RoundData(
                    screenshot_before=before_path,
                    model_output=output,
                    screenshot_after=after_path,
                ))

            return ActionStepData(
                step_index=step_idx,
                instruction=InstructionData(
                    original=step.description,
                    optimized=detail.optimized_instruction,
                ),
                injected_history_length=detail.injected_history_length,
                rounds=rounds,
                conversation_history=detail.conversation_history,
                result={
                    "success": detail.success,
                    "rounds": detail.rounds,
                    "actions_taken": detail.actions_taken,
                    "error": detail.error,
                    "time_ms": step_result.timing.duration_ms,
                },
            )
        else:
            detail: AssertResult = step_result.detail
            # 保存 assert 截图到磁盘
            assert_screenshot_path = ""
            if detail.screenshot_base64:
                assert_screenshot_path = os.path.join(
                    screenshots_dir, f"c{case_idx}_s{step_idx}_assert.png",
                )
                self._save_screenshot(detail.screenshot_base64, assert_screenshot_path)
            return AssertStepData(
                step_index=step_idx,
                expectation=step.expectation,
                severity=step.severity,
                screenshot=assert_screenshot_path,
                result={
                    "passed": detail.passed,
                    "reason": detail.reason,
                    "confidence": detail.confidence,
                    "retried": detail.retried,
                },
            )

    @staticmethod
    def _save_screenshot(base64_data: str, path: str):
        if not base64_data:
            return
        import base64 as b64
        with open(path, "wb") as f:
            f.write(b64.b64decode(base64_data))
