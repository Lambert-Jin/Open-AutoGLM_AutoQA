"""TestRunner：测试编排与容错重试"""

from __future__ import annotations

import logging
from datetime import datetime

from asserter import Asserter
from config.settings import AssertResult
from executor import ExecutorActionResult, TestExecutor
from monitor import NoopMonitorGateway
from screenshot import ScreenshotManager
from suite import (
    ActionStep,
    AssertStep,
    StepResult,
    TestCase,
    TestCaseResult,
    TestSuite,
    TestSuiteResult,
    Timing,
)

logger = logging.getLogger(__name__)


class TestRunner:
    """测试运行器：编排 TestCase、执行步骤、容错重试"""

    def __init__(
        self,
        executor: TestExecutor,
        asserter: Asserter,
        screenshot_mgr: ScreenshotManager | None = None,
        monitor=None,
    ):
        self.executor = executor
        self.asserter = asserter
        self.screenshot_mgr = screenshot_mgr or ScreenshotManager(device=executor.device)
        self.monitor = monitor or NoopMonitorGateway()

    def run_suite(self, suite: TestSuite) -> TestSuiteResult:
        """运行整个测试套件"""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.monitor.on_run_start(run_id, suite.name)
        print(f"\n{'='*60}")
        print(f"  测试套件: {suite.name}")
        print(f"  用例数量: {len(suite.test_cases)}")
        print(f"{'='*60}\n")

        cases: list[TestCaseResult] = []
        for i, test_case in enumerate(suite.test_cases, 1):
            # 多用例时，在每个用例开始前清理环境
            if i > 1:
                self._cleanup_device()

            print(f"── 用例 {i}/{len(suite.test_cases)}: {test_case.name} ──\n")
            self.monitor.on_case_start(run_id, i, test_case.name)
            result = self.run_case(test_case, run_id=run_id, case_index=i)
            cases.append(result)
            self.monitor.on_case_end(run_id, i, test_case.name, result.status == "passed")
            self._print_case_summary(result)

        suite_result = TestSuiteResult(suite_name=suite.name, cases=cases)
        self.monitor.on_run_end(run_id, suite_result.failed == 0)
        self._print_suite_summary(suite_result)
        return suite_result

    def run_case(self, case: TestCase, run_id: str = "", case_index: int = 0) -> TestCaseResult:
        """运行单个测试用例"""
        self.executor.reset()

        step_results: list[StepResult] = []

        for i, step in enumerate(case.steps, 1):
            step_type = "action" if isinstance(step, ActionStep) else "assert"
            step_title = step.description if isinstance(step, ActionStep) else step.expectation
            self.monitor.on_step_start(run_id, case_index, i, step_type, step_title)
            if isinstance(step, ActionStep):
                next_action = self._peek_next_action(case.steps, i)
                result = self._run_action(
                    step, i,
                    next_instruction=next_action.description if next_action else None,
                    run_id=run_id,
                    case_index=case_index,
                )
            elif isinstance(step, AssertStep):
                result = self._run_assert(step, i)
            else:
                continue

            step_results.append(result)
            self.monitor.on_step_end(
                run_id,
                case_index,
                i,
                step_type,
                step_title,
                result.success,
                message=self._step_message(result),
            )

            if not result.success and not case.continue_on_error:
                logger.info("步骤失败且 continueOnError=False，中断后续步骤")
                break

        status = "passed" if all(r.success for r in step_results) else "failed"
        return TestCaseResult(case_name=case.name, steps=step_results, status=status)

    def _run_action(
        self,
        step: ActionStep,
        step_num: int,
        next_instruction: str | None = None,
        *,
        run_id: str = "",
        case_index: int = 0,
    ) -> StepResult:
        """执行操作步骤"""
        timing = Timing.start_now()

        print(f"  Step {step_num}: [操作] {step.description} ... ", end="", flush=True)

        result: ExecutorActionResult = self.executor.execute_action(
            step.description, cache_key=step.cache_key,
            next_instruction=next_instruction,
            run_id=run_id,
            case_index=case_index,
            step_index=step_num,
        )

        timing.stop()
        duration = timing.duration_ms / 1000

        if result.success:
            print(f"OK ({result.rounds} 轮, {duration:.1f}s)")
        else:
            print(f"FAIL ({result.error})")

        return StepResult(
            step=step,
            success=result.success,
            timing=timing,
            detail=result,
        )

    def _run_assert(self, step: AssertStep, step_num: int) -> StepResult:
        """执行断言步骤，含容错重试"""
        timing = Timing.start_now()

        print(f"  Step {step_num}: [断言] {step.expectation} ... ", end="", flush=True)

        # 截图并断言（直接用已绑定 device 的 screenshot_mgr）
        screenshot = self.screenshot_mgr.capture()
        result: AssertResult = self.asserter.verify(screenshot, step.expectation)

        # 容错：断言失败 → 清理环境 → 重试
        if not result.passed and step.retry_on_fail:
            print(f"RETRY ", end="", flush=True)
            logger.info("断言失败，尝试清理后重试: %s", step.retry_cleanup)

            self.executor.handle_unexpected(step.retry_cleanup)

            screenshot = self.screenshot_mgr.capture()
            result = self.asserter.verify(screenshot, step.expectation)
            result.retried = True

        timing.stop()
        duration = timing.duration_ms / 1000

        if result.passed:
            retry_mark = "(重试后) " if result.retried else ""
            print(f"PASS {retry_mark}(confidence: {result.confidence:.2f}, {duration:.1f}s)")
        else:
            sev = f"[{step.severity}] " if step.severity != "critical" else ""
            print(f"FAIL {sev}(confidence: {result.confidence:.2f}, {duration:.1f}s)")
            print(f"           reason: {result.reason}")

        return StepResult(
            step=step,
            success=result.passed,
            timing=timing,
            detail=result,
        )

    @staticmethod
    def _peek_next_action(steps: list, current_index: int) -> ActionStep | None:
        """向前查找下一个 ActionStep（current_index 从 1 开始）"""
        for step in steps[current_index:]:
            if isinstance(step, ActionStep):
                return step
        return None

    def _cleanup_device(self):
        """用例间清理：回桌面 + 关闭所有后台 App"""
        device = self.executor.device
        print("  [清理] 回到桌面，关闭后台 App ... ", end="", flush=True)
        try:
            device.home()
            device.kill_all_apps()
            print("OK")
        except Exception as e:
            print(f"WARN ({e})")
            logger.warning("设备清理失败: %s", e)

    @staticmethod
    def _step_message(result: StepResult) -> str | None:
        detail = result.detail
        if detail is None:
            return None
        if hasattr(detail, "error") and detail.error:
            return detail.error
        if hasattr(detail, "reason") and detail.reason:
            return detail.reason
        return None

    @staticmethod
    def _print_case_summary(result: TestCaseResult):
        icon = "+" if result.status == "passed" else "x"
        print(f"\n  [{icon}] 用例结果: {result.case_name} — "
              f"{result.passed_count}/{result.total_count} passed\n")

    @staticmethod
    def _print_suite_summary(result: TestSuiteResult):
        print(f"{'='*60}")
        print(f"  测试结果: {result.passed}/{result.total} passed, "
              f"{result.failed} failed, {result.duration_ms/1000:.1f}s")

        # 列出失败详情
        for case in result.cases:
            if case.status == "failed":
                print(f"\n  [x] {case.case_name}:")
                for i, step_result in enumerate(case.steps, 1):
                    if not step_result.success:
                        step = step_result.step
                        if isinstance(step, AssertStep):
                            reason = ""
                            if hasattr(step_result.detail, "reason"):
                                reason = f" — {step_result.detail.reason}"
                            print(f"     Step {i} [断言失败] {step.expectation}{reason}")
                        elif isinstance(step, ActionStep):
                            error = ""
                            if hasattr(step_result.detail, "error") and step_result.detail.error:
                                error = f" — {step_result.detail.error}"
                            print(f"     Step {i} [操作失败] {step.description}{error}")

        if result.failed == 0:
            print(f"\n  [+] {result.suite_name}")
        else:
            print(f"\n  [x] {result.suite_name}")
        print(f"{'='*60}\n")
