"""测试套件数据模型"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


# ── 步骤定义 ──

@dataclass
class ActionStep:
    """操作步骤"""
    description: str
    timeout: int = 30
    cache_key: str = ""  # 归一化缓存键，由 planner 生成或 YAML 手写


@dataclass
class AssertStep:
    """断言步骤"""
    expectation: str
    severity: str = "critical"          # critical | warning | info
    retry_on_fail: bool = False
    retry_cleanup: str = "关闭当前弹窗或广告"


Step = ActionStep | AssertStep


# ── 测试用例与套件 ──

@dataclass
class TestCase:
    """一个测试用例，包含多个步骤"""
    name: str
    steps: list[Step] = field(default_factory=list)
    continue_on_error: bool = False
    description: str = ""               # 自然语言描述（可选，供 Planner 使用）


@dataclass
class TestSuite:
    """测试套件，包含多个测试用例"""
    name: str
    test_cases: list[TestCase] = field(default_factory=list)


# ── 执行结果 ──

@dataclass
class Timing:
    """步骤计时"""
    start_time: float = 0.0
    end_time: float = 0.0

    @classmethod
    def start_now(cls) -> Timing:
        return cls(start_time=time.time())

    def stop(self):
        self.end_time = time.time()

    @property
    def duration_ms(self) -> float:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0


@dataclass
class StepResult:
    """单步执行结果"""
    step: Step
    success: bool
    timing: Timing = field(default_factory=Timing)
    detail: Any = None                  # ExecutorActionResult 或 AssertResult


@dataclass
class TestCaseResult:
    """测试用例执行结果"""
    case_name: str
    steps: list[StepResult] = field(default_factory=list)
    status: str = "pending"             # passed | failed | error

    @property
    def passed_count(self) -> int:
        return sum(1 for s in self.steps if s.success)

    @property
    def total_count(self) -> int:
        return len(self.steps)


@dataclass
class TestSuiteResult:
    """测试套件执行结果"""
    suite_name: str
    cases: list[TestCaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(c.total_count for c in self.cases)

    @property
    def passed(self) -> int:
        return sum(c.passed_count for c in self.cases)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def duration_ms(self) -> float:
        total = 0.0
        for case in self.cases:
            for step in case.steps:
                total += step.timing.duration_ms
        return total
