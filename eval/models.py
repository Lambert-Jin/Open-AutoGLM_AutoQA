"""评测系统数据模型"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenCount:
    prompt: int = 0
    completion: int = 0


@dataclass
class TokenUsage:
    autoglm: TokenCount = field(default_factory=TokenCount)
    vlm: TokenCount = field(default_factory=TokenCount)
    llm: TokenCount = field(default_factory=TokenCount)


@dataclass
class InstructionData:
    original: str
    optimized: str  # ActionOptimizer 输出（无历史时与 original 相同）


@dataclass
class RoundData:
    screenshot_before: str
    model_output: dict[str, Any]
    parsed_action: dict[str, Any]
    screenshot_after: str


@dataclass
class ActionStepData:
    step_index: int
    instruction: InstructionData
    injected_history_length: int = 0  # 注入了多少条历史对话
    rounds: list[RoundData] = field(default_factory=list)
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssertStepData:
    step_index: int
    expectation: str
    severity: str
    screenshot: str
    result: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalCaseData:
    case_name: str
    status: str
    description: str
    steps: list[ActionStepData | AssertStepData] = field(default_factory=list)


@dataclass
class _StepWithContext:
    """评测器消费的步骤数据，附带 case 上下文"""
    case_name: str
    step: ActionStepData | AssertStepData


@dataclass
class EvalManifest:
    run_id: str
    suite_name: str
    yaml_path: str
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    cases: list[EvalCaseData] = field(default_factory=list)

    def get_assert_steps(self) -> list[_StepWithContext]:
        result = []
        for case in self.cases:
            for step in case.steps:
                if isinstance(step, AssertStepData):
                    result.append(_StepWithContext(case_name=case.case_name, step=step))
        return result

    def get_action_steps(self) -> list[_StepWithContext]:
        result = []
        for case in self.cases:
            for step in case.steps:
                if isinstance(step, ActionStepData):
                    result.append(_StepWithContext(case_name=case.case_name, step=step))
        return result

    def has_planner_cases(self) -> bool:
        return any(c.description for c in self.cases)

    def to_json(self) -> str:
        return json.dumps(self._to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> EvalManifest:
        data = json.loads(json_str)
        return cls._from_dict(data)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> EvalManifest:
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_json(f.read())

    def _to_dict(self) -> dict:
        def _dc_to_dict(obj):
            if hasattr(obj, "__dataclass_fields__"):
                result = {}
                for k in obj.__dataclass_fields__:
                    result[k] = _dc_to_dict(getattr(obj, k))
                return result
            if isinstance(obj, list):
                return [_dc_to_dict(item) for item in obj]
            if isinstance(obj, dict):
                return {k: _dc_to_dict(v) for k, v in obj.items()}
            return obj

        d = _dc_to_dict(self)
        # 给 step 加上 step_type 标记以便反序列化区分
        for case in d.get("cases", []):
            for step in case.get("steps", []):
                if "instruction" in step:
                    step["step_type"] = "action"
                else:
                    step["step_type"] = "assert"
        return d

    @classmethod
    def _from_dict(cls, data: dict) -> EvalManifest:
        tu = data.get("token_usage", {})
        token_usage = TokenUsage(
            autoglm=TokenCount(**tu.get("autoglm", {})),
            vlm=TokenCount(**tu.get("vlm", {})),
            llm=TokenCount(**tu.get("llm", {})),
        )
        cases = []
        for c in data.get("cases", []):
            steps = []
            for s in c.get("steps", []):
                if s.get("step_type") == "action":
                    inst = s["instruction"]
                    rounds = [RoundData(**r) for r in s.get("rounds", [])]
                    steps.append(ActionStepData(
                        step_index=s["step_index"],
                        instruction=InstructionData(**inst),
                        injected_history_length=s.get("injected_history_length", 0),
                        rounds=rounds,
                        conversation_history=s.get("conversation_history", []),
                        result=s.get("result", {}),
                    ))
                else:
                    steps.append(AssertStepData(
                        step_index=s["step_index"],
                        expectation=s["expectation"],
                        severity=s["severity"],
                        screenshot=s["screenshot"],
                        result=s.get("result", {}),
                    ))
            cases.append(EvalCaseData(
                case_name=c["case_name"],
                status=c["status"],
                description=c.get("description", ""),
                steps=steps,
            ))
        return cls(
            run_id=data["run_id"],
            suite_name=data["suite_name"],
            yaml_path=data["yaml_path"],
            token_usage=token_usage,
            cases=cases,
        )
