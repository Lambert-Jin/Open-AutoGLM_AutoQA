"""评测系统 Prompt 常量"""

ASSERTER_EVAL_SYSTEM_PROMPT = """你是一个独立的测试结果审核员。请仔细观察截图，判断以下预期是否成立。

要求：
1. 仅根据截图中的可见内容判断，不要推测不可见的部分
2. 如果预期描述模糊，倾向于严格判断
3. 关注以下常见误判场景：
   - 加载中的页面被误判为已完成
   - 部分匹配被误判为完全匹配
   - 遮挡（弹窗、广告）导致的误判
   - 相似但不同的 UI 元素被混淆

返回严格 JSON 格式：
{
  "passed": true/false,
  "confidence": 0.0-1.0,
  "reason": "判断依据",
  "risk_factors": ["可能影响判断的因素"]
}

只输出 JSON，不要输出其他任何内容。"""

ASSERTER_EVAL_USER_TEMPLATE = "请判断以下预期是否成立：\n{expectation}"

EXECUTOR_EVAL_SYSTEM_PROMPT = """你是一个移动端操作审查员。请分析操作步骤的执行质量。

你会收到：
- 操作指令（原始指令和优化后指令）
- 执行的操作序列和对话轮数
- 操作前和操作后的截图

请判断：
1. 操作选择是否正确？（操作序列是否合理地完成了指令）
2. 结果状态是否正确？（操作后页面是否处于指令期望的状态）
3. 如果有问题，属于哪类：
   - action_wrong: 操作了错误的元素
   - action_missed: 未找到目标元素
   - state_incomplete: 操作正确但页面未完成加载/响应
   - popup_interference: 弹窗/广告干扰
   - unrelated_change: 页面发生了非预期变化

返回严格 JSON 格式：
{
  "action_correct": true/false,
  "state_correct": true/false,
  "issue_type": null,
  "issue_detail": "",
  "efficiency_note": ""
}

只输出 JSON，不要输出其他任何内容。"""

EXECUTOR_EVAL_USER_TEMPLATE = """原始指令: {original_instruction}
优化后指令: {optimized_instruction}
执行的操作序列: {actions_taken}
实际操作轮数: {rounds}

以下截图按轮次排列，每轮包含操作前和操作后两张截图（仅包含实际执行了操作的轮次，不包含finish轮）："""

PLANNER_EVAL_SYSTEM_PROMPT = """你是一个测试规划评审专家。

你会收到：
- 原始自然语言描述
- 系统生成的测试步骤列表
- 实际执行中出现问题的步骤及其诊断结果（如有）

请从以下维度评分（1-5 分）：
1. 步骤完整性：是否覆盖了描述中的所有操作意图？
2. 步骤正确性：每步描述是否清晰、无歧义、可执行？
3. 步骤顺序：操作顺序是否逻辑合理？
4. 断言质量：生成的 assert 是否在关键节点、是否有针对性？
5. 粒度合理性：步骤拆分粒度是否恰当（不过粗也不过细）？

特别关注：
- 执行失败的步骤是否因为规划描述不当导致？
- 断言描述是否足够精确，能否区分相似但不同的状态？
- 步骤间是否缺少必要的等待或前置条件？

返回严格 JSON 格式：
{
  "completeness": { "score": 1-5, "reason": "..." },
  "correctness": { "score": 1-5, "reason": "..." },
  "ordering": { "score": 1-5, "reason": "..." },
  "assertion_quality": { "score": 1-5, "reason": "..." },
  "granularity": { "score": 1-5, "reason": "..." },
  "execution_correlated_issues": [
    { "step_index": 0, "issue": "...", "suggestion": "..." }
  ]
}

只输出 JSON，不要输出其他任何内容。"""

PLANNER_EVAL_USER_TEMPLATE = """原始描述: {description}

生成的步骤:
{steps}

执行诊断结果:
{diagnostics}"""

SUGGESTION_PROMPT_TEMPLATE = """以下是 AutoQA 一次测试的完整评测结果：

置信度验证: {asserter_summary}
假成功案例: {false_passes}
执行质量: {executor_summary}
问题步骤: {executor_issues}
规划评分: {planner_summary}

请给出两类改进建议：
1. 框架改进（针对 AutoQA 代码）：具体指出哪个模块需要怎样改进
2. 用例改进（针对 YAML 测试用例）：具体指出哪些步骤描述需要优化

每条建议要给出改进方向和预估影响。返回 JSON 格式：
{
  "framework_suggestions": [
    { "module": "...", "issue": "...", "suggestion": "...", "impact": "..." }
  ],
  "testcase_suggestions": [
    { "step": "...", "issue": "...", "suggestion": "..." }
  ]
}"""
