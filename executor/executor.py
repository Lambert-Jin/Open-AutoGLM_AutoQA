"""TestExecutor：步骤级上下文隔离，依赖 ActionModel + Device"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from device import Device
from device.errors import ScreenshotSensitiveError
from executor.actions import ActionType, UnifiedAction
from executor.action_executor import ActionExecutor, ActionExecuteResult as _AER
from executor.model_protocol import ActionModel
from monitor import ActionTraceContext, NoopMonitorGateway

if TYPE_CHECKING:
    from cache import ActionCache
    from monitor import MonitorGateway
    from optimizer import ActionOptimizer

logger = logging.getLogger(__name__)

# topic loggers (midscene 风格)
log_ai_call = logging.getLogger("autoqa:ai:call")        # 模型请求/响应
log_ai_stats = logging.getLogger("autoqa:ai:stats")      # token/耗时统计
log_executor = logging.getLogger("autoqa:executor")       # 执行流程
log_cache = logging.getLogger("autoqa:cache")             # 缓存命中/写入
log_optimizer = logging.getLogger("autoqa:optimizer")     # 指令优化


@dataclass
class ExecutorActionResult:
    """单个 ActionStep 的执行结果"""
    success: bool
    actions_taken: list[dict] = field(default_factory=list)
    rounds: int = 0
    error: str | None = None
    # 评测扩展字段
    optimized_instruction: str = ""
    conversation_history: list[dict] = field(default_factory=list)
    round_screenshots: list[tuple[str, str]] = field(default_factory=list)  # [(before_b64, after_b64)]
    round_outputs: list[dict] = field(default_factory=list)  # [{"thinking", "action_text", "timing", "feedback"}]
    injected_history_length: int = 0  # 注入的历史对话条数


class TestExecutor:
    """
    测试执行器。

    上下文策略：步骤级隔离 + 跨步骤指令优化
    - 每个 execute_action() 调用独立维护上下文（system prompt + 当前步骤对话）
    - 步骤间不共享完整历史，避免前序步骤的残留信息误导模型
    - 通过 ActionOptimizer 将上一步的对话历史 + 当前指令生成优化后的指令
    """

    MAX_ROUNDS_PER_STEP = 15  # 单步骤内最多对话轮次

    def __init__(
        self,
        model: ActionModel,
        device: Device,
        max_steps_per_action: int = 20,
        action_cache: ActionCache | None = None,
        post_action_delay: float = 2.0,
        action_optimizer: ActionOptimizer | None = None,
        monitor: MonitorGateway | None = None,
    ):
        self.model = model
        self.device = device
        self.device_id = device.device_id
        self.action_executor = ActionExecutor(device)
        self.max_steps = max_steps_per_action
        self.action_cache = action_cache
        self.post_action_delay = post_action_delay  # 动作执行后等待页面加载的延迟（秒）
        self.action_optimizer = action_optimizer
        self.monitor = monitor or NoopMonitorGateway()
        self._system_prompt: str = ""  # 缓存 system prompt，避免重复获取
        self._last_step_context: list[dict[str, Any]] = []  # 上一步的对话历史

    def execute_action(
        self,
        description: str,
        cache_key: str = "",
        next_instruction: str | None = None,
        run_id: str = "",
        case_index: int = 0,
        step_index: int = 0,
    ) -> ExecutorActionResult:
        """
        执行一个语义级操作步骤。

        内部多轮循环直到模型返回 finish 或达到 max_steps。
        每次调用创建独立上下文，步骤间互不干扰。
        """
        actions_taken: list[dict] = []
        verbose = logger.isEnabledFor(logging.DEBUG)
        original_description = description  # 保留原始描述用于日志
        optimized_instruction = description
        round_screenshots: list[tuple[str, str]] = []
        round_outputs: list[dict] = []
        injected_history_length = 0

        # 构建上下文：system prompt + 历史对话 + 当前步骤
        if not self._system_prompt:
            self._system_prompt = self.model.get_system_prompt()
        context: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
        ]

        # ── 指令优化：将历史对话 + 当前指令输入给 optimizer ──
        if self.action_optimizer and self._last_step_context:
            log_optimizer.info("开始优化指令 (历史 %d 条消息): %s", len(self._last_step_context), description)
            description = self.action_optimizer.optimize(description, self._last_step_context)
            log_optimizer.info("优化结果: %s", description)
        elif self.action_optimizer:
            log_optimizer.info("跳过优化 (无历史对话): %s", description)
        optimized_instruction = description  # description was modified by optimizer

        # ── 注入历史对话（跳过 system prompt，旧截图替换为占位文字）──
        if self._last_step_context:
            for msg in self._last_step_context:
                if msg.get("role") == "system":
                    continue
                context.append(self._strip_images(msg))
            log_executor.info("注入历史对话: %d 条消息", len(context) - 1)
            injected_history_length = len(context) - 1  # minus system prompt

        # 截图 + 构造消息
        screenshot = self.device.screenshot()
        current_app = self.device.current_app()

        # ── 缓存快速路径 ──
        if cache_key and self.action_cache:
            activity = self.device.current_activity()
            cached = self.action_cache.lookup(
                cache_key, current_app, activity, screenshot,
            )
            if cached:
                log_cache.info("命中: %s (相似度 %.2f)", cache_key, cached.similarity)
                params = cached.to_action_params()
                # 归一化坐标 (0-999) → 绝对像素
                sw, sh = screenshot.width, screenshot.height
                abs_x = int(params["x"] / 999 * sw)
                abs_y = int(params["y"] / 999 * sh)
                abs_end_x = int(params["end_x"] / 999 * sw) if params.get("end_x") is not None else None
                abs_end_y = int(params["end_y"] / 999 * sh) if params.get("end_y") is not None else None
                cache_action = UnifiedAction(
                    type=ActionType(params["action_type"].lower()),
                    x=abs_x, y=abs_y,
                    end_x=abs_end_x,
                    end_y=abs_end_y,
                )
                cache_trace = self._build_trace_context(
                    run_id, case_index, step_index, 1, original_description, from_cache=True,
                )
                self.monitor.on_action_before(cache_trace, cache_action, screenshot)
                cache_result = self.action_executor.execute(cache_action)
                cache_after = screenshot
                if cache_result.success:
                    if self.post_action_delay > 0:
                        time.sleep(self.post_action_delay)
                    try:
                        cache_after = self.device.screenshot()
                    except ScreenshotSensitiveError:
                        cache_after = screenshot
                    self.monitor.on_action_after(
                        cache_trace, cache_action, cache_after,
                        success=True, message=cache_result.message,
                    )
                    self.action_cache.record_hit(cached.entry)
                    return ExecutorActionResult(
                        success=True,
                        actions_taken=[{"type": params["action_type"], "x": params["x"], "y": params["y"]}],
                        rounds=0,
                        optimized_instruction=optimized_instruction,
                        conversation_history=list(context),
                        round_screenshots=[],
                        round_outputs=[],
                        injected_history_length=injected_history_length,
                    )
                self.monitor.on_action_after(
                    cache_trace, cache_action, cache_after,
                    success=False, message=cache_result.message,
                )
                log_cache.warning("动作执行失败，fallback 到正常流程")
            else:
                log_cache.info("未命中: %s (app=%s, activity=%s)", cache_key, current_app, activity)
        initial_screenshot = screenshot  # 缓存写回用
        screen_info = self.model.build_screen_info(current_app)

        context.append(
            self.model.build_user_message(
                text=f"{description}\n\n{screen_info}",
                image_base64=screenshot.base64_data,
                screen_width=screenshot.width,
                screen_height=screenshot.height,
            )
        )

        for round_num in range(self.max_steps):
            if verbose:
                self._log_request(context, round_num + 1)

            # 调用模型
            round_before_screenshot = screenshot.base64_data
            log_ai_call.info("sending request to %s (round %d)", self.model.__class__.__name__, round_num + 1)
            try:
                output = self.model.call(context)
            except Exception as e:
                log_ai_call.error("调用失败: %s", e)
                return ExecutorActionResult(
                    success=False, actions_taken=actions_taken,
                    rounds=round_num + 1, error=f"Model error: {e}",
                    optimized_instruction=optimized_instruction,
                    conversation_history=list(context),
                    round_screenshots=round_screenshots,
                    round_outputs=round_outputs,
                    injected_history_length=injected_history_length,
                )

            # 耗时统计
            log_ai_stats.info(
                "round %d, ttft %.2fs, total %.2fs",
                round_num + 1,
                output.time_to_first_token or 0,
                output.total_time or 0,
            )

            # 解析为 UnifiedAction
            action = self.model.parse(output, screenshot.width, screenshot.height)
            round_outputs.append({
                "thinking": output.thinking,
                "action_text": output.action_text,
                "raw_content": output.raw_content,
                "timing": {"ttft": output.time_to_first_token, "total": output.total_time},
                "prompt_tokens": output.prompt_tokens,
                "completion_tokens": output.completion_tokens,
                "feedback": "",
            })

            if verbose:
                self._log_response(round_num + 1, action, output)

            # 更新上下文：移除旧图片 + 添加 assistant 回复
            context[-1] = self.model.remove_images(context[-1])
            context.append(
                self.model.build_assistant_message(output.raw_content)
            )

            # finish → 步骤完成
            if action.is_finish:
                log_executor.info("步骤完成: %s (共 %d 轮)", original_description, round_num + 1)
                round_screenshots.append((round_before_screenshot, ""))  # finish 无 after 截图
                self._last_step_context = context  # 保存对话历史供下一步 optimizer 使用
                exec_result = ExecutorActionResult(
                    success=True, actions_taken=actions_taken,
                    rounds=round_num + 1,
                    optimized_instruction=optimized_instruction,
                    conversation_history=list(context),
                    round_screenshots=round_screenshots,
                    round_outputs=round_outputs,
                    injected_history_length=injected_history_length,
                )
                self._maybe_cache_action(
                    cache_key, exec_result, actions_taken,
                    current_app, initial_screenshot,
                )
                return exec_result

            # 执行动作
            trace = self._build_trace_context(
                run_id, case_index, step_index, round_num + 1, original_description,
            )
            self.monitor.on_action_before(trace, action, screenshot)
            result = self.action_executor.execute(action)
            actions_taken.append({
                "type": action.type.value,
                "x": action.x, "y": action.y,
                "end_x": action.end_x, "end_y": action.end_y,
            })

            if result.should_finish:
                self.monitor.on_action_after(
                    trace, action, screenshot, success=result.success, message=result.message,
                )
                round_screenshots.append((round_before_screenshot, ""))  # should_finish 无 after 截图
                self._last_step_context = context
                exec_result = ExecutorActionResult(
                    success=result.success, actions_taken=actions_taken,
                    rounds=round_num + 1, error=result.message,
                    optimized_instruction=optimized_instruction,
                    conversation_history=list(context),
                    round_screenshots=round_screenshots,
                    round_outputs=round_outputs,
                    injected_history_length=injected_history_length,
                )
                self._maybe_cache_action(
                    cache_key, exec_result, actions_taken,
                    current_app, initial_screenshot,
                )
                return exec_result

            # 等待页面加载后再截图
            if self.post_action_delay > 0:
                time.sleep(self.post_action_delay)

            # 下一轮截图
            try:
                screenshot = self.device.screenshot()
            except ScreenshotSensitiveError:
                self.monitor.on_action_after(
                    trace, action, screenshot, success=result.success, message=result.message,
                )
                # 敏感屏幕（支付/安全页面）：使用上次的截图尺寸，不发图片
                current_app = self.device.current_app()
                screen_info = self.model.build_screen_info(current_app)
                feedback = self._build_feedback(action, result)
                if round_outputs:
                    round_outputs[-1]["feedback"] = feedback
                round_screenshots.append((round_before_screenshot, ""))  # 敏感页面无 after 截图
                context.append(
                    self.model.build_user_message(
                        text=f"{feedback}\n{screen_info}\n"
                             "⚠️ 当前页面截图受限（可能是支付/安全页面），请根据之前的上下文继续操作",
                    )
                )
                continue

            self.monitor.on_action_after(
                trace, action, screenshot, success=result.success, message=result.message,
            )
            current_app = self.device.current_app()
            screen_info = self.model.build_screen_info(current_app)
            feedback = self._build_feedback(action, result)
            if round_outputs:
                round_outputs[-1]["feedback"] = feedback
            round_screenshots.append((round_before_screenshot, screenshot.base64_data))

            context.append(
                self.model.build_user_message(
                    text=f"{feedback}\n{screen_info}",
                    image_base64=screenshot.base64_data,
                    screen_width=screenshot.width,
                    screen_height=screenshot.height,
                )
            )

        self._last_step_context = context
        return ExecutorActionResult(
            success=False, actions_taken=actions_taken,
            rounds=self.max_steps, error="max_steps exceeded",
            optimized_instruction=optimized_instruction,
            conversation_history=list(context),
            round_screenshots=round_screenshots,
            round_outputs=round_outputs,
            injected_history_length=injected_history_length,
        )

    def handle_unexpected(
        self,
        instruction: str = "关闭当前弹窗或广告",
        max_steps: int = 3,
    ) -> bool:
        """处理意外情况（弹窗、广告等）"""
        original_max = self.max_steps
        self.max_steps = max_steps
        result = self.execute_action(instruction)
        self.max_steps = original_max
        return result.success

    def reset(self):
        """重置状态（切换 TestCase 时调用）"""
        self._system_prompt = ""
        self._last_step_context = []
        log_executor.debug("已重置")

    @staticmethod
    def _build_trace_context(
        run_id: str,
        case_index: int,
        step_index: int,
        round_index: int,
        step_description: str,
        *,
        from_cache: bool = False,
    ) -> ActionTraceContext:
        safe_run_id = run_id or "adhoc"
        action_id = f"a_{case_index:02d}_{step_index:02d}_{round_index:02d}"
        return ActionTraceContext(
            run_id=safe_run_id,
            case_index=case_index,
            step_index=step_index,
            round_index=round_index,
            step_description=step_description,
            action_id=action_id,
            from_cache=from_cache,
        )

    def _maybe_cache_action(
        self,
        cache_key: str,
        result: ExecutorActionResult,
        actions_taken: list[dict],
        app: str,
        screenshot,
    ):
        """成功且仅执行 1 个动作时写入缓存（多动作操作不缓存）"""
        if not cache_key or not self.action_cache:
            return
        if not result.success or len(actions_taken) != 1:
            return

        first = actions_taken[0]

        # 无坐标的动作（launch/back/home 等）不缓存
        if first["x"] is None or first["y"] is None:
            return

        activity = self.device.current_activity()

        # 绝对像素坐标 → 归一化坐标 (0-999)
        sw, sh = screenshot.width, screenshot.height
        x_norm = int(first["x"] / sw * 999) if sw else first["x"]
        y_norm = int(first["y"] / sh * 999) if sh else first["y"]
        end_x_norm = int(first["end_x"] / sw * 999) if first.get("end_x") and sw else first.get("end_x")
        end_y_norm = int(first["end_y"] / sh * 999) if first.get("end_y") and sh else first.get("end_y")

        try:
            self.action_cache.store_action(
                cache_key=cache_key,
                app=app,
                activity=activity,
                action_type=first["type"],
                x=x_norm,
                y=y_norm,
                end_x=end_x_norm,
                end_y=end_y_norm,
                screenshot=screenshot,
            )
            log_cache.info("写入: %s (action=%s, x=%d, y=%d, app=%s, activity=%s)",
                          cache_key, first["type"], x_norm, y_norm, app, activity)
        except Exception as e:
            log_cache.warning("写入失败: %s", e)

    @staticmethod
    def _strip_images(message: dict) -> dict:
        """将消息中的图片替换为占位文字（用于历史对话注入，节省 token）"""
        content = message.get("content", "")
        if not isinstance(content, list):
            return message
        stripped = []
        for part in content:
            if part.get("type") == "image_url":
                stripped.append({"type": "text", "text": "(历史截图已省略)"})
            else:
                stripped.append(part)
        return {**message, "content": stripped}

    @staticmethod
    def _build_feedback(action: UnifiedAction, result: _AER) -> str:
        """构建执行反馈文本，注入到下一轮 user message 中"""
        atype = action.type
        if atype in (ActionType.TAP, ActionType.DOUBLE_TAP, ActionType.LONG_PRESS):
            params = f"({action.x}, {action.y})"
        elif atype == ActionType.SWIPE:
            params = f"(({action.x},{action.y})→({action.end_x},{action.end_y}))"
        elif atype == ActionType.TYPE:
            text_preview = action.text[:20] + "..." if action.text and len(action.text) > 20 else action.text
            params = f'("{text_preview}")'
        elif atype == ActionType.LAUNCH:
            params = f"({action.text})"
        elif atype == ActionType.WAIT:
            params = f"({action.duration_ms or 2000}ms)"
        else:
            params = ""

        status = "执行成功" if result.success else f"执行失败：{result.message}" if result.message else "执行失败"
        return f"[执行反馈] 已执行 {atype.value}{params}，{status}。以下是最新截图："

    @staticmethod
    def _log_request(context: list[dict], round_num: int):
        import json

        def _truncate_msg(msg: dict) -> dict:
            """截断图片数据，保留结构"""
            content = msg.get("content", "")
            if not isinstance(content, list):
                return msg
            truncated = []
            for part in content:
                if part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    truncated.append({
                        "type": "image_url",
                        "image_url": {"url": f"{url[:80]}...[truncated]"},
                    })
                else:
                    truncated.append(part)
            return {**msg, "content": truncated}

        display = [_truncate_msg(m) for m in context]
        log_ai_call.debug(
            "request messages (%d messages):\n%s",
            len(context),
            json.dumps(display, ensure_ascii=False, indent=2),
        )

    @staticmethod
    def _log_response(round_num: int, action: UnifiedAction, output):
        log_ai_call.debug(
            "response content:\n%s",
            output.raw_content,
        )
