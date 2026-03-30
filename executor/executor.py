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
from executor.context_agent import ContextAgent
from executor.model_protocol import ActionModel

if TYPE_CHECKING:
    from describer import PageDescriber
    from optimizer import ActionOptimizer

logger = logging.getLogger(__name__)


@dataclass
class ExecutorActionResult:
    """单个 ActionStep 的执行结果"""
    success: bool
    actions_taken: list[dict] = field(default_factory=list)
    rounds: int = 0
    error: str | None = None


class TestExecutor:
    """
    测试执行器。

    上下文策略：步骤级隔离 + ActionOptimizer 跨步骤指令优化
    - 每个 execute_action() 调用独立维护上下文（system prompt + 当前步骤对话）
    - 步骤间不共享完整历史，避免前序步骤的残留信息误导模型
    - 通过 PageDescriber（VLM）分析每步截图提取页面关键信息
    - 通过 ActionOptimizer（LLM）根据累积的历史步骤改写当前指令，使其更具体
    - AutoGLM 收到优化后的指令 + 当前截图
    """

    MAX_ROUNDS_PER_STEP = 15  # 单步骤内最多对话轮次

    def __init__(
        self,
        model: ActionModel,
        device: Device,
        max_steps_per_action: int = 20,
        action_cache: ActionCache | None = None,
        post_action_delay: float = 1.0,
        page_describer: PageDescriber | None = None,
        action_optimizer: ActionOptimizer | None = None,
    ):
        self.model = model
        self.device = device
        self.device_id = device.device_id
        self.action_executor = ActionExecutor(device)
        self.max_steps = max_steps_per_action
        self.action_cache = action_cache
        self.post_action_delay = post_action_delay  # 动作执行后等待页面加载的延迟（秒）
        self.page_describer = page_describer
        self.action_optimizer = action_optimizer
        self._system_prompt: str = ""  # 缓存 system prompt，避免重复获取

        # 异步上下文子代理：describe + record + precompute optimize 在后台并行
        self._context_agent: ContextAgent | None = None
        if self.page_describer or self.action_optimizer:
            self._context_agent = ContextAgent(self.page_describer, self.action_optimizer)

    def execute_action(
        self,
        description: str,
        cache_key: str = "",
        next_instruction: str | None = None,
    ) -> ExecutorActionResult:
        """
        执行一个语义级操作步骤。

        内部多轮循环直到模型返回 finish 或达到 max_steps。
        每次调用创建独立上下文，步骤间互不干扰。

        Args:
            next_instruction: 下一步指令，用于 ContextAgent 预计算优化
        """
        actions_taken: list[dict] = []
        verbose = logger.isEnabledFor(logging.DEBUG)
        original_description = description  # 保留原始描述用于日志

        # 每步独立上下文：system prompt + 当前步骤对话
        if not self._system_prompt:
            self._system_prompt = self.model.get_system_prompt()
        context: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
        ]

        # ── 取预计算结果（上一步 ContextAgent 预计算的优化指令）──
        precomputed = None
        if self._context_agent:
            precomputed = self._context_agent.take_precomputed()

        # 截图 + 构造消息
        screenshot = self.device.screenshot()
        current_app = self.device.current_app()

        # ── 指令优化：优先用预计算结果，否则同步降级 ──
        if self.action_optimizer:
            if precomputed is not None:
                description = precomputed
                logger.info("指令优化: 使用预计算结果")
            else:
                description = self.action_optimizer.optimize(description)
                if self._context_agent:
                    logger.info("指令优化: 同步降级（无预计算）")

        # ── 缓存快速路径 ──
        if cache_key and self.action_cache:
            activity = self.device.current_activity()
            cached = self.action_cache.lookup(
                cache_key, current_app, activity, screenshot,
            )
            if cached:
                logger.info("缓存命中: %s (相似度 %.2f)", cache_key, cached.similarity)
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
                cache_result = self.action_executor.execute(cache_action)
                if cache_result.success:
                    self.action_cache.record_hit(cached.entry)
                    if self._context_agent:
                        self._context_agent.submit(screenshot, original_description, next_instruction)
                    return ExecutorActionResult(
                        success=True,
                        actions_taken=[{"type": params["action_type"], "x": params["x"], "y": params["y"]}],
                        rounds=0,
                    )
                logger.warning("缓存动作执行失败，fallback 到正常流程")
            else:
                logger.info("缓存未命中: %s (app=%s, activity=%s)", cache_key, current_app, activity)
        initial_screenshot = screenshot  # 缓存写回用
        screen_info = self.model.build_screen_info(current_app)

        # ── 提交给 ContextAgent（AutoGLM 之前！与 AutoGLM 并行执行）──
        if self._context_agent:
            self._context_agent.submit(screenshot, original_description, next_instruction)
            logger.info("ContextAgent: 已提交，开始与 AutoGLM 并行")

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
            try:
                output = self.model.call(context)
            except Exception as e:
                logger.error("模型调用失败: %s", e)
                return ExecutorActionResult(
                    success=False, actions_taken=actions_taken,
                    rounds=round_num + 1, error=f"Model error: {e}",
                )

            # 解析为 UnifiedAction
            action = self.model.parse(output, screenshot.width, screenshot.height)

            if verbose:
                self._log_response(round_num + 1, action, output)

            # 更新上下文：移除旧图片 + 添加 assistant 回复
            context[-1] = self.model.remove_images(context[-1])
            context.append(
                self.model.build_assistant_message(output.raw_content)
            )

            # finish → 步骤完成
            if action.is_finish:
                logger.info("步骤完成: %s (共 %d 轮)", original_description, round_num + 1)
                exec_result = ExecutorActionResult(
                    success=True, actions_taken=actions_taken,
                    rounds=round_num + 1,
                )
                self._maybe_cache_action(
                    cache_key, exec_result, actions_taken,
                    current_app, initial_screenshot,
                )
                return exec_result

            # 执行动作
            result = self.action_executor.execute(action)
            actions_taken.append({
                "type": action.type.value,
                "x": action.x, "y": action.y,
                "end_x": action.end_x, "end_y": action.end_y,
            })

            if result.should_finish:
                exec_result = ExecutorActionResult(
                    success=result.success, actions_taken=actions_taken,
                    rounds=round_num + 1, error=result.message,
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
                # 敏感屏幕（支付/安全页面）：使用上次的截图尺寸，不发图片
                current_app = self.device.current_app()
                screen_info = self.model.build_screen_info(current_app)
                context.append(
                    self.model.build_user_message(
                        text=f"** Screen Info **\n{screen_info}\n"
                             "⚠️ 当前页面截图受限（可能是支付/安全页面），请根据之前的上下文继续操作",
                    )
                )
                continue

            current_app = self.device.current_app()
            screen_info = self.model.build_screen_info(current_app)

            context.append(
                self.model.build_user_message(
                    text=f"** Screen Info **\n{screen_info}",
                    image_base64=screenshot.base64_data,
                    screen_width=screenshot.width,
                    screen_height=screenshot.height,
                )
            )

        return ExecutorActionResult(
            success=False, actions_taken=actions_taken,
            rounds=self.max_steps, error="max_steps exceeded",
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
        if self._context_agent:
            self._context_agent.reset()
        self._system_prompt = ""
        if self.action_optimizer:
            self.action_optimizer.reset()
        logger.debug("Executor 已重置")

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
            logger.info("已写入缓存: %s (action=%s, x=%d, y=%d, app=%s, activity=%s)",
                        cache_key, first["type"], x_norm, y_norm, app, activity)
        except Exception as e:
            logger.warning("缓存写入失败: %s", e)

    @staticmethod
    def _log_request(context: list[dict], round_num: int):
        last_user = next(
            (m for m in reversed(context) if m.get("role") == "user"), None
        )
        if not last_user:
            return
        content = last_user.get("content", "")
        if isinstance(content, list):
            text = "\n".join(c.get("text", "") for c in content if c.get("type") == "text")
            has_image = any(c.get("type") == "image_url" for c in content)
        else:
            text = str(content)
            has_image = False
        image_tag = " [+截图]" if has_image else ""
        logger.debug(
            "──── 📤 Round %d | 上下文 %d 条消息%s ────\n%s",
            round_num, len(context), image_tag, text,
        )

    @staticmethod
    def _log_response(round_num: int, action: UnifiedAction, output):
        logger.debug(
            "──── 📥 Round %d | 动作: %s (%s, %s) ────\n%s",
            round_num, action.type.value, action.x, action.y, output.thinking,
        )
