"""上下文子代理：后台异步执行页面描述、历史记录、指令预优化"""

from __future__ import annotations

import logging
import time
import threading
from queue import Queue

logger = logging.getLogger(__name__)

_SENTINEL = None  # 停止信号


class ContextAgent:
    """后台子代理，负责 describe → record → precompute optimize。

    Executor 通过两个方法交互：
    - submit(screenshot, description, next_instruction): 提交当前步骤（非阻塞）
    - take_precomputed(): 取走上一步预计算的优化指令（一次性消费）

    用 generation 计数器标记每次 submit，防止过期 agent 结果被错误消费。
    """

    def __init__(self, page_describer=None, action_optimizer=None, take_timeout: float = 5.0):
        self._page_describer = page_describer
        self._action_optimizer = action_optimizer
        self._take_timeout = take_timeout
        self._queue: Queue = Queue()
        self._generation: int = 0  # 每次 submit 递增，用于过滤过期预计算
        self._precomputed: tuple[int, str] | None = None  # (generation, result)
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._ready.set()  # 初始状态：就绪（无待处理任务）
        self._thread = threading.Thread(target=self._loop, daemon=True, name="context-agent")
        self._thread.start()

    # ── 公开接口 ──

    def submit(self, screenshot, description: str, next_instruction: str | None = None):
        """非阻塞：提交当前步骤的上下文处理任务"""
        self._generation += 1
        gen = self._generation
        self._ready.clear()  # 标记为忙
        self._queue.put((gen, screenshot, description, next_instruction))
        logger.debug("submit: gen=%d, next=%s", gen, next_instruction[:20] if next_instruction else None)

    def take_precomputed(self, timeout: float | None = None) -> str | None:
        """等待并取走预计算的优化指令。

        等待 agent 处理完毕（最多 timeout 秒），然后取走结果。
        只接受当前 generation 的预计算结果，过期结果直接丢弃。
        """

        t = timeout if timeout is not None else self._take_timeout
        expected_gen = self._generation
        t0 = time.monotonic()
        ready = self._ready.wait(timeout=t)
        wait_ms = (time.monotonic() - t0) * 1000

        with self._lock:
            if self._precomputed and self._precomputed[0] == expected_gen:
                result = self._precomputed[1]
                self._precomputed = None
                logger.info("take: 命中预计算 gen=%d (等待 %.0fms)", expected_gen, wait_ms)
                return result
            if self._precomputed:
                logger.info("take: 丢弃过期预计算 gen=%d (期望 gen=%d, 等待 %.0fms)",
                            self._precomputed[0], expected_gen, wait_ms)
                self._precomputed = None
            elif not ready:
                logger.info("take: 超时 %.0fms，降级同步 (gen=%d)", wait_ms, expected_gen)
            else:
                logger.debug("take: 无预计算结果 (等待 %.0fms, gen=%d)", wait_ms, expected_gen)
        return None

    def drain(self):
        """等待队列中所有任务处理完毕"""
        self._queue.join()

    def stop(self):
        """停止后台线程"""
        self._queue.put(_SENTINEL)
        self._thread.join(timeout=10)

    def reset(self):
        """重置状态：等待当前任务完成，清空预计算结果"""
        self.drain()
        with self._lock:
            self._precomputed = None
        self._ready.set()

    # ── 后台事件循环 ──

    def _loop(self):
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                self._queue.task_done()
                break
            gen, screenshot, description, next_instruction = item
            try:
                self._process(gen, screenshot, description, next_instruction)
            except Exception as e:
                logger.warning("ContextAgent 处理失败 (gen=%d): %s", gen, e)
            finally:
                # 只有最新 generation 的任务完成时才唤醒 take
                if gen == self._generation:
                    self._ready.set()
                self._queue.task_done()

    def _process(self, gen: int, screenshot, description: str, next_instruction: str | None):
        t0 = time.monotonic()

        # 1. describe
        page_desc = ""
        if self._page_describer:
            try:
                page_desc = self._page_describer.describe(screenshot)
                logger.debug("agent: describe 完成 gen=%d (%.1fs)", gen, time.monotonic() - t0)
            except Exception as e:
                logger.warning("页面描述失败: %s", e)

        # 2. record
        if self._action_optimizer:
            self._action_optimizer.record(description, page_desc)

        # 3. precompute next optimize
        if next_instruction and self._action_optimizer:
            try:
                t1 = time.monotonic()
                optimized = self._action_optimizer.optimize(next_instruction)
                with self._lock:
                    self._precomputed = (gen, optimized)
                logger.debug("agent: optimize 完成 gen=%d (%.1fs)", gen, time.monotonic() - t1)
            except Exception as e:
                logger.warning("预计算优化失败: %s", e)

        logger.info("agent: 链完成 gen=%d (总 %.1fs) [describe+record+optimize]", gen, time.monotonic() - t0)