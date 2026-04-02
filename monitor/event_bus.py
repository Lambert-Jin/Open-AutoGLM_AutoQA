"""监控事件总线"""

from __future__ import annotations

from collections import deque
from threading import Condition
from typing import Any


class MonitorEventBus:
    """线程安全事件缓冲"""

    def __init__(self, max_events: int = 1000):
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._cursor = 0
        self._condition = Condition()

    def publish(self, payload: dict[str, Any]) -> int:
        with self._condition:
            self._cursor += 1
            event = dict(payload)
            event["cursor"] = self._cursor
            self._events.append(event)
            self._condition.notify_all()
            return self._cursor

    def list_since(self, cursor: int = 0) -> tuple[list[dict[str, Any]], int]:
        with self._condition:
            events = [event for event in self._events if event["cursor"] > cursor]
            return events, self._cursor

    def wait_for_newer_than(self, cursor: int, timeout: float = 1.0) -> tuple[list[dict[str, Any]], int]:
        with self._condition:
            if self._cursor <= cursor:
                self._condition.wait(timeout=timeout)
            events = [event for event in self._events if event["cursor"] > cursor]
            return events, self._cursor
