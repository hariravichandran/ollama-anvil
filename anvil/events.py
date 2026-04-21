"""Event bus — pub/sub for internal anvil events.

Lightweight synchronous bus. Every lifecycle point (tool call, LLM
call, skill injection, circuit-breaker trip, hook fired) can publish a
named event; any module can subscribe. Enables custom observability,
plugins, UIs without forking anvil itself.

::

    from anvil.events import bus, on

    @on("tool.executed")
    def log_tool(event):
        print(event.name, event.payload["tool"], event.payload["elapsed_s"])

    # Publishers — typically called from inside anvil, but any code may:
    bus.publish("tool.executed", tool="read_file", elapsed_s=0.02)

**Sync semantics.** Subscribers run in the publisher's thread inline.
If you want async handling, put the work on your own queue inside the
subscriber — the bus stays simple and predictable. Errors inside a
subscriber are captured and logged but do NOT stop other subscribers
from running. This matches the hooks-system contract from Feature 2.

**Module-level convenience.** ``bus`` is the default singleton; ``on``
is a decorator that registers on it. Tests that want isolation
instantiate their own :class:`EventBus`.
"""

from __future__ import annotations

import fnmatch
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from anvil.utils.logging import get_logger

log = get_logger("events")

__all__ = [
    "Event",
    "EventBus",
    "bus",
    "on",
    "publish",
]


@dataclass(slots=True)
class Event:
    """One published event."""

    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = ""


# Subscriber signature: ``(event) -> None``. Synchronous.
Subscriber = Callable[[Event], None]


class EventBus:
    """Named synchronous event bus with glob subscriptions.

    Subscribers filter by name pattern (``"tool.*"`` matches
    ``"tool.executed"``, ``"tool.denied"``). Exact names work too.
    """

    def __init__(self) -> None:
        self._subscribers: list[tuple[str, Subscriber]] = []
        self._lock = threading.Lock()
        self._published_count = 0

    def __repr__(self) -> str:
        return f"EventBus(subscribers={len(self._subscribers)}, published={self._published_count})"

    # ─── Subscriptions ───────────────────────────────────────────

    def subscribe(self, pattern: str, handler: Subscriber) -> Subscriber:
        """Register ``handler`` for events matching ``pattern``.

        Returns the same callable so this can be used as a decorator::

            @bus.subscribe("llm.*")
            def log_llm(event): ...
        """
        if not pattern:
            raise ValueError("pattern is required")
        if not callable(handler):
            raise ValueError("handler must be callable")
        with self._lock:
            self._subscribers.append((pattern, handler))
        return handler

    def unsubscribe(self, handler: Subscriber) -> int:
        """Remove every subscription for ``handler``. Returns the count."""
        with self._lock:
            before = len(self._subscribers)
            self._subscribers = [
                (pat, fn) for (pat, fn) in self._subscribers if fn is not handler
            ]
            return before - len(self._subscribers)

    def clear(self) -> None:
        """Drop every subscriber. Handy for test isolation."""
        with self._lock:
            self._subscribers = []

    # ─── Publish ─────────────────────────────────────────────────

    def publish(self, name: str, _source: str = "", **payload: Any) -> Event:
        """Publish an event. Runs every matching subscriber synchronously."""
        event = Event(name=name, payload=payload, source=_source)
        with self._lock:
            self._published_count += 1
            matched = [
                fn for (pat, fn) in self._subscribers
                if _matches(pat, name)
            ]
        for fn in matched:
            try:
                fn(event)
            except Exception as e:  # noqa: BLE001 — subscriber is user code
                log.debug("event subscriber raised on %s: %s", name, e)
        return event

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "subscribers": len(self._subscribers),
                "published": self._published_count,
                "patterns": sorted({pat for pat, _ in self._subscribers}),
            }


def _matches(pattern: str, name: str) -> bool:
    if pattern == name:
        return True
    return fnmatch.fnmatchcase(name, pattern)


# ─── Module-level singleton ─────────────────────────────────────────

bus = EventBus()


def on(pattern: str) -> Callable[[Subscriber], Subscriber]:
    """Decorator form of ``bus.subscribe``::

        @on("tool.executed")
        def handle(event): ...
    """
    def _decorator(fn: Subscriber) -> Subscriber:
        bus.subscribe(pattern, fn)
        return fn
    return _decorator


def publish(name: str, _source: str = "", **payload: Any) -> Event:
    """Publish on the default bus."""
    return bus.publish(name, _source=_source, **payload)
