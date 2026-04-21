"""Observability stub.

ollama-anvil does not ship OTLP/tracing by default. This module exposes
a no-op ``current_tracer()`` so callers (notably :mod:`anvil.agents.base`)
can call ``current_tracer().span(...)`` without a hard dependency on a
tracing backend. If a real tracer is wired in later, replace this module.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any


class _NoopSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        return

    def add_event(self, name: str, attributes: dict | None = None) -> None:
        return

    def record_exception(self, exc: BaseException) -> None:
        return

    def __enter__(self) -> _NoopSpan:
        return self

    def __exit__(self, *exc: Any) -> None:
        return


class _NoopTracer:
    @contextmanager
    def span(self, name: str, **attributes: Any):
        yield _NoopSpan()

    def event(self, name: str, **attributes: Any) -> None:
        return


_TRACER = _NoopTracer()


def current_tracer() -> _NoopTracer:
    """Return a no-op tracer. Replace module to enable real tracing."""
    return _TRACER
