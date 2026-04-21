"""Model-context switching — temporarily swap an LLM client's active model.

Extracted from the ``original = client.model; client.switch_model(new);
try: ... finally: client.switch_model(original)`` pattern in
:mod:`anvil.llm.consensus`, :mod:`anvil.llm.expert_panel`,
:mod:`anvil.llm.best_of_n`, :mod:`anvil.llm.structured`, and
:mod:`anvil.tools.vision_ocr` — five callers, five subtly-different
implementations of the same dance.

Usage::

    from anvil.utils.model_context import use_model

    with use_model(client, "qwen2.5:14b"):
        result = client.chat(...)

Guarantees:

* If ``client`` has no ``switch_model`` method, does nothing (returns the
  client as-is so no-op callers still work).
* If the switch fails, does not raise — logs at debug and leaves the
  client on its original model.
* Always restores the original model on exit, even under exception.

Pair with :func:`switch_model_safe` when you need the bare switch without
a context manager (e.g. one-shot dispatch with no later restore).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from anvil.utils.logging import get_logger

log = get_logger("model_context")

__all__ = ["use_model", "switch_model_safe"]


@contextmanager
def use_model(client: Any, model: str | None) -> Iterator[Any]:
    """Temporarily run ``client`` on ``model`` for the context body.

    ``model=None`` or an absent ``switch_model`` method → no-op. Always
    restores the pre-context model on exit.
    """
    if not model or not hasattr(client, "switch_model"):
        yield client
        return

    original = getattr(client, "model", None)
    # Skip the switch entirely when the client is already on the target
    # — the server-side reload on Ollama is expensive and our tests
    # encode this as an invariant.
    if original == model:
        yield client
        return

    switched = switch_model_safe(client, model)
    try:
        yield client
    finally:
        if switched and original is not None:
            switch_model_safe(client, original)


def switch_model_safe(client: Any, model: str) -> bool:
    """Call ``client.switch_model(model)``, swallowing any failure.

    Returns ``True`` when the switch succeeded. On failure (method missing,
    raised exception, returned falsy), returns ``False`` without raising.
    """
    if not hasattr(client, "switch_model"):
        return False
    try:
        result = client.switch_model(model)
    except Exception as e:  # noqa: BLE001 — client is user code
        log.debug("switch_model(%s) raised: %s", model, e)
        return False
    return bool(result) if result is not None else True
