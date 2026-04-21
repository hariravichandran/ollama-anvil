"""Context budget — live token / context telemetry for the UI.

Gives both the TUI and the web UI a simple, non-blocking way to answer
"how much of my context window have I used?" without asking the model
to tokenise itself. Uses a character-based heuristic (``chars / 4``)
when no tokeniser is available, which matches OpenAI's
rule-of-thumb and is within ~10% for English prose. For code-heavy
conversations we also offer a more conservative estimate (``chars / 3``)
that surfaces via the ``code_bias`` flag.

What's exposed:

- :class:`BudgetInfo` — model, context used / limit / percent, message
  counts, estimated tokens.
- :func:`compute_budget(agent, client=None, profile=None)` — pure
  function that reads the live agent + client state.
- :func:`format_budget(info)` — renders a one-line summary for the TUI.

Not persisted. Not rate-limited. Callers decide how often to sample
(e.g., the web UI polls via ``/api/budget`` every few seconds).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anvil.agents.base import BaseAgent
    from anvil.llm.client import OllamaClient

__all__ = [
    "BudgetInfo",
    "compute_budget",
    "format_budget",
    "estimate_tokens",
    "CHARS_PER_TOKEN",
    "CHARS_PER_TOKEN_CODE",
    "DEFAULT_CONTEXT_LIMIT",
]

# Rough tokens/char multipliers for English prose vs code-heavy content.
CHARS_PER_TOKEN = 4.0
CHARS_PER_TOKEN_CODE = 3.0

# Fallback context limit when neither the client nor a hardware profile
# tells us anything useful. 8K is the smallest common window we expect
# a user to actually be running.
DEFAULT_CONTEXT_LIMIT = 8192


@dataclass(slots=True)
class BudgetInfo:
    """Snapshot of the agent's current context utilisation."""

    model: str = ""
    agent: str = ""
    messages: int = 0
    context_used_tokens: int = 0
    context_limit_tokens: int = DEFAULT_CONTEXT_LIMIT
    context_percent: float = 0.0
    tool_calls: int = 0
    tool_cache_hits: int = 0
    tool_cache_misses: int = 0
    code_bias: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "agent": self.agent,
            "messages": self.messages,
            "context_used_tokens": self.context_used_tokens,
            "context_limit_tokens": self.context_limit_tokens,
            "context_percent": self.context_percent,
            "tool_calls": self.tool_calls,
            "tool_cache_hits": self.tool_cache_hits,
            "tool_cache_misses": self.tool_cache_misses,
            "code_bias": self.code_bias,
        }


def estimate_tokens(text: str, code_bias: bool = False) -> int:
    """Rough character-based token estimate.

    Returns ``int(len(text) / ratio)``. ``code_bias=True`` uses a
    denser ratio that covers Python / JS where long identifiers and
    heavy punctuation inflate the token count. This is a live-display
    estimate — off by up to ~15% — and deliberately cheap.
    """
    if not text:
        return 0
    ratio = CHARS_PER_TOKEN_CODE if code_bias else CHARS_PER_TOKEN
    return int(len(text) / ratio)


def _sum_message_chars(messages: list[dict[str, Any]]) -> int:
    """Total characters across every message's 'content'."""
    total = 0
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, str):
            total += len(content)
    return total


def _looks_like_code(messages: list[dict[str, Any]]) -> bool:
    """Cheap heuristic: triple-backticks present anywhere in history?"""
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, str) and "```" in content:
            return True
    return False


def compute_budget(
    agent: BaseAgent | None,
    client: OllamaClient | None = None,
    context_limit: int | None = None,
) -> BudgetInfo:
    """Compute the current context utilisation for ``agent``.

    Pulls ``context_limit`` from (in order):

    1. The explicit ``context_limit`` argument.
    2. The agent's ``config.max_context`` if set.
    3. The client's ``num_ctx`` attribute (set by OllamaClient).
    4. :data:`DEFAULT_CONTEXT_LIMIT`.
    """
    if agent is None:
        return BudgetInfo(
            context_limit_tokens=context_limit or DEFAULT_CONTEXT_LIMIT,
        )

    messages = list(getattr(agent, "messages", []) or [])
    code_bias = _looks_like_code(messages)
    tokens = estimate_tokens("".join(
        m["content"] for m in messages
        if isinstance(m, dict) and isinstance(m.get("content"), str)
    ), code_bias=code_bias)

    limit = context_limit
    if limit is None:
        limit = getattr(getattr(agent, "config", None), "max_context", 0) or 0
    if not limit and client is not None:
        limit = getattr(client, "num_ctx", 0) or 0
    if not limit:
        limit = DEFAULT_CONTEXT_LIMIT

    percent = (tokens / limit) * 100.0 if limit else 0.0

    return BudgetInfo(
        model=getattr(client, "model", "") if client is not None else "",
        agent=getattr(getattr(agent, "config", None), "name", ""),
        messages=len(messages),
        context_used_tokens=tokens,
        context_limit_tokens=int(limit),
        context_percent=round(percent, 1),
        tool_calls=int(getattr(agent, "_tool_call_count", 0)),
        tool_cache_hits=int(getattr(agent, "_tool_cache_hits", 0)),
        tool_cache_misses=int(getattr(agent, "_tool_cache_misses", 0)),
        code_bias=code_bias,
    )


def format_budget(info: BudgetInfo) -> str:
    """Render a compact one-line status string.

    Example output::

        qwen2.5:14b · data-engineer · 3,200/8,192 tok (39.1%) · 12 msgs · 4 tool calls
    """
    parts = []
    if info.model:
        parts.append(info.model)
    if info.agent:
        parts.append(info.agent)
    parts.append(
        f"{info.context_used_tokens:,}/{info.context_limit_tokens:,} tok "
        f"({info.context_percent:.1f}%)"
    )
    parts.append(f"{info.messages} msgs")
    if info.tool_calls:
        parts.append(f"{info.tool_calls} tool calls")
    return " · ".join(parts)
