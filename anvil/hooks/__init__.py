"""Hooks system — pluggable lifecycle callbacks for the agent loop.

Borrows the JSON-on-stdin contract from Claude Code so users can write
hook handlers in any language and chain them into the agent's lifecycle
without touching forge internals.

Supported events (initial set):
- ``UserPromptSubmit`` — fires just before the user's message is sent to the model.
- ``PreToolUse``       — fires before a tool function executes; can rewrite args or deny the call.
- ``PostToolUse``      — fires after a tool function returns; gets both args and result.

Config location: ``.anvil/hooks.json`` in the working directory, or
``~/.config/ollama-anvil/hooks.json`` as a user-wide fallback.

See :class:`forge.hooks.manager.HookManager` for the full contract.
"""

from anvil.hooks.manager import (
    HookConfig,
    HookDecision,
    HookEvent,
    HookManager,
    HookResult,
    load_hooks,
)

__all__ = [
    "HookConfig",
    "HookDecision",
    "HookEvent",
    "HookManager",
    "HookResult",
    "load_hooks",
]
