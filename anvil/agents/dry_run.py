"""Dry-run / preview mode — lets agents plan destructive work safely.

When ``dry_run=True`` every tool call whose function name looks
destructive (``write_*``, ``edit_*``, ``delete_*``, ``run_*``,
``commit_*``, ``push_*``, ``transfer_to``, etc.) is intercepted before
it reaches the tool. Instead of running, we return a synthetic result
that describes *what would have happened*. The LLM sees the preview
and continues its plan — perfect for "draft this refactor, show me
what you'd do, then I'll say go."

Read-only operations (``read_*``, ``list_*``, ``find_*``, ``search_*``,
``git_status``, ``web_search``, ``sql_query`` on an in-memory engine,
etc.) still run normally so the agent can explore the codebase.

The preview result follows a predictable shape so downstream tools
(hooks, tracing) can distinguish it from a real tool result::

    [DRY-RUN] write_file(...): no changes made. Arguments:
      path: forge/x.py
      content: (512 chars omitted)

The decision of what counts as destructive is an allowlist of name
*prefixes* — simple, transparent, overridable per agent via
:attr:`DryRunConfig.destructive_prefixes` and the companion exact-name
sets. No model inference or regex matching is involved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DryRunConfig",
    "is_destructive",
    "build_preview",
    "DEFAULT_DESTRUCTIVE_PREFIXES",
    "DEFAULT_DESTRUCTIVE_NAMES",
    "DEFAULT_SAFE_NAMES",
    "MAX_ARG_DISPLAY_CHARS",
]

# Function-name prefixes that mark a tool call as destructive.
DEFAULT_DESTRUCTIVE_PREFIXES = frozenset({
    "write_", "edit_", "append_", "delete_",
    "run_", "exec_", "commit_", "push_",
    "mkdir_", "mv_", "rm_", "install_", "uninstall_",
})

# Exact names that are destructive but don't match a common prefix.
DEFAULT_DESTRUCTIVE_NAMES = frozenset({
    "fuzzy_edit",
    "transfer_to",         # handoff swaps active agent
    "save_skill",          # writes to .anvil/skills/
    "update_skill",
    "sql_register_csv",    # mutates SQL tool state (still usually safe though)
})

# Exact names that look destructive by prefix but are *safe* in practice.
# We override the prefix test for these so the agent can still explore.
DEFAULT_SAFE_NAMES = frozenset({
    "run_tests",           # diagnostic — commonly expected to be dry-run-safe
    "edit_view",           # hypothetical read-only sibling
})

# How much of each argument to show in the preview before truncating.
MAX_ARG_DISPLAY_CHARS = 240


@dataclass(slots=True)
class DryRunConfig:
    """Knob-set for dry-run behaviour.

    Defaults cover the common tools. Agents that want to loosen or
    tighten scope pass a custom config to BaseAgent.
    """

    enabled: bool = True
    destructive_prefixes: frozenset[str] = field(default_factory=lambda: DEFAULT_DESTRUCTIVE_PREFIXES)
    destructive_names: frozenset[str] = field(default_factory=lambda: DEFAULT_DESTRUCTIVE_NAMES)
    safe_names: frozenset[str] = field(default_factory=lambda: DEFAULT_SAFE_NAMES)

    def is_destructive(self, function_name: str) -> bool:
        return is_destructive(
            function_name,
            prefixes=self.destructive_prefixes,
            names=self.destructive_names,
            safe=self.safe_names,
        )


def is_destructive(
    function_name: str,
    prefixes: frozenset[str] = DEFAULT_DESTRUCTIVE_PREFIXES,
    names: frozenset[str] = DEFAULT_DESTRUCTIVE_NAMES,
    safe: frozenset[str] = DEFAULT_SAFE_NAMES,
) -> bool:
    """Return True if ``function_name`` should be intercepted by dry-run.

    Precedence: explicit safe_names win over everything (so ``run_tests``
    stays safe even though it starts with ``run_``). Then destructive
    names, then destructive prefixes.
    """
    if not function_name:
        return False
    if function_name in safe:
        return False
    if function_name in names:
        return True
    return any(function_name.startswith(prefix) for prefix in prefixes)


def build_preview(function_name: str, args: dict[str, Any]) -> str:
    """Render a [DRY-RUN] preview string for a tool call.

    Values that exceed :data:`MAX_ARG_DISPLAY_CHARS` are summarised as
    "(N chars omitted)" so the LLM sees structure without context bloat.
    """
    lines = [f"[DRY-RUN] {function_name}(...): no changes made. Arguments:"]
    if not isinstance(args, dict) or not args:
        lines.append("  (no arguments)")
        return "\n".join(lines)

    for key, value in args.items():
        rendered = _render_value(value)
        lines.append(f"  {key}: {rendered}")
    return "\n".join(lines)


def _render_value(value: Any) -> str:
    """Render one arg value, truncating long strings."""
    if isinstance(value, str):
        if len(value) > MAX_ARG_DISPLAY_CHARS:
            return f"({len(value)} chars omitted)"
        return value
    if isinstance(value, (list, tuple)):
        # Render the container's length, not its full contents.
        if len(value) > 8:
            return f"[{len(value)} items]"
        return repr(value)
    return repr(value)
