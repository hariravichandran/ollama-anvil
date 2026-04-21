"""HookManager — dispatches lifecycle events to user-configured handlers.

Design goals, borrowed from Claude Code:

- **Users write hooks in any language.** Each handler receives a JSON
  payload on stdin and may return JSON on stdout to influence the event.
- **No framework lock-in.** Hooks are shell commands. Forge does not care
  whether the script is Python, Bash, Node, Go — just that it exits.
- **Matchers pick which calls trigger a hook.** For tool events a matcher
  is matched against the tool function name (exact string, regex, or
  ``"*"`` wildcard). For non-tool events the matcher is ignored.
- **Non-response = allow.** If a hook exits 0 and prints nothing, we
  treat it as a pass-through — perfect for logging / telemetry.
- **Hooks can rewrite inputs.** A ``PreToolUse`` hook can return
  ``{"updatedInput": {...}}`` to substitute the args the tool will see.
- **Hooks can block.** Returning ``{"decision": "deny", "message": ...}``
  short-circuits the tool call and surfaces the message to the model.

Config JSON shape::

    {
      "PreToolUse": [
        {"matcher": "run_command", "type": "command", "command": "/p/h.sh", "timeout": 5}
      ],
      "PostToolUse": [...],
      "UserPromptSubmit": [...]
    }

Search order for the config file:

1. ``<working_dir>/.anvil/hooks.json`` (project-scoped).
2. ``$XDG_CONFIG_HOME/ollama-anvil/hooks.json`` (user-scoped).
3. ``~/.config/ollama-anvil/hooks.json`` (user-scoped fallback).

The first file found wins; they are not merged. A project-scoped config
is loud-and-clear opt-in per-repo, matching how ``.claude/settings.json``
works in Claude Code.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from anvil.utils.logging import get_logger

log = get_logger("hooks")

__all__ = [
    "HookConfig",
    "HookDecision",
    "HookEvent",
    "HookManager",
    "HookResult",
    "load_hooks",
    "DEFAULT_HOOK_TIMEOUT",
    "MAX_HOOK_OUTPUT_BYTES",
    "MAX_HOOK_HANDLERS",
]

# How long a single hook handler may run before we SIGKILL it.
DEFAULT_HOOK_TIMEOUT = 30.0

# Max stdout bytes we will read from a hook before truncating.
MAX_HOOK_OUTPUT_BYTES = 64 * 1024

# Upper bound on the number of handlers loaded per event, to keep startup fast.
MAX_HOOK_HANDLERS = 32


class HookEvent(str, Enum):
    """Lifecycle events the HookManager can dispatch.

    Subclassing ``str`` makes the enum JSON-serialisable and allows
    comparison against raw strings in user config files.
    """

    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"


class HookDecision(str, Enum):
    """How the agent should react to a PreToolUse hook's response."""

    ALLOW = "allow"      # Proceed with the tool call.
    DENY = "deny"        # Short-circuit; return the hook's message to the model.
    ASK = "ask"          # Fall through to the normal permission check.


@dataclass(slots=True)
class HookConfig:
    """A single configured hook handler."""

    event: HookEvent
    matcher: str = "*"          # Only used for tool events; regex or literal.
    type: str = "command"       # Only "command" is supported today.
    command: str = ""
    timeout: float = DEFAULT_HOOK_TIMEOUT

    def matches_tool(self, tool_name: str) -> bool:
        """Return True if this hook should fire for ``tool_name``.

        Uses exact match if possible, regex otherwise, with ``*``
        interpreted as "match everything".
        """
        if self.matcher in ("*", ""):
            return True
        if self.matcher == tool_name:
            return True
        try:
            return re.fullmatch(self.matcher, tool_name) is not None
        except re.error:
            log.debug("Invalid matcher regex %r; falling back to literal", self.matcher)
            return False


@dataclass(slots=True)
class HookResult:
    """Outcome of running a single hook handler."""

    ok: bool
    decision: HookDecision = HookDecision.ALLOW
    updated_input: dict[str, Any] | None = None
    message: str = ""
    elapsed_s: float = 0.0
    raw_stdout: str = ""
    raw_stderr: str = ""


def load_hooks(working_dir: str | Path = ".") -> list[HookConfig]:
    """Load hook configs from the first config file found.

    Returns an empty list when no config file exists — hooks are
    entirely opt-in.
    """
    candidates: list[Path] = [
        Path(working_dir) / ".forge" / "hooks.json",
    ]
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        candidates.append(Path(xdg) / "ollama-anvil" / "hooks.json")
    candidates.append(Path.home() / ".config" / "ollama-anvil" / "hooks.json")

    for path in candidates:
        if path.exists():
            try:
                return _parse_hooks_file(path)
            except (OSError, ValueError, json.JSONDecodeError) as e:
                log.warning("Failed to load hooks from %s: %s", path, e)
                return []
    return []


def _parse_hooks_file(path: Path) -> list[HookConfig]:
    """Parse a hooks.json file into a list of HookConfig entries."""
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"hooks.json at {path} must be a JSON object")

    configs: list[HookConfig] = []
    for event_name, entries in raw.items():
        try:
            event = HookEvent(event_name)
        except ValueError:
            log.warning("Ignoring unknown hook event %r in %s", event_name, path)
            continue
        if not isinstance(entries, list):
            log.warning("Hook event %s entries must be a list; skipping", event_name)
            continue
        for entry in entries[:MAX_HOOK_HANDLERS]:
            if not isinstance(entry, dict):
                continue
            command = str(entry.get("command", "")).strip()
            if not command:
                log.warning("Hook entry for %s has no command; skipping", event_name)
                continue
            try:
                timeout = float(entry.get("timeout", DEFAULT_HOOK_TIMEOUT))
            except (TypeError, ValueError):
                timeout = DEFAULT_HOOK_TIMEOUT
            configs.append(HookConfig(
                event=event,
                matcher=str(entry.get("matcher", "*")),
                type=str(entry.get("type", "command")),
                command=command,
                timeout=timeout,
            ))
    return configs


@dataclass(slots=True)
class HookManager:
    """Runtime dispatcher for hooks.

    Instantiate once per session. Call ``run(event, payload, tool=...)``
    at the relevant points in the agent loop. The manager walks the
    registered handlers in config order and returns the most restrictive
    decision — a single ``deny`` wins over later ``allow`` responses.
    """

    configs: list[HookConfig] = field(default_factory=list)
    session_id: str = ""

    def matching(self, event: HookEvent, tool_name: str = "") -> list[HookConfig]:
        """Return the subset of configs that fire for this event+tool."""
        return [
            c for c in self.configs
            if c.event == event and c.matches_tool(tool_name)
        ]

    def run(
        self,
        event: HookEvent,
        payload: dict[str, Any],
        tool_name: str = "",
    ) -> HookResult:
        """Dispatch ``event`` to all matching handlers.

        Returns the *first* denying result, or the last allowing result
        after applying any ``updatedInput`` rewrites in sequence.
        """
        matched = self.matching(event, tool_name)
        if not matched:
            return HookResult(ok=True, decision=HookDecision.ALLOW)

        payload = dict(payload)  # defensive copy
        payload.setdefault("event", event.value)
        payload.setdefault("session_id", self.session_id)

        # Run handlers in order. If any denies, bail out; otherwise carry
        # forward updatedInput rewrites so later hooks see the merged view.
        final = HookResult(ok=True, decision=HookDecision.ALLOW)
        for cfg in matched:
            result = _run_single(cfg, payload)
            if result.updated_input is not None:
                # Merge into the payload so later handlers see the rewrite.
                if isinstance(payload.get("tool"), dict):
                    payload["tool"]["arguments"] = result.updated_input
                # Preserve the latest rewrite so the caller can pick it up.
                final.updated_input = result.updated_input
            if result.message:
                final.message = result.message
            if result.decision == HookDecision.DENY:
                # First deny wins.
                return result
            final.ok = final.ok and result.ok
            final.elapsed_s += result.elapsed_s

        return final


def _run_single(cfg: HookConfig, payload: dict[str, Any]) -> HookResult:
    """Execute one hook handler and parse its response."""
    if cfg.type != "command":
        log.warning("Unsupported hook type %r; skipping", cfg.type)
        return HookResult(ok=True, decision=HookDecision.ALLOW)

    try:
        argv = shlex.split(cfg.command)
    except ValueError as e:
        log.warning("Invalid hook command %r: %s", cfg.command, e)
        return HookResult(ok=False, decision=HookDecision.ALLOW, message=str(e))

    start = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=cfg.timeout,
            check=False,
        )
    except FileNotFoundError as e:
        log.warning("Hook command not found: %s", e)
        return HookResult(ok=False, decision=HookDecision.ALLOW, message=str(e))
    except subprocess.TimeoutExpired:
        log.warning("Hook command %r exceeded timeout %.1fs", cfg.command, cfg.timeout)
        return HookResult(
            ok=False, decision=HookDecision.ALLOW,
            message=f"Hook {cfg.command!r} timed out",
            elapsed_s=cfg.timeout,
        )

    elapsed = time.monotonic() - start
    stdout = (proc.stdout or "")[:MAX_HOOK_OUTPUT_BYTES]
    stderr = (proc.stderr or "")[:MAX_HOOK_OUTPUT_BYTES]

    # Non-zero exit = treat as deny, using stderr as the message. This is
    # a convenient "fail = reject" contract for quick shell one-liners.
    if proc.returncode != 0:
        return HookResult(
            ok=False,
            decision=HookDecision.DENY,
            message=stderr.strip() or f"hook exit {proc.returncode}",
            elapsed_s=elapsed,
            raw_stdout=stdout,
            raw_stderr=stderr,
        )

    # Empty stdout = pass-through allow.
    if not stdout.strip():
        return HookResult(ok=True, decision=HookDecision.ALLOW, elapsed_s=elapsed, raw_stdout=stdout)

    # Parse JSON response. Tolerate garbage — a broken hook should not
    # crash the agent; we just log and treat it as allow.
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        log.debug("Hook %r produced non-JSON output; treating as allow", cfg.command)
        return HookResult(ok=True, decision=HookDecision.ALLOW, elapsed_s=elapsed, raw_stdout=stdout)

    if not isinstance(data, dict):
        return HookResult(ok=True, decision=HookDecision.ALLOW, elapsed_s=elapsed, raw_stdout=stdout)

    decision_str = str(data.get("decision", "allow")).lower()
    try:
        decision = HookDecision(decision_str)
    except ValueError:
        decision = HookDecision.ALLOW

    updated_input = data.get("updatedInput")
    if updated_input is not None and not isinstance(updated_input, dict):
        updated_input = None

    return HookResult(
        ok=True,
        decision=decision,
        updated_input=updated_input,
        message=str(data.get("message", "")),
        elapsed_s=elapsed,
        raw_stdout=stdout,
        raw_stderr=stderr,
    )
