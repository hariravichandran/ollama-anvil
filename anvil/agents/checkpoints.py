"""Session checkpoints — snapshot and restore agent state on demand.

Gives users a save/restore UX for long conversations: "I want to try
something risky — let me checkpoint first." Matches the Claude Code
``/resume`` and Aider ``/undo`` semantics but implemented as an
explicit named save instead of an implicit history stack, so users
can hold multiple parallel what-if branches at once.

Storage: ``<working_dir>/.anvil/checkpoints/<name>.json``. One JSON
file per checkpoint with an atomically-written schema:

    {
      "schema_version": 1,
      "name": "...",
      "agent_name": "...",
      "created_at": 1745000000.0,
      "messages": [...],
      "metadata": {...}
    }

A checkpoint only captures the conversation messages and the active
agent's identity — everything else (tool cache, circuit-breaker
counters, compressor state, hook state) is ephemeral runtime state
that should not persist across rewinds.

The :class:`CheckpointStore` exposes save / load / list / delete +
helpers for the orchestrator's ``/checkpoint`` slash command.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from anvil.utils.logging import get_logger

if TYPE_CHECKING:
    from anvil.agents.base import BaseAgent

log = get_logger("agents.checkpoints")

__all__ = [
    "Checkpoint",
    "CheckpointStore",
    "MAX_NAME_LENGTH",
    "MAX_CHECKPOINTS",
    "SCHEMA_VERSION",
]

SCHEMA_VERSION = 1

# Checkpoint names are filenames — keep them sane.
MAX_NAME_LENGTH = 64

# Upper bound on stored checkpoints — prevents a runaway user from
# filling disk with snapshots.
MAX_CHECKPOINTS = 100

# Name sanitiser: alnum + hyphen + underscore only.
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(slots=True)
class Checkpoint:
    """A serialised conversation snapshot."""

    name: str
    agent_name: str
    messages: list[dict[str, str]]
    created_at: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "agent_name": self.agent_name,
            "created_at": self.created_at,
            "messages": self.messages,
            "metadata": self.metadata,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Checkpoint:
        return cls(
            name=str(data.get("name", "")),
            agent_name=str(data.get("agent_name", "")),
            messages=list(data.get("messages", [])),
            created_at=float(data.get("created_at", 0.0)),
            metadata=dict(data.get("metadata", {})),
        )


class CheckpointStore:
    """On-disk store for named checkpoints of agent conversations."""

    def __init__(self, working_dir: str | Path = "."):
        self.root = Path(working_dir) / ".forge" / "checkpoints"

    def __repr__(self) -> str:
        return f"CheckpointStore(root={self.root})"

    # ─── Public API ──────────────────────────────────────────────

    def save(self, name: str, agent: BaseAgent) -> tuple[bool, str]:
        """Snapshot ``agent``'s state under ``name``.

        Returns ``(ok, message)``. Overwrites an existing checkpoint
        with the same name silently — callers that need atomic "create
        only" semantics should check :meth:`exists` first.
        """
        err = _validate_name(name)
        if err:
            return False, err

        if self._count() >= MAX_CHECKPOINTS and not self.exists(name):
            return False, f"Error: checkpoint cap reached ({MAX_CHECKPOINTS})"

        # Deep-copy messages so later mutation in the live agent doesn't
        # retroactively change the snapshot.
        messages = [dict(m) for m in getattr(agent, "messages", [])]
        checkpoint = Checkpoint(
            name=name,
            agent_name=getattr(agent.config, "name", "") if hasattr(agent, "config") else "",
            messages=messages,
            created_at=time.time(),
            metadata={
                "message_count": len(messages),
            },
        )

        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return False, f"Error: could not create checkpoint dir: {e}"

        path = self._path_for(name)
        payload = json.dumps(checkpoint.to_json(), indent=2, default=str)
        # Write atomically via a temp file.
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(path)
        except OSError as e:
            return False, f"Error: could not write checkpoint: {e}"
        log.info("Checkpoint saved: %s (%d messages)", name, len(messages))
        return True, f"Saved checkpoint '{name}' ({len(messages)} messages)"

    def load(self, name: str, agent: BaseAgent) -> tuple[bool, str]:
        """Restore the named checkpoint into ``agent``.

        Mutates ``agent.messages`` in place. Other runtime state (tool
        caches, circuit breaker counters, etc.) is NOT restored — we
        deliberately reset these so the replay starts clean.
        """
        err = _validate_name(name)
        if err:
            return False, err

        path = self._path_for(name)
        if not path.exists():
            return False, f"Error: no checkpoint named '{name}'"

        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
            return False, f"Error: could not read checkpoint: {e}"

        if not isinstance(data, dict):
            return False, "Error: malformed checkpoint (not an object)"

        try:
            checkpoint = Checkpoint.from_json(data)
        except (TypeError, ValueError) as e:
            return False, f"Error: malformed checkpoint: {e}"

        agent.messages = [dict(m) for m in checkpoint.messages]
        # Reset runtime-only state so rewound sessions start fresh.
        if hasattr(agent, "_tool_cache"):
            agent._tool_cache = {}
        if hasattr(agent, "_tool_failure_counts"):
            agent._tool_failure_counts = {}
        if hasattr(agent, "_tool_history"):
            agent._tool_history = []

        log.info("Checkpoint loaded: %s (%d messages)", name, len(checkpoint.messages))
        return True, (
            f"Loaded checkpoint '{name}' ({len(checkpoint.messages)} messages, "
            f"saved {_age(checkpoint.created_at)})"
        )

    def list(self) -> list[Checkpoint]:
        """Return every saved checkpoint, newest first."""
        if not self.root.is_dir():
            return []
        out: list[Checkpoint] = []
        for path in self.root.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
                log.debug("Skipping malformed checkpoint %s: %s", path, e)
                continue
            if not isinstance(data, dict):
                continue
            try:
                out.append(Checkpoint.from_json(data))
            except (TypeError, ValueError):
                continue
        out.sort(key=lambda c: c.created_at, reverse=True)
        return out

    def delete(self, name: str) -> tuple[bool, str]:
        err = _validate_name(name)
        if err:
            return False, err
        path = self._path_for(name)
        if not path.exists():
            return False, f"Error: no checkpoint named '{name}'"
        try:
            path.unlink()
        except OSError as e:
            return False, f"Error: could not delete checkpoint: {e}"
        return True, f"Deleted checkpoint '{name}'"

    def exists(self, name: str) -> bool:
        if _validate_name(name):
            return False
        return self._path_for(name).exists()

    def format_list(self) -> str:
        """Human-readable list used by the orchestrator slash command."""
        items = self.list()
        if not items:
            return "No checkpoints saved. Use /checkpoint save <name> to create one."
        lines = ["Saved checkpoints (newest first):"]
        for c in items:
            lines.append(
                f"  {c.name:24s} agent={c.agent_name:14s} "
                f"messages={c.metadata.get('message_count', len(c.messages)):>4}  "
                f"{_age(c.created_at)}"
            )
        return "\n".join(lines)

    # ─── Internals ───────────────────────────────────────────────

    def _path_for(self, name: str) -> Path:
        return self.root / f"{name}.json"

    def _count(self) -> int:
        if not self.root.is_dir():
            return 0
        return sum(1 for _ in self.root.glob("*.json"))


def _validate_name(name: str) -> str:
    """Return an error message for an invalid name, or empty string."""
    if not name:
        return "Error: checkpoint name is required"
    if len(name) > MAX_NAME_LENGTH:
        return f"Error: checkpoint name too long (max {MAX_NAME_LENGTH})"
    if not _NAME_RE.match(name):
        return "Error: checkpoint name must be alphanumeric (hyphen + underscore allowed)"
    return ""


def _age(ts: float) -> str:
    """Format ``ts`` as 'Ns ago' / 'Nm ago' / 'Nh ago' / 'Nd ago'."""
    if ts <= 0:
        return "unknown"
    delta = max(0.0, time.time() - ts)
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    return f"{int(delta // 86400)}d ago"
