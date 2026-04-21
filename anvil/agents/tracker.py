"""Agent tracker: manage and monitor single-agent and multi-agent systems.

Tracks agent configurations, conversation histories, and performance metrics.
Users can create, list, switch between, and monitor their agent systems.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from anvil.utils.fileio import atomic_write
from anvil.utils.logging import get_logger

log = get_logger("agents.tracker")

__all__ = [
    "AgentSystemInfo",
    "AgentTracker",
    "MAX_SYSTEM_NAME_LENGTH",
    "MAX_DESCRIPTION_LENGTH",
    "MAX_AGENTS_PER_SYSTEM",
    "VALID_SYSTEM_TYPES",
    "NAME_COLUMN_WIDTH",
]

# Validation limits
MAX_SYSTEM_NAME_LENGTH = 100
MAX_DESCRIPTION_LENGTH = 500
MAX_AGENTS_PER_SYSTEM = 20
VALID_SYSTEM_TYPES = {"single", "multi"}

# Display
NAME_COLUMN_WIDTH = 25


@dataclass(slots=True)
class AgentSystemInfo:
    """Metadata for a tracked agent system."""

    name: str
    system_type: str  # "single" or "multi"
    agents: list[str]  # agent names in this system
    description: str = ""
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    total_messages: int = 0
    total_tool_calls: int = 0


class AgentTracker:
    """Tracks and manages agent systems (single-agent and multi-agent).

    Provides:
    - Create/delete tracked systems
    - List all systems with status
    - Monitor performance metrics
    - Persist system configurations
    """

    def __init__(self, state_dir: str = ".anvil_state"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.tracker_file = self.state_dir / "agent_systems.json"
        self.systems: dict[str, AgentSystemInfo] = self._load()

    def close(self) -> None:
        """Persist state and release resources."""
        self._save()

    def __enter__(self) -> AgentTracker:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def create_system(
        self,
        name: str,
        system_type: str,
        agents: list[str],
        description: str = "",
    ) -> str:
        """Register a new agent system."""
        if not name or not name.strip():
            return "System name cannot be empty"
        if len(name) > MAX_SYSTEM_NAME_LENGTH:
            return f"System name too long (max {MAX_SYSTEM_NAME_LENGTH} chars)"
        if system_type not in VALID_SYSTEM_TYPES:
            return f"Invalid system_type '{system_type}'. Must be one of: {', '.join(sorted(VALID_SYSTEM_TYPES))}"
        if not agents:
            return "At least one agent is required"
        if len(agents) > MAX_AGENTS_PER_SYSTEM:
            return f"Too many agents (max {MAX_AGENTS_PER_SYSTEM})"
        if len(description) > MAX_DESCRIPTION_LENGTH:
            description = description[:MAX_DESCRIPTION_LENGTH]
        if name in self.systems:
            return f"System '{name}' already exists"

        self.systems[name] = AgentSystemInfo(
            name=name,
            system_type=system_type,
            agents=agents,
            description=description,
        )
        self._save()
        return f"Created {system_type}-agent system: {name} ({', '.join(agents)})"

    def delete_system(self, name: str) -> str:
        """Remove a tracked system."""
        if name not in self.systems:
            return f"System '{name}' not found"
        del self.systems[name]
        self._save()
        return f"Deleted system: {name}"

    def record_activity(self, system_name: str, messages: int = 0, tool_calls: int = 0) -> None:
        """Record activity for a system."""
        messages = max(0, messages)
        tool_calls = max(0, tool_calls)
        system = self.systems.get(system_name)
        if system:
            system.last_active = time.time()
            system.total_messages += messages
            system.total_tool_calls += tool_calls
            self._save()

    def list_systems(self) -> str:
        """List all tracked agent systems."""
        if not self.systems:
            return "No agent systems tracked yet. Create one with 'forge agent create'."

        lines = ["Agent Systems:\n"]
        for sys in sorted(self.systems.values(), key=lambda s: s.last_active, reverse=True):
            type_icon = "[single]" if sys.system_type == "single" else "[multi] "
            agents_str = ", ".join(sys.agents)
            lines.append(f"  {type_icon} {sys.name:{NAME_COLUMN_WIDTH}s} agents: {agents_str}")
            if sys.description:
                lines.append(f"           {sys.description}")
            lines.append(f"           msgs: {sys.total_messages}  tools: {sys.total_tool_calls}")
            lines.append("")

        return "\n".join(lines)

    def get_system(self, name: str) -> AgentSystemInfo | None:
        """Get info for a specific system."""
        return self.systems.get(name)

    def _load(self) -> dict[str, AgentSystemInfo]:
        if not self.tracker_file.exists():
            return {}
        try:
            data = json.loads(self.tracker_file.read_text(encoding="utf-8"))
            systems = {}
            known = {f.name for f in fields(AgentSystemInfo)}
            for name, info in data.items():
                systems[name] = AgentSystemInfo(**{k: v for k, v in info.items() if k in known})
            return systems
        except (json.JSONDecodeError, TypeError, OSError) as e:
            log.warning("Could not load agent tracker data from %s: %s", self.tracker_file, e)
            return {}

    def _save(self) -> None:
        data = {name: asdict(sys) for name, sys in self.systems.items()}
        content = json.dumps(data, indent=2)
        try:
            atomic_write(self.tracker_file, content)
        except OSError as e:
            log.error("Could not save agent tracker data to %s: %s", self.tracker_file, e)
