"""Model alias manager for creating short names for long model identifiers.

Aliases map user-friendly short names (e.g., 'q', 'code', 'think') to full
Ollama model names (e.g., 'qwen2.5:7b-instruct'). Stored as JSON in
~/.anvil/aliases.json.

Usage:
    from anvil.llm.aliases import AliasManager
    mgr = AliasManager()
    mgr.set("q", "qwen2.5:7b-instruct")
    full = mgr.resolve("q")  # -> "qwen2.5:7b-instruct"
"""

from __future__ import annotations

import json
from pathlib import Path

from anvil.utils.fileio import atomic_write
from anvil.utils.logging import get_logger

log = get_logger("llm.aliases")

__all__ = [
    "AliasManager",
    "ALIASES_FILE",
    "MAX_ALIAS_LENGTH",
    "MAX_ALIASES",
]

# Persistence path
ALIASES_FILE = Path.home() / ".anvil" / "aliases.json"

# Safety limits
MAX_ALIAS_LENGTH = 30
MAX_ALIASES = 100


class AliasManager:
    """Manage short aliases for model names.

    Aliases are persisted to ~/.anvil/aliases.json and auto-loaded on init.
    """

    def __init__(self, aliases_file: Path | None = None):
        self._file = aliases_file or ALIASES_FILE
        self._aliases: dict[str, str] = {}
        self._load()

    def __repr__(self) -> str:
        return f"AliasManager(aliases={len(self._aliases)}, file={self._file!r})"

    def _load(self) -> None:
        """Load aliases from disk."""
        if not self._file.exists():
            return
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._aliases = {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Failed to load aliases from %s: %s", self._file, e)

    def save(self) -> None:
        """Persist aliases to disk."""
        self._file.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(self._file, json.dumps(self._aliases, indent=2))

    def set(self, alias: str, model: str) -> str:
        """Create or update an alias.

        Args:
            alias: Short name (max 30 chars, alphanumeric + hyphens/underscores).
            model: Full Ollama model name.

        Returns:
            Success or error message.
        """
        alias = alias.strip().lower()
        model = model.strip()

        if not alias:
            return "Alias cannot be empty."
        if len(alias) > MAX_ALIAS_LENGTH:
            return f"Alias too long (max {MAX_ALIAS_LENGTH} chars)."
        if not alias.replace("-", "").replace("_", "").replace(".", "").isalnum():
            return "Alias must be alphanumeric (hyphens, underscores, dots allowed)."
        if not model:
            return "Model name cannot be empty."
        if len(self._aliases) >= MAX_ALIASES and alias not in self._aliases:
            return f"Too many aliases (max {MAX_ALIASES}). Remove some first."

        self._aliases[alias] = model
        self.save()
        return f"Alias '{alias}' → {model}"

    def remove(self, alias: str) -> str:
        """Remove an alias.

        Returns:
            Success or error message.
        """
        alias = alias.strip().lower()
        if alias in self._aliases:
            old = self._aliases.pop(alias)
            self.save()
            return f"Removed alias '{alias}' (was → {old})"
        return f"Alias '{alias}' not found."

    def resolve(self, name: str) -> str:
        """Resolve a model name or alias to the full model name.

        If name is a known alias, returns the mapped model name.
        Otherwise, returns the name as-is (pass-through).
        """
        return self._aliases.get(name.strip().lower(), name)

    def list_aliases(self) -> dict[str, str]:
        """Return all aliases as a dict."""
        return dict(self._aliases)
