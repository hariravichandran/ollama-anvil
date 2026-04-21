"""Configuration profiles — named presets for different use cases.

Profiles bundle model selection, inference parameters, and behavioral
settings into named configurations that users can switch between.

Usage::

    from anvil.llm.profiles import ProfileManager, Profile

    manager = ProfileManager("~/.anvil/profiles.json")

    # Create profiles
    manager.create("fast", model="llama3.2:3b", description="Quick responses")
    manager.create("quality", model="qwen2.5:14b", description="Best quality")
    manager.create("code", model="qwen2.5-coder:14b", temperature=0.2)

    # Get a profile
    profile = manager.get("fast")
    print(profile.summary())

    # List profiles
    for p in manager.list_profiles():
        print(p.name, p.model)

    # Set active profile
    manager.set_active("quality")
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anvil.utils.logging import get_logger

log = get_logger("llm.profiles")

__all__ = [
    "Profile",
    "ProfileManager",
    "DEFAULT_PROFILES",
    "MAX_PROFILES",
    "MAX_NAME_LENGTH",
]

MAX_PROFILES = 50
MAX_NAME_LENGTH = 40


@dataclass(frozen=True, slots=True)
class Profile:
    """A named configuration profile."""

    name: str
    model: str
    description: str = ""
    temperature: float | None = None
    num_ctx: int | None = None
    system_prompt: str = ""
    agent: str = ""
    keep_alive: str = ""
    created_at: float = 0.0

    def summary(self) -> str:
        """Human-readable profile summary."""
        parts = [f"{self.name}: {self.model}"]
        if self.description:
            parts.append(f"({self.description})")
        if self.temperature is not None:
            parts.append(f"temp={self.temperature}")
        if self.num_ctx:
            parts.append(f"ctx={self.num_ctx}")
        if self.agent:
            parts.append(f"agent={self.agent}")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        d: dict[str, Any] = {"name": self.name, "model": self.model}
        if self.description:
            d["description"] = self.description
        if self.temperature is not None:
            d["temperature"] = self.temperature
        if self.num_ctx is not None:
            d["num_ctx"] = self.num_ctx
        if self.system_prompt:
            d["system_prompt"] = self.system_prompt
        if self.agent:
            d["agent"] = self.agent
        if self.keep_alive:
            d["keep_alive"] = self.keep_alive
        d["created_at"] = self.created_at
        return d

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Profile:
        """Deserialize from dict."""
        return Profile(
            name=data["name"],
            model=data["model"],
            description=data.get("description", ""),
            temperature=data.get("temperature"),
            num_ctx=data.get("num_ctx"),
            system_prompt=data.get("system_prompt", ""),
            agent=data.get("agent", ""),
            keep_alive=data.get("keep_alive", ""),
            created_at=data.get("created_at", 0.0),
        )


# Built-in profiles that are always available
DEFAULT_PROFILES: dict[str, Profile] = {
    "fast": Profile(
        name="fast",
        model="llama3.2:3b",
        description="Quick responses with a compact model",
        temperature=0.7,
        num_ctx=4096,
    ),
    "balanced": Profile(
        name="balanced",
        model="qwen2.5:7b",
        description="Good balance of speed and quality",
        temperature=0.7,
        num_ctx=8192,
    ),
    "quality": Profile(
        name="quality",
        model="qwen2.5:14b",
        description="Best quality, slower responses",
        temperature=0.7,
        num_ctx=16384,
    ),
    "code": Profile(
        name="code",
        model="qwen2.5-coder:7b",
        description="Optimized for code generation",
        temperature=0.2,
        num_ctx=8192,
        agent="coder",
    ),
    "reasoning": Profile(
        name="reasoning",
        model="deepseek-r1:7b",
        description="Chain-of-thought reasoning",
        temperature=0.6,
        num_ctx=8192,
    ),
}


class ProfileManager:
    """Manages named configuration profiles.

    Combines built-in defaults with user-created profiles. User profiles
    persist to a JSON file. Built-in profiles can be overridden.

    Args:
        path: Path to JSON file for user profiles. None for in-memory only.
    """

    def __init__(self, path: str | Path | None = None):
        self._path = Path(path).expanduser() if path else None
        self._lock = threading.Lock()
        self._user_profiles: dict[str, Profile] = {}
        self._active: str = ""

        if self._path and self._path.exists():
            self._load()

    def __repr__(self) -> str:
        total = len(DEFAULT_PROFILES) + len(self._user_profiles)
        active = f", active={self._active!r}" if self._active else ""
        return f"ProfileManager(profiles={total}{active})"

    def get(self, name: str) -> Profile | None:
        """Get a profile by name. User profiles override built-ins."""
        with self._lock:
            return self._user_profiles.get(name) or DEFAULT_PROFILES.get(name)

    def create(
        self,
        name: str,
        model: str,
        description: str = "",
        temperature: float | None = None,
        num_ctx: int | None = None,
        system_prompt: str = "",
        agent: str = "",
        keep_alive: str = "",
    ) -> Profile:
        """Create or update a user profile.

        Args:
            name: Profile name (max 40 chars, alphanumeric + hyphens).
            model: Model to use.
            description: Human-readable description.
            temperature: Generation temperature override.
            num_ctx: Context window override.
            system_prompt: Custom system prompt.
            agent: Agent name to use.
            keep_alive: Ollama keep-alive duration.

        Returns:
            The created Profile.
        """
        name = name.strip().lower()[:MAX_NAME_LENGTH]
        if not name:
            raise ValueError("Profile name cannot be empty")

        profile = Profile(
            name=name,
            model=model,
            description=description,
            temperature=temperature,
            num_ctx=num_ctx,
            system_prompt=system_prompt,
            agent=agent,
            keep_alive=keep_alive,
            created_at=time.time(),
        )

        with self._lock:
            if len(self._user_profiles) >= MAX_PROFILES and name not in self._user_profiles:
                raise ValueError(f"Maximum {MAX_PROFILES} profiles reached")
            self._user_profiles[name] = profile

        return profile

    def delete(self, name: str) -> bool:
        """Delete a user profile. Cannot delete built-in profiles."""
        with self._lock:
            if name in self._user_profiles:
                del self._user_profiles[name]
                if self._active == name:
                    self._active = ""
                return True
        return False

    def list_profiles(self) -> list[Profile]:
        """List all profiles (built-in + user), sorted by name."""
        with self._lock:
            combined = {**DEFAULT_PROFILES, **self._user_profiles}
        return sorted(combined.values(), key=lambda p: p.name)

    def set_active(self, name: str) -> bool:
        """Set the active profile."""
        profile = self.get(name)
        if not profile:
            return False
        with self._lock:
            self._active = name
        return True

    def active(self) -> Profile | None:
        """Get the currently active profile."""
        if not self._active:
            return None
        return self.get(self._active)

    def save(self) -> None:
        """Persist user profiles to disk."""
        if not self._path:
            return

        with self._lock:
            data = {
                "active": self._active,
                "profiles": {
                    name: p.to_dict()
                    for name, p in self._user_profiles.items()
                },
            }

        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(self._path)
            log.debug("Saved %d profiles to %s", len(data["profiles"]), self._path)
        except OSError as exc:
            log.warning("Failed to save profiles: %s", exc)

    def reset(self) -> None:
        """Clear all user profiles and active selection."""
        with self._lock:
            self._user_profiles.clear()
            self._active = ""

    def _load(self) -> None:
        """Load user profiles from disk."""
        if not self._path or not self._path.exists():
            return
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
            self._active = data.get("active", "")
            for name, pdata in data.get("profiles", {}).items():
                self._user_profiles[name] = Profile.from_dict(pdata)
            log.info("Loaded %d user profiles", len(self._user_profiles))
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            log.warning("Failed to load profiles from %s: %s", self._path, exc)
