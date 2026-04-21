"""Environment variable loading from .env files."""

from __future__ import annotations

import os
from pathlib import Path

from anvil.utils.logging import get_logger

log = get_logger("utils.env")


def load_env(env_path: Path | None = None) -> dict[str, str]:
    """Load environment variables from a .env file.

    Uses setdefault so existing env vars are not overwritten.
    Returns a dict of all variables that were loaded.

    Features:
    - Strips inline comments (# after value)
    - Warns about duplicate keys
    - Handles encoding errors gracefully
    - Supports 'export KEY=VALUE' syntax
    """
    if env_path is None:
        # Walk up from cwd looking for .env
        env_path = _find_env_file()
    if env_path is None or not env_path.exists():
        log.debug("No .env file found")
        return {}

    try:
        content = env_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        log.warning("Cannot read .env file %s: %s", env_path, e)
        return {}

    loaded: dict[str, str] = {}
    seen_keys: dict[str, int] = {}  # key → line number for duplicate detection

    for line_num, line in enumerate(content.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        # Support 'export KEY=VALUE' syntax
        if line.startswith("export "):
            line = line[7:].strip()

        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue

        value = value.strip()

        # Strip surrounding quotes (need at least 2 chars for valid quoting)
        if len(value) >= 2 and (
            (value.startswith('"') and value.endswith('"')) or
            (value.startswith("'") and value.endswith("'"))
        ):
            value = value[1:-1]
        else:
            # Strip inline comments (only for unquoted values)
            comment_idx = value.find(" #")
            if comment_idx >= 0:
                value = value[:comment_idx].rstrip()

        # Warn about duplicate keys
        if key in seen_keys:
            log.debug("Duplicate key '%s' in .env (lines %d and %d), using last value",
                      key, seen_keys[key], line_num)
        seen_keys[key] = line_num

        os.environ.setdefault(key, value)
        loaded[key] = value

    return loaded


def _find_env_file() -> Path | None:
    """Walk up from cwd looking for a .env file."""
    current = Path.cwd()
    for _ in range(10):  # max 10 levels up
        candidate = current / ".env"
        if candidate.exists():
            return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def get_env(key: str, default: str = "") -> str:
    """Get an environment variable with a default."""
    return os.environ.get(key, default)


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get an environment variable as a boolean.

    Recognizes: 1, true, yes, on (case-insensitive) as True.
    """
    val = os.environ.get(key, "").lower().strip()
    if not val:
        return default
    return val in ("1", "true", "yes", "on")


def get_env_int(key: str, default: int = 0) -> int:
    """Get an environment variable as an integer."""
    val = os.environ.get(key, "")
    try:
        return int(val)
    except (ValueError, TypeError):
        log.debug("Could not parse env var %s=%r as int, using default %d", key, val, default)
        return default
