"""Status-line runner — user-configurable status widgets.

Mirrors the Claude Code ``statusline`` contract: a shell command receives
a JSON payload on stdin and prints a short one-line string on stdout.
Forge surfaces that string in the TUI header and the web UI sidebar.

The design choice (command over plugin) is deliberate:

- Users write status lines in whatever language is handy (shell, Python,
  Node). No plugin API lock-in.
- The same script works under Claude Code, Cursor, Codex, and anvil —
  the JSON payload has the same shape.
- Failures fall back silently. A broken statusline should never crash
  the chat UX.

Config location (first match wins):

1. ``<working_dir>/.anvil/statusline.json``
2. ``$XDG_CONFIG_HOME/ollama-anvil/statusline.json``
3. ``~/.config/ollama-anvil/statusline.json``

Config schema::

    {
      "command": "/path/to/script.sh",
      "timeout": 5.0
    }

Payload shape (stdin, JSON)::

    {
      "model": "qwen2.5:14b",
      "agent": "coder",
      "messages": 12,
      "context_used": 3200,
      "context_total": 8192,
      "working_dir": "/abs/path",
      "session_id": "...",
      "cwd": "/abs/path"
    }

Response: plain text on stdout (anything after the first newline is
discarded). Exit-code 0 required; non-zero → fall back to the empty
status string.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anvil.utils.config_paths import config_roots
from anvil.utils.logging import get_logger

log = get_logger("ui.statusline")

__all__ = [
    "StatusLineConfig",
    "StatusLineResult",
    "load_statusline_config",
    "run_statusline",
    "DEFAULT_STATUSLINE_TIMEOUT",
    "MAX_STATUSLINE_BYTES",
]

# Timeout after which the status script is killed. Short by default
# because a slow statusline blocks UI refresh.
DEFAULT_STATUSLINE_TIMEOUT = 5.0

# Cap on stdout we'll keep (prevents a runaway script from filling
# stdout and costing us memory).
MAX_STATUSLINE_BYTES = 4 * 1024


@dataclass(slots=True)
class StatusLineConfig:
    """A loaded statusline configuration."""

    command: str
    timeout: float = DEFAULT_STATUSLINE_TIMEOUT
    source: str = ""  # config file path, for debugging


@dataclass(slots=True)
class StatusLineResult:
    """Outcome of running the statusline once."""

    text: str = ""
    ok: bool = True
    elapsed_s: float = 0.0
    error: str = ""


def load_statusline_config(working_dir: str | Path = ".") -> StatusLineConfig | None:
    """Find and parse a statusline config from the conventional paths.

    Returns ``None`` when no config is present. On parse / validation
    errors we also return ``None`` and log a warning rather than raising
    — statuslines are purely cosmetic.
    """
    candidates = config_roots("statusline.json", working_dir)

    for path in candidates:
        if not path.exists():
            continue
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            log.warning("Bad statusline config %s: %s", path, e)
            return None

        if not isinstance(data, dict):
            log.warning("Statusline config %s is not a JSON object", path)
            return None

        command = str(data.get("command", "")).strip()
        if not command:
            log.warning("Statusline config %s has empty command", path)
            return None

        try:
            timeout = float(data.get("timeout", DEFAULT_STATUSLINE_TIMEOUT))
        except (TypeError, ValueError):
            timeout = DEFAULT_STATUSLINE_TIMEOUT
        # Clamp to a sensible range — no-one wants a 60-second statusline.
        timeout = max(0.1, min(timeout, 15.0))

        return StatusLineConfig(command=command, timeout=timeout, source=str(path))

    return None


def run_statusline(
    cfg: StatusLineConfig,
    payload: dict[str, Any],
) -> StatusLineResult:
    """Execute the configured statusline command with ``payload`` on stdin.

    Returns a :class:`StatusLineResult` whose ``text`` is the first line
    of the command's stdout (trimmed). Failures set ``ok=False`` and
    populate ``error`` but never raise — keeps UI code simple.
    """
    try:
        argv = shlex.split(cfg.command)
    except ValueError as e:
        return StatusLineResult(ok=False, error=f"invalid command: {e}")

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
        return StatusLineResult(ok=False, error=f"command not found: {e}")
    except subprocess.TimeoutExpired:
        return StatusLineResult(
            ok=False,
            error=f"statusline timed out after {cfg.timeout:.1f}s",
            elapsed_s=cfg.timeout,
        )

    elapsed = time.monotonic() - start

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()[:MAX_STATUSLINE_BYTES]
        return StatusLineResult(
            ok=False,
            error=stderr or f"statusline exit {proc.returncode}",
            elapsed_s=elapsed,
        )

    stdout = (proc.stdout or "")[:MAX_STATUSLINE_BYTES]
    # Take the first non-empty line — statuslines are one-line by convention.
    first_line = ""
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped:
            first_line = stripped
            break

    return StatusLineResult(text=first_line, ok=True, elapsed_s=elapsed)
