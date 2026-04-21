"""YAML frontmatter parsing — shared across skills, rules, prompts, etc.

Extracted from the repeated ``^---\\n...yaml...\\n---\\n...body...`` pattern
in :mod:`anvil.agents.skills`, :mod:`anvil.agents.rules`,
:mod:`anvil.agents.prompt_library`, :mod:`anvil.agents.backlog`, and
:mod:`anvil.trading.strategy`. Each of those used an identical regex plus a
pyyaml-with-fallback parser.

Usage::

    from anvil.utils.frontmatter import parse_frontmatter

    meta, body = parse_frontmatter(file_text)
    name = meta.get("name", "")

If pyyaml is not installed, the fallback parses ``key: value`` and
``key: [a, b, c]`` — enough for the simple frontmatter formats in-repo, but
not a general YAML replacement.
"""

from __future__ import annotations

import re
from typing import Any

from anvil.utils.logging import get_logger

log = get_logger("frontmatter")

__all__ = [
    "FRONTMATTER_RE",
    "parse_frontmatter",
    "parse_yaml_text",
]

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split ``text`` into (metadata_dict, body_str).

    Returns ``({}, text)`` when no frontmatter block is present — this
    matches how every caller in the repo handled the no-frontmatter case.
    """
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    meta = parse_yaml_text(match.group(1))
    return meta, match.group(2)


def parse_yaml_text(text: str) -> dict[str, Any]:
    """Parse a YAML blob. Returns ``{}`` on any failure.

    Falls back to the simple key-value parser when pyyaml is missing —
    many environments don't have it, and our frontmatter is intentionally
    simple.
    """
    try:
        import yaml
    except ImportError:
        return _parse_simple(text)
    try:
        data = yaml.safe_load(text)
    except Exception as e:  # noqa: BLE001 — yaml errors span many classes
        log.debug("frontmatter yaml parse error: %s", e)
        return {}
    return data if isinstance(data, dict) else {}


def _parse_simple(text: str) -> dict[str, Any]:
    """Minimal ``key: value`` / ``key: [a, b]`` parser for pyyaml-less envs."""
    out: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            out[key] = [v.strip().strip("\"'") for v in val[1:-1].split(",") if v.strip()]
        else:
            out[key] = val.strip("\"'")
    return out
