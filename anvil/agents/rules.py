"""Project rules: per-project custom instructions for agents.

Two layers in the same module:

1. **Monolithic rules** (legacy, still supported): a single ``.forge-rules``
   or ``FORGE.md`` / ``CLAUDE.md`` file at the project root. Loaded by
   :func:`load_project_rules` and always prepended to the system prompt.

2. **Glob-scoped rules** (Continue / Cursor pattern): individual markdown
   files under ``.anvil/rules/``, each with optional frontmatter::

        ---
        name: python-style
        description: Style guide for Python files
        globs: ["**/*.py"]
        priority: 10
        ---
        body...

   Rules without globs are treated as "always on". Rules with globs are
   included in a given turn only if one of the glob patterns matches the
   current working directory, a file path the user mentioned, or the
   paths returned by tool calls in the conversation so far.

The rules file is plain text (Markdown supported) and is automatically prepended
to the agent's system prompt when working in that directory.

Supports hierarchical rules:
1. Global rules: ~/.config/ollama-anvil/rules.md
2. Project rules: .forge-rules (in project root)
3. Directory rules: .forge-rules (in subdirectory — overrides project-level)

Rules are merged top-down: global → project → directory.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path

from anvil.utils.frontmatter import parse_frontmatter
from anvil.utils.logging import get_logger

log = get_logger("agents.rules")

__all__ = [
    "load_project_rules",
    "create_rules_template",
    "load_glob_rules",
    "match_glob_rules",
    "build_glob_rules_fragment",
    "GlobRule",
    "RULES_FILENAMES",
    "GLOBAL_RULES_PATH",
    "MAX_RULES_FILE_SIZE",
    "MAX_GLOB_RULES_INJECTED",
    "MAX_GLOB_RULE_FILE_BYTES",
]

# Upper bound on rules injected in a single turn. Too many and the system
# prompt balloons past usable context on small local models.
MAX_GLOB_RULES_INJECTED = 6

# Per-rule-file cap, same rationale as skills.
MAX_GLOB_RULE_FILE_BYTES = 64 * 1024

# File names to look for (in priority order)
RULES_FILENAMES = [".forge-rules", "FORGE.md", "CLAUDE.md"]

# Global rules location
GLOBAL_RULES_PATH = Path.home() / ".config" / "ollama-anvil" / "rules.md"

# Size limit for rules files (prevent loading huge files)
MAX_RULES_FILE_SIZE = 100_000  # 100 KB


def load_project_rules(working_dir: str = ".") -> str:
    """Load and merge rules from global, project, and directory levels.

    Returns the combined rules text, or empty string if no rules found.
    """
    parts = []

    # 1. Global rules
    global_rules = _read_rules_file(GLOBAL_RULES_PATH)
    if global_rules:
        parts.append(f"# Global Rules\n{global_rules}")

    # 2. Walk up from working_dir to find project-level rules
    project_rules, project_dir = _find_project_rules(working_dir)
    if project_rules:
        parts.append(f"# Project Rules\n{project_rules}")

    # 3. Directory-level rules (if different from project root)
    work_path = Path(working_dir).resolve()
    if project_dir and work_path != project_dir:
        dir_rules = _find_rules_in_dir(work_path)
        if dir_rules:
            parts.append(f"# Directory Rules\n{dir_rules}")

    combined = "\n\n".join(parts)
    if combined:
        log.info("Loaded project rules (%d chars)", len(combined))
    return combined


def _find_project_rules(working_dir: str) -> tuple[str, Path | None]:
    """Walk up from working_dir to find the nearest rules file.

    Stops at filesystem root or home directory.
    """
    current = Path(working_dir).resolve()
    home = Path.home()

    while current != current.parent:
        rules = _find_rules_in_dir(current)
        if rules:
            return rules, current

        # Don't search above home directory
        if current == home:
            break
        current = current.parent

    return "", None


def _find_rules_in_dir(directory: Path) -> str:
    """Look for a rules file in a specific directory."""
    for filename in RULES_FILENAMES:
        rules_path = directory / filename
        content = _read_rules_file(rules_path)
        if content:
            return content
    return ""


def _read_rules_file(path: Path) -> str:
    """Read a rules file if it exists."""
    if path.exists() and path.is_file():
        try:
            if path.stat().st_size > MAX_RULES_FILE_SIZE:
                log.warning("Rules file %s too large (%d bytes, max %d), skipping",
                            path, path.stat().st_size, MAX_RULES_FILE_SIZE)
                return ""
            content = path.read_text(encoding="utf-8").strip()
            if content:
                log.debug("Read rules from %s", path)
                return content
        except (OSError, UnicodeDecodeError) as e:
            log.warning("Could not read rules file %s: %s", path, e)
    return ""


def create_rules_template(directory: str = ".") -> str:
    """Create a template .forge-rules file in the given directory."""
    path = Path(directory) / ".forge-rules"
    if path.exists():
        return f"Rules file already exists: {path}"

    template = """# .forge-rules — Custom instructions for AI agents

# This file is automatically read by ollama-anvil when working in this directory.
# Add project-specific instructions, coding conventions, and constraints here.

## Project Overview
# Describe your project briefly so the agent understands the context.

## Coding Style
# - Language and version requirements
# - Formatting preferences (tabs vs spaces, line length, etc.)
# - Naming conventions (camelCase, snake_case, etc.)

## Architecture
# - Key patterns used (MVC, microservices, etc.)
# - Important directories and their purposes
# - Dependencies and frameworks

## Constraints
# - Things the agent should NEVER do
# - Security requirements
# - Performance considerations

## Testing
# - How to run tests
# - Test framework preferences
# - Coverage requirements
"""
    path.write_text(template, encoding="utf-8")
    return f"Created rules template: {path}"


# ════════════════════════════════════════════════════════════════════
# Glob-scoped rules (.anvil/rules/*.md)
# ════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class GlobRule:
    """A single glob-scoped rule loaded from ``.anvil/rules/*.md``.

    Rules without globs are treated as always-on. Priority (higher first)
    breaks ties when more than :data:`MAX_GLOB_RULES_INJECTED` rules match.
    """

    name: str
    description: str = ""
    globs: list[str] = field(default_factory=list)
    priority: int = 0
    body: str = ""
    path: str = ""

    @property
    def always_on(self) -> bool:
        """True if this rule is always injected (no glob restrictions)."""
        return not self.globs

    def matches(self, candidate_paths: list[str]) -> bool:
        """Return True when any candidate path matches any of our globs.

        Always-on rules (no globs) are considered to match every turn. A
        glob matches if :func:`fnmatch.fnmatch` accepts either the full
        path or the path's basename.
        """
        if self.always_on:
            return True
        for pattern in self.globs:
            for cand in candidate_paths:
                if fnmatch.fnmatch(cand, pattern) or fnmatch.fnmatch(Path(cand).name, pattern):
                    return True
        return False

    def to_prompt_fragment(self) -> str:
        header = f"### Rule: {self.name}"
        if self.description:
            header += f" — {self.description}"
        return f"{header}\n\n{self.body.strip()}\n"


def load_glob_rules(working_dir: str | Path = ".") -> list[GlobRule]:
    """Load every ``*.md`` file under ``<working_dir>/.anvil/rules/``.

    Silently tolerates parse errors on individual files. Rules are
    returned in filename order for stable injection ordering.
    """
    rules_dir = Path(working_dir) / ".forge" / "rules"
    if not rules_dir.is_dir():
        return []

    rules: list[GlobRule] = []
    for md_path in sorted(rules_dir.glob("*.md")):
        rule = _load_glob_rule_file(md_path)
        if rule is not None:
            rules.append(rule)
    return rules


def _load_glob_rule_file(path: Path) -> GlobRule | None:
    """Parse one glob-rule markdown file."""
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        log.debug("Cannot read rule %s: %s", path, e)
        return None
    if len(raw) > MAX_GLOB_RULE_FILE_BYTES:
        log.warning("Rule %s exceeds %d bytes; truncating", path, MAX_GLOB_RULE_FILE_BYTES)
        raw = raw[:MAX_GLOB_RULE_FILE_BYTES]

    meta, body = parse_frontmatter(raw)

    name = str(meta.get("name") or path.stem).strip()
    description = str(meta.get("description", "")).strip()
    globs = _coerce_string_list(meta.get("globs", []))
    priority = _coerce_int(meta.get("priority", 0))

    return GlobRule(
        name=name,
        description=description,
        globs=globs,
        priority=priority,
        body=body.strip(),
        path=str(path),
    )


def _coerce_string_list(raw: object) -> list[str]:
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        return [s.strip() for s in raw.split(",") if s.strip()]
    return []


def _coerce_int(raw: object) -> int:
    try:
        return int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


_PATH_LIKE_RE = re.compile(r"([\w./\\-]+\.[a-zA-Z0-9]{1,6})")


def extract_candidate_paths(user_message: str, working_dir: str | Path = ".") -> list[str]:
    """Gather candidate paths to match globs against.

    Sources:

    - The working directory itself (so rules with globs like ``src/web/**``
      match when the agent is simply operating in that tree).
    - File-path-looking tokens in the user message (anything that looks
      like ``foo.py`` or ``src/bar.tsx``).

    Duplicates are removed while preserving order.
    """
    candidates: list[str] = [str(working_dir)]
    seen: set[str] = set(candidates)
    for match in _PATH_LIKE_RE.finditer(user_message or ""):
        token = match.group(1)
        if token not in seen:
            candidates.append(token)
            seen.add(token)
    return candidates


def match_glob_rules(
    rules: list[GlobRule],
    user_message: str,
    working_dir: str | Path = ".",
    max_rules: int = MAX_GLOB_RULES_INJECTED,
) -> list[GlobRule]:
    """Select the rules that apply to ``user_message`` in ``working_dir``.

    Always-on rules are included first, then glob-scoped rules whose
    patterns match a candidate path. Results are capped at ``max_rules``
    with priority-high-first, filename-stable tie break.
    """
    candidates = extract_candidate_paths(user_message, working_dir)
    always: list[GlobRule] = []
    scoped: list[GlobRule] = []
    for rule in rules:
        if rule.always_on:
            always.append(rule)
        elif rule.matches(candidates):
            scoped.append(rule)

    # Priority ordering: highest first. Keep load-order as a stable
    # secondary sort (Python's sort is stable, so the pre-sort order is
    # preserved within equal priorities).
    always.sort(key=lambda r: -r.priority)
    scoped.sort(key=lambda r: -r.priority)

    combined = always + scoped
    return combined[:max_rules]


def build_glob_rules_fragment(
    rules: list[GlobRule],
    user_message: str,
    working_dir: str | Path = ".",
    max_rules: int = MAX_GLOB_RULES_INJECTED,
) -> str:
    """Render the matching glob rules as a system-prompt fragment.

    Returns ``""`` when nothing matches so callers can concatenate
    unconditionally.
    """
    matched = match_glob_rules(rules, user_message, working_dir, max_rules)
    if not matched:
        return ""
    parts = [r.to_prompt_fragment() for r in matched]
    return "--- Rules ---\n" + "\n".join(parts).rstrip() + "\n"
