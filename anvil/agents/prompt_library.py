"""Prompt template library — reusable prompts invoked by name.

Drops markdown prompt files into ``.anvil/prompts/*.md`` and makes them
callable as ``/prompt <name>`` via the orchestrator's slash-command
router. Each file may declare default args via frontmatter and use
``{placeholder}`` markers in the body — rendered via ``str.format_map``
with a defaulting dict so missing placeholders don't raise.

File format::

    ---
    name: extract-policy
    description: Extract a policy number from scanned text
    default_args:
      context: (no context provided)
    ---
    Extract the policy number from the following text. If the number
    is not visible, respond with NONE.

    <text>
    {text}
    </text>

    Context: {context}

Usage from code::

    from anvil.agents.prompt_library import PromptLibrary

    lib = PromptLibrary.discover("/path/to/project")
    prompt = lib.render("extract-policy", {"text": scanned})
    # prompt is a string ready to send to client.chat(...)

From chat:: the orchestrator accepts ``/prompt extract-policy text="..."``
and sends the rendered prompt to the active agent.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from string import Formatter
from typing import Any

from anvil.utils.config_paths import config_roots
from anvil.utils.frontmatter import parse_frontmatter
from anvil.utils.logging import get_logger

log = get_logger("agents.prompt_library")

__all__ = [
    "Prompt",
    "PromptLibrary",
    "load_prompt_file",
    "render_prompt",
    "default_prompt_roots",
    "MAX_PROMPT_BYTES",
]

# Cap on prompt file size — if a prompt is >64 KB it's not a prompt.
MAX_PROMPT_BYTES = 64 * 1024


@dataclass(slots=True)
class Prompt:
    """A single prompt template loaded from a markdown file."""

    name: str
    description: str = ""
    body: str = ""
    default_args: dict[str, Any] = field(default_factory=dict)
    path: str = ""

    def placeholders(self) -> list[str]:
        """Return every unique ``{placeholder}`` name used in the body."""
        names: list[str] = []
        seen: set[str] = set()
        for _literal, field_name, _spec, _conv in Formatter().parse(self.body):
            if field_name and field_name not in seen:
                names.append(field_name)
                seen.add(field_name)
        return names

    def render(self, args: dict[str, Any] | None = None) -> str:
        """Render the body with ``args`` merged over ``default_args``.

        Missing placeholders render as ``""`` rather than raising — a
        prompt that stays parse-able even with sparse input is more
        useful in production.
        """
        merged: dict[str, Any] = dict(self.default_args)
        if args:
            merged.update(args)
        return render_prompt(self.body, merged)


def render_prompt(template: str, args: dict[str, Any]) -> str:
    """Render ``template`` against ``args`` with missing-key → "" semantics."""
    class _Defaulting(dict):
        def __missing__(self, key: str) -> str:
            return ""
    return template.format_map(_Defaulting(args))


def load_prompt_file(path: str | Path) -> Prompt | None:
    """Load one prompt file. Returns ``None`` on unparseable input."""
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        log.debug("Cannot read prompt %s: %s", path, e)
        return None
    if len(raw) > MAX_PROMPT_BYTES:
        log.warning("Prompt %s exceeds %d bytes; truncating", path, MAX_PROMPT_BYTES)
        raw = raw[:MAX_PROMPT_BYTES]

    meta, body = parse_frontmatter(raw)

    name = str(meta.get("name") or p.stem).strip()
    description = str(meta.get("description", "")).strip()
    raw_defaults = meta.get("default_args", {})
    default_args: dict[str, Any] = {}
    if isinstance(raw_defaults, dict):
        default_args = {str(k): v for k, v in raw_defaults.items()}

    return Prompt(
        name=name,
        description=description,
        body=body.strip(),
        default_args=default_args,
        path=str(p),
    )


def default_prompt_roots(working_dir: str | Path = ".") -> list[Path]:
    """Canonical prompt-library search path."""
    return config_roots("prompts", working_dir)


@dataclass(slots=True)
class PromptLibrary:
    """Collection of prompts loaded from one or more directories."""

    prompts: dict[str, Prompt] = field(default_factory=dict)
    roots: list[Path] = field(default_factory=list)

    @classmethod
    def discover(cls, working_dir: str | Path = ".") -> PromptLibrary:
        return cls.from_paths(default_prompt_roots(working_dir))

    @classmethod
    def from_paths(cls, paths: list[Path] | list[str]) -> PromptLibrary:
        prompts: dict[str, Prompt] = {}
        real_roots: list[Path] = []
        for raw_root in paths:
            root = Path(raw_root)
            if not root.is_dir():
                continue
            real_roots.append(root)
            for md in sorted(root.glob("*.md")):
                prompt = load_prompt_file(md)
                if prompt is None or prompt.name in prompts:
                    continue
                prompts[prompt.name] = prompt
        return cls(prompts=prompts, roots=real_roots)

    def __len__(self) -> int:
        return len(self.prompts)

    def names(self) -> list[str]:
        return sorted(self.prompts)

    def get(self, name: str) -> Prompt | None:
        return self.prompts.get(name)

    def render(self, name: str, args: dict[str, Any] | None = None) -> str:
        """Render ``name`` with ``args``. Raises KeyError on missing prompt."""
        prompt = self.prompts.get(name)
        if prompt is None:
            raise KeyError(f"no prompt named {name!r}")
        return prompt.render(args)

    def handle_command(self, message: str) -> tuple[bool, str]:
        """Parse a ``/prompt ...`` slash command.

        Returns ``(handled, output)``. When ``handled`` is False the
        message was not a /prompt command — caller should continue its
        normal dispatch. When True, ``output`` is either the rendered
        prompt (caller should send to the model) or an error message
        ready to display to the user.
        """
        stripped = message.strip()
        if not stripped.startswith("/prompt"):
            return False, ""
        remainder = stripped[len("/prompt"):].strip()
        if not remainder:
            return True, self.format_list()

        # Tokenise the rest of the line; first token is the prompt name,
        # the rest are key=value pairs (shell-quoting allowed).
        try:
            parts = shlex.split(remainder)
        except ValueError as e:
            return True, f"Error parsing command: {e}"
        if not parts:
            return True, self.format_list()
        name = parts[0]
        args: dict[str, Any] = {}
        for token in parts[1:]:
            if "=" not in token:
                return True, f"Error: argument {token!r} is not key=value"
            k, _, v = token.partition("=")
            args[k.strip()] = v
        prompt = self.prompts.get(name)
        if prompt is None:
            return True, (
                f"Error: no prompt named {name!r}. "
                f"Available: {', '.join(self.names()) or '(none)'}"
            )
        return True, prompt.render(args)

    def format_list(self) -> str:
        """Human-readable list for ``/prompt`` with no args."""
        if not self.prompts:
            return "No prompts installed. Drop .md files into .anvil/prompts/ to add some."
        lines = ["Available prompts:"]
        for p in sorted(self.prompts.values(), key=lambda p: p.name):
            desc = f" — {p.description}" if p.description else ""
            placeholders = p.placeholders()
            ph = f"  placeholders: {', '.join(placeholders)}" if placeholders else ""
            lines.append(f"  /prompt {p.name}{desc}")
            if ph:
                lines.append(ph)
        return "\n".join(lines)
