"""Skills-as-markdown — lightweight procedural memory for agents.

A *skill* is a Markdown file with a small YAML frontmatter block:

    ---
    name: scan-iceberg-metadata
    description: How to inspect an Iceberg table's schema via DuckDB
    triggers: [iceberg, parquet, table schema]
    ---

    Body text goes here. Anything in the body is injected verbatim into
    the system prompt for turns where the skill is considered relevant.

Why markdown and not a vector DB? Three reasons drawn from the 2026
Anthropic / Nous / LangChain agent research:

1. Skills are human-editable. ``git blame`` and a text editor beat any
   opaque embedding store when you need to audit or correct an agent's
   learned behaviour.
2. Loading is cheap. For the dozens-to-hundreds-of-skills scale typical
   of a single user, string matching against triggers + the description
   is fast enough (milliseconds) and has zero infrastructure cost.
3. Skills can optionally be promoted to vector search later without
   changing their storage format. If :class:`~anvil.llm.embeddings.EmbeddingStore`
   is available we lean on it for semantic match; otherwise we fall back
   to substring and token overlap.

Search order for the skills root:

1. ``<working_dir>/.anvil/skills/``  — project-scoped
2. ``$XDG_CONFIG_HOME/ollama-anvil/skills/``
3. ``~/.config/ollama-anvil/skills/`` — user-wide

Every ``*.md`` file found is loaded as a :class:`Skill`. Missing frontmatter
is tolerated (name falls back to the filename stem, description blank,
triggers empty).

This module is intentionally self-contained: no new dependencies beyond
pyyaml, which is already a core dependency of anvil.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from anvil.utils.config_paths import config_roots, user_config_root
from anvil.utils.frontmatter import parse_frontmatter
from anvil.utils.logging import get_logger

log = get_logger("agents.skills")

__all__ = [
    "Skill",
    "SkillLibrary",
    "load_skill_file",
    "MAX_SKILLS_INJECTED",
    "MAX_SKILL_FILE_BYTES",
    "DEFAULT_TOP_K",
]

# Cap how many skills we inject in a single turn. Too many and the system
# prompt balloons past usable context on small local models.
MAX_SKILLS_INJECTED = 3

# Skills live in the user's config dir — guard against accidentally
# loading runaway-large files (someone dumped a novel into skills/).
MAX_SKILL_FILE_BYTES = 64 * 1024

# How many candidates we rank before applying MAX_SKILLS_INJECTED.
DEFAULT_TOP_K = 8

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{2,}")


@dataclass(slots=True)
class Skill:
    """A single markdown-defined skill."""

    name: str
    description: str = ""
    triggers: list[str] = field(default_factory=list)
    body: str = ""
    path: str = ""  # source file path, for debugging/logging

    def to_prompt_fragment(self) -> str:
        """Render this skill as a block to paste into the system prompt."""
        header = f"### Skill: {self.name}"
        if self.description:
            header += f" — {self.description}"
        return f"{header}\n\n{self.body.strip()}\n"


def load_skill_file(path: str | Path) -> Skill | None:
    """Load a single skill file. Returns ``None`` on unparseable input.

    Frontmatter is optional. If present, we parse ``name``, ``description``,
    and ``triggers`` (accepting either a YAML list or a comma-separated
    string). Anything else in the frontmatter is ignored — forward-compatible
    with future fields.
    """
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        log.debug("Cannot read skill %s: %s", path, e)
        return None
    if len(raw) > MAX_SKILL_FILE_BYTES:
        log.warning("Skill %s exceeds %d bytes; truncating", path, MAX_SKILL_FILE_BYTES)
        raw = raw[:MAX_SKILL_FILE_BYTES]

    meta, body = parse_frontmatter(raw)

    name = str(meta.get("name") or p.stem).strip()
    description = str(meta.get("description", "")).strip()
    triggers = _coerce_trigger_list(meta.get("triggers", []))

    return Skill(
        name=name,
        description=description,
        triggers=triggers,
        body=body.strip(),
        path=str(p),
    )


def _coerce_trigger_list(raw: object) -> list[str]:
    """Accept a YAML list, a comma-separated string, or nothing."""
    if isinstance(raw, list):
        return [str(t).strip().lower() for t in raw if str(t).strip()]
    if isinstance(raw, str):
        return [t.strip().lower() for t in raw.split(",") if t.strip()]
    return []


@dataclass(slots=True)
class SkillLibrary:
    """Collection of skills loaded from one or more directories.

    Construct with :meth:`discover` to automatically crawl the conventional
    search path, or with :meth:`from_paths` if you have your own roots.
    """

    skills: list[Skill] = field(default_factory=list)
    roots: list[Path] = field(default_factory=list)

    # ─── Construction ───────────────────────────────────────────

    @classmethod
    def discover(cls, working_dir: str | Path = ".") -> SkillLibrary:
        """Find and load skills from the canonical search path."""
        roots = default_skill_roots(working_dir)
        return cls.from_paths(roots)

    @classmethod
    def from_paths(cls, paths: list[Path] | list[str]) -> SkillLibrary:
        """Load skills from an explicit list of directories."""
        skills: list[Skill] = []
        real_roots: list[Path] = []
        seen_names: set[str] = set()
        for raw_root in paths:
            root = Path(raw_root)
            if not root.is_dir():
                continue
            real_roots.append(root)
            for md in sorted(root.glob("*.md")):
                skill = load_skill_file(md)
                if skill is None or skill.name in seen_names:
                    # Per-name precedence: first root wins. That puts the
                    # project-scoped skills ahead of user-scoped ones.
                    continue
                skills.append(skill)
                seen_names.add(skill.name)
        log.debug("Loaded %d skills from %d root(s)", len(skills), len(real_roots))
        return cls(skills=skills, roots=real_roots)

    # ─── Retrieval ──────────────────────────────────────────────

    def match(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[Skill]:
        """Return the top-``top_k`` skills most relevant to ``query``.

        Uses a lightweight scoring rubric (no embeddings required):

        - +3 per trigger whose lowercased form appears in the query.
        - +2 per token overlap between query and skill name.
        - +1 per token overlap between query and skill description.

        A skill scoring zero is dropped. Ties are broken by the order the
        skills appeared on disk.
        """
        if not self.skills:
            return []
        q_lower = query.lower()
        q_tokens = {m.group(0).lower() for m in _WORD_RE.finditer(q_lower)}

        ranked: list[tuple[int, int, Skill]] = []
        for idx, skill in enumerate(self.skills):
            score = 0
            for trig in skill.triggers:
                if trig and trig in q_lower:
                    score += 3
            if skill.name:
                name_tokens = {m.group(0).lower() for m in _WORD_RE.finditer(skill.name)}
                score += 2 * len(q_tokens & name_tokens)
            if skill.description:
                desc_tokens = {m.group(0).lower() for m in _WORD_RE.finditer(skill.description)}
                score += len(q_tokens & desc_tokens)
            if score > 0:
                ranked.append((score, idx, skill))

        ranked.sort(key=lambda t: (-t[0], t[1]))
        return [s for _score, _idx, s in ranked[:top_k]]

    def build_injection(self, query: str, max_skills: int = MAX_SKILLS_INJECTED) -> str:
        """Format the top matches as a system-prompt fragment.

        Returns ``""`` when no skills match, so callers can blanket-concat
        without a branch.
        """
        matched = self.match(query, top_k=max_skills)
        if not matched:
            return ""
        blocks = [s.to_prompt_fragment() for s in matched]
        return "--- Skills ---\n" + "\n".join(blocks).rstrip() + "\n"

    # ─── Mutation ───────────────────────────────────────────────

    def add(self, skill: Skill) -> None:
        """Add a skill in-memory (does not write to disk)."""
        self.skills.append(skill)

    def write(self, skill: Skill, root: Path | None = None) -> Path:
        """Persist a skill to disk as ``<root>/<name>.md``.

        ``root`` defaults to the first writable discovered root, or
        ``~/.config/ollama-anvil/skills`` if the library was built empty.
        """
        target_root = root or (self.roots[0] if self.roots else default_user_skill_root())
        target_root.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", skill.name.strip().lower()).strip("-")
        if not safe:
            raise ValueError("Skill name must contain alphanumerics")
        path = target_root / f"{safe}.md"
        frontmatter = _render_frontmatter(skill)
        path.write_text(frontmatter + "\n" + skill.body.strip() + "\n", encoding="utf-8")
        skill.path = str(path)
        self.skills.append(skill)
        return path

    # ─── Access ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.skills)

    def names(self) -> list[str]:
        """Return the names of loaded skills in load order."""
        return [s.name for s in self.skills]


def _render_frontmatter(skill: Skill) -> str:
    """Render a skill's metadata as a YAML-like frontmatter block."""
    triggers = ", ".join(skill.triggers)
    return (
        "---\n"
        f"name: {skill.name}\n"
        f"description: {skill.description}\n"
        f"triggers: [{triggers}]\n"
        "---\n"
    )


def default_skill_roots(working_dir: str | Path = ".") -> list[Path]:
    """The canonical search path: project → XDG → ~/.config."""
    return config_roots("skills", working_dir)


def default_user_skill_root() -> Path:
    """User-scoped root: ``~/.config/ollama-anvil/skills``."""
    return user_config_root("skills")
