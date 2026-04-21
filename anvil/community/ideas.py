"""Community ideas collection — anonymous, opt-out, crowdsourced improvements.

Users can submit ideas that improve ollama-anvil. Ideas are collected anonymously
and appended to a shared document on git. Users can opt out entirely.

Flow:
1. User has an idea (or the LLM suggests one during conversation)
2. Idea is anonymized (no usernames, IPs, or system info unless user opts in)
3. Idea is appended to community_ideas.jsonl (local) and optionally pushed to git
4. The self-improvement agent reads ideas and evaluates them
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from anvil.utils.fileio import atomic_write
from anvil.utils.logging import get_logger

log = get_logger("community.ideas")

__all__ = [
    "Idea",
    "IdeaCollector",
    "VALID_CATEGORIES",
    "VALID_SOURCES",
    "VALID_STATUSES",
    "MIN_TITLE_LENGTH",
    "MAX_TITLE_LENGTH",
    "MAX_DESCRIPTION_LENGTH",
    "FUZZY_DEDUP_THRESHOLD",
    "FUZZY_LENGTH_RATIO_MIN",
    "FUZZY_LENGTH_RATIO_MAX",
    "MAX_DESCRIPTION_DISPLAY",
    "MAX_IDEAS_DISPLAY",
]

# Valid categories for ideas
VALID_CATEGORIES = {"feature", "improvement", "bugfix", "performance", "ux", "other"}

# Valid sources for ideas
VALID_SOURCES = {"user", "agent", "self-improve"}

# Validation limits
MIN_TITLE_LENGTH = 5
MAX_TITLE_LENGTH = 200
MAX_DESCRIPTION_LENGTH = 2000

# Fuzzy duplicate detection threshold (SequenceMatcher-based)
FUZZY_DEDUP_THRESHOLD = 0.85
FUZZY_LENGTH_RATIO_MIN = 0.5  # skip comparison if length ratio below this
FUZZY_LENGTH_RATIO_MAX = 2.0  # skip comparison if length ratio above this

# Display limits
MAX_DESCRIPTION_DISPLAY = 120  # truncation for description in list view
MAX_IDEAS_DISPLAY = 100  # max ideas to show in format_ideas

# Valid statuses
VALID_STATUSES = {"new", "evaluated", "accepted", "rejected", "implemented"}


@dataclass(slots=True)
class Idea:
    """A community-submitted idea."""

    id: str  # short hash for dedup
    category: str  # "feature", "improvement", "bugfix", "performance", "ux", "other"
    title: str
    description: str
    submitted_at: float
    source: str = "user"  # "user", "agent", "self-improve"
    status: str = "new"  # "new", "evaluated", "accepted", "rejected", "implemented"
    votes: int = 1


class IdeaCollector:
    """Collects and manages community ideas.

    Ideas are stored locally in community_ideas.jsonl.
    When push_enabled=True, new ideas are staged for the next git push.

    Users can opt out by setting ANVIL_COMMUNITY_IDEAS=0 in .env.
    """

    def __init__(
        self,
        ideas_dir: str = "community",
        enabled: bool = True,
    ):
        self.ideas_dir = Path(ideas_dir)
        self.ideas_dir.mkdir(parents=True, exist_ok=True)
        self.ideas_file = self.ideas_dir / "community_ideas.jsonl"
        self.enabled = enabled
        self._ideas: dict[str, Idea] = self._load()

    def __repr__(self) -> str:
        return f"IdeaCollector(ideas={len(self._ideas)}, enabled={self.enabled})"

    def submit(self, title: str, description: str, category: str = "improvement", source: str = "user") -> str:
        """Submit a new idea with validation and fuzzy dedup.

        Validates:
        - Title length (5-200 chars)
        - Description length (max 2000 chars)
        - Category is a known enum value
        - Source is a known value
        - Not a near-duplicate of existing ideas (>85% similar)

        Returns confirmation message.
        """
        if not self.enabled:
            return "Community ideas collection is disabled. Enable with ANVIL_COMMUNITY_IDEAS=1"

        # Input validation
        errors = self._validate_idea(title, description, category, source)
        if errors:
            return f"Invalid idea: {'; '.join(errors)}"

        # Normalize category
        category = category.lower().strip()

        # Generate short ID from content hash
        content_hash = hashlib.sha256(f"{title}\x00{description}".encode()).hexdigest()[:8]

        # Check for exact duplicates
        if content_hash in self._ideas:
            self._ideas[content_hash].votes += 1
            self._save()
            return f"Similar idea already exists (id: {content_hash}). Added your vote (+1)."

        # Check for fuzzy duplicates
        fuzzy_match = self._find_fuzzy_duplicate(title, description)
        if fuzzy_match:
            fuzzy_match.votes += 1
            self._save()
            return f"Very similar idea already exists (id: {fuzzy_match.id}). Added your vote (+1)."

        idea = Idea(
            id=content_hash,
            category=category,
            title=title.strip(),
            description=description.strip(),
            submitted_at=time.time(),
            source=source,
        )
        self._ideas[content_hash] = idea
        self._save()
        log.info("New idea submitted: %s (%s)", title, content_hash)
        return f"Idea submitted! ID: {content_hash}. Thank you for contributing."

    @staticmethod
    def _validate_idea(title: str, description: str, category: str, source: str) -> list[str]:
        """Validate idea inputs. Returns list of errors (empty = valid)."""
        errors: list[str] = []

        if not title or len(title.strip()) < MIN_TITLE_LENGTH:
            errors.append(f"title must be at least {MIN_TITLE_LENGTH} characters")
        elif len(title) > MAX_TITLE_LENGTH:
            errors.append(f"title must be at most {MAX_TITLE_LENGTH} characters")

        if len(description) > MAX_DESCRIPTION_LENGTH:
            errors.append(f"description must be at most {MAX_DESCRIPTION_LENGTH} characters")

        if category.lower().strip() not in VALID_CATEGORIES:
            errors.append(f"category must be one of: {', '.join(sorted(VALID_CATEGORIES))}")

        if source.lower().strip() not in VALID_SOURCES:
            errors.append(f"source must be one of: {', '.join(sorted(VALID_SOURCES))}")

        return errors

    def _find_fuzzy_duplicate(self, title: str, description: str) -> Idea | None:
        """Find a near-duplicate idea using text similarity.

        Uses SequenceMatcher to detect ideas with >85% similarity
        in title+description, catching minor rewordings.
        """
        from difflib import SequenceMatcher

        new_text = f"{title} {description}".lower()
        for existing in self._ideas.values():
            existing_text = f"{existing.title} {existing.description}".lower()
            # Quick length ratio check
            len_ratio = len(new_text) / max(len(existing_text), 1)
            if len_ratio < FUZZY_LENGTH_RATIO_MIN or len_ratio > FUZZY_LENGTH_RATIO_MAX:
                continue
            ratio = SequenceMatcher(None, new_text, existing_text).ratio()
            if ratio >= FUZZY_DEDUP_THRESHOLD:
                return existing
        return None

    def list_ideas(self, status: str = "", category: str = "") -> list[Idea]:
        """List ideas, optionally filtered by status or category."""
        ideas = list(self._ideas.values())
        if status:
            ideas = [i for i in ideas if i.status == status]
        if category:
            ideas = [i for i in ideas if i.category == category]
        return sorted(ideas, key=lambda i: i.submitted_at, reverse=True)

    def get_new_ideas(self) -> list[Idea]:
        """Get all ideas that haven't been evaluated yet."""
        return [i for i in self._ideas.values() if i.status == "new"]

    def update_status(self, idea_id: str, status: str, reason: str = "") -> str:
        """Update an idea's status (used by self-improvement agent)."""
        if status not in VALID_STATUSES:
            return f"Invalid status '{status}'. Must be one of: {', '.join(sorted(VALID_STATUSES))}"
        idea = self._ideas.get(idea_id)
        if not idea:
            return f"Idea {idea_id} not found"
        idea.status = status
        self._save()
        return f"Idea {idea_id} status updated to: {status}"

    def format_ideas(self, ideas: list[Idea] | None = None) -> str:
        """Format ideas as a readable string."""
        ideas = ideas or self.list_ideas()
        if not ideas:
            return "No community ideas yet. Submit one with 'anvil idea submit'."

        lines = ["Community Ideas:\n"]
        for idea in ideas:
            status_icon = {
                "new": "[new]",
                "evaluated": "[eval]",
                "accepted": "[ok]",
                "rejected": "[no]",
                "implemented": "[done]",
            }.get(idea.status, "[?]")
            lines.append(f"  {status_icon} [{idea.category}] {idea.title} (votes: {idea.votes})")
            lines.append(f"          {idea.description[:MAX_DESCRIPTION_DISPLAY]}")
            lines.append("")
        return "\n".join(lines)

    def _load(self) -> dict[str, Idea]:
        if not self.ideas_file.exists():
            return {}
        ideas = {}
        for line in self.ideas_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                idea = Idea(**data)
                ideas[idea.id] = idea
            except (json.JSONDecodeError, TypeError) as e:
                log.debug("Skipping malformed idea entry: %s", e)
                continue
        return ideas

    def _save(self) -> None:
        lines = [json.dumps(asdict(idea)) for idea in self._ideas.values()]
        content = "\n".join(lines) + "\n" if lines else ""
        try:
            atomic_write(self.ideas_file, content)
        except OSError as e:
            log.error("Could not save ideas to %s: %s", self.ideas_file, e)
