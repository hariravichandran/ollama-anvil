"""Session persistence: save, load, and resume chat sessions.

Allows users to save their work and resume later exactly where they left off.
Sessions include the full conversation history, agent state, and metadata.

Storage: ~/.config/ollama-anvil/sessions/

Usage:
    sessions = SessionManager()

    # Save current session
    session_id = sessions.save(messages, agent_name="coder", metadata={...})

    # List saved sessions
    for s in sessions.list_sessions():
        print(f"{s.session_id}: {s.title} ({s.message_count} messages)")

    # Load and resume
    session = sessions.load(session_id)
    messages = session.messages
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from anvil.utils.fileio import atomic_write
from anvil.utils.logging import get_logger

log = get_logger("agents.sessions")

__all__ = [
    "Session",
    "SessionSummary",
    "SessionManager",
    "SESSIONS_DIR",
    "MAX_SESSION_SIZE_MB",
    "MAX_MESSAGES_PER_SESSION",
    "MAX_SESSIONS_ON_DISK",
    "DEFAULT_LIST_LIMIT",
    "DEFAULT_SEARCH_LIMIT",
    "VALID_EXPORT_FORMATS",
    "SECONDS_PER_MINUTE",
    "SECONDS_PER_HOUR",
    "SECONDS_PER_DAY",
    "MAX_TITLE_LENGTH",
    "MAX_SYSTEM_CONTENT_DISPLAY",
    "MAX_SEARCH_RESULT_CONTENT",
    "MIN_LIST_LIMIT",
    "MAX_LIST_LIMIT",
    "MAX_SEARCH_LIMIT",
    "TRUNCATION_DIVISOR",
]

SESSIONS_DIR = Path.home() / ".config" / "ollama-anvil" / "sessions"

# Session limits
MAX_SESSION_SIZE_MB = 10  # Maximum session file size
MAX_MESSAGES_PER_SESSION = 10_000  # Max messages in a single session
MAX_SESSIONS_ON_DISK = 1000  # Cleanup oldest sessions beyond this

# Time thresholds for age display
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400

# Display limits
MAX_TITLE_LENGTH = 50  # title auto-generation truncation
MAX_SYSTEM_CONTENT_DISPLAY = 100  # system message preview
MAX_SEARCH_RESULT_CONTENT = 200  # search result truncation
DEFAULT_LIST_LIMIT = 20  # default sessions to show
MIN_LIST_LIMIT = 1
MAX_LIST_LIMIT = 500
DEFAULT_SEARCH_LIMIT = 10
MAX_SEARCH_LIMIT = 100

# Validation
VALID_EXPORT_FORMATS = {"markdown", "json", "html"}
TRUNCATION_DIVISOR = 2  # keep last half of messages when oversized


@dataclass(slots=True)
class Session:
    """A saved chat session."""

    session_id: str
    title: str
    agent_name: str
    model: str
    messages: list[dict[str, str]]
    created_at: float
    updated_at: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def message_count(self) -> int:
        """Total number of messages in this session."""
        return len(self.messages)

    @property
    def duration_s(self) -> float:
        """Elapsed time in seconds between session creation and last update."""
        return self.updated_at - self.created_at

    def summary(self) -> str:
        """One-line summary for listing."""
        age = time.time() - self.updated_at
        if age < SECONDS_PER_HOUR:
            age_str = f"{int(age / SECONDS_PER_MINUTE)}m ago"
        elif age < SECONDS_PER_DAY:
            age_str = f"{int(age / SECONDS_PER_HOUR)}h ago"
        else:
            age_str = f"{int(age / SECONDS_PER_DAY)}d ago"
        return f"{self.session_id[:8]} | {self.title} | {self.agent_name} | {self.message_count} msgs | {age_str}"


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """Lightweight session info for listing (without full messages)."""

    session_id: str
    title: str
    agent_name: str
    model: str
    message_count: int
    created_at: float
    updated_at: float


class SessionManager:
    """Manages persistent chat sessions."""

    def __init__(self, sessions_dir: Path | None = None):
        self.sessions_dir = sessions_dir or SESSIONS_DIR
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_orphaned_files()

    def __repr__(self) -> str:
        return f"SessionManager(dir={self.sessions_dir!r})"

    def close(self) -> None:
        """No-op: SessionManager has no persistent handles to close."""
        pass

    def __enter__(self) -> SessionManager:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def save(
        self,
        messages: list[dict[str, str]],
        agent_name: str = "assistant",
        model: str = "",
        title: str = "",
        session_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save a chat session.

        Args:
            messages: Conversation message history.
            agent_name: Name of the active agent.
            model: Model used in this session.
            title: Session title (auto-generated if empty).
            session_id: Existing session ID to update, or empty for new.
            metadata: Additional metadata to store.

        Returns:
            Session ID.
        """
        now = time.time()

        if not session_id:
            session_id = f"session-{uuid.uuid4().hex[:8]}"

        if not title:
            title = self._generate_title(messages)

        session = Session(
            session_id=session_id,
            title=title,
            agent_name=agent_name,
            model=model,
            messages=messages,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )

        # Check if updating existing session
        existing_path = self.sessions_dir / f"{session_id}.json"
        if existing_path.exists():
            try:
                existing = json.loads(existing_path.read_text(encoding="utf-8"))
                session.created_at = existing.get("created_at", now)
            except (json.JSONDecodeError, OSError) as e:
                log.debug("Could not read existing session %s: %s", session_id, e)

        # Enforce message limit
        if len(messages) > MAX_MESSAGES_PER_SESSION:
            log.warning("Session %s has %d messages (max %d), truncating",
                        session_id, len(messages), MAX_MESSAGES_PER_SESSION)
            session.messages = messages[-MAX_MESSAGES_PER_SESSION:]

        # Set updated_at after all modifications
        session.updated_at = now

        # Write to disk
        session_path = self.sessions_dir / f"{session_id}.json"
        content = json.dumps(asdict(session), indent=2)

        # Check size before writing
        size_mb = len(content.encode("utf-8")) / (1024 * 1024)
        if size_mb > MAX_SESSION_SIZE_MB:
            log.warning("Session %s too large (%.1f MB), truncating messages", session_id, size_mb)
            # Keep only the last half of messages
            half = len(session.messages) // TRUNCATION_DIVISOR
            session.messages = session.messages[half:]
            content = json.dumps(asdict(session), indent=2)

        atomic_write(session_path, content)
        log.info("Saved session %s (%d messages)", session_id, len(session.messages))
        return session_id

    def load(self, session_id: str) -> Session | None:
        """Load a saved session by ID.

        Supports partial ID matching (first 8 chars).
        """
        # Try exact match first
        session_path = self.sessions_dir / f"{session_id}.json"
        if not session_path.exists():
            # Try partial match
            matches = list(self.sessions_dir.glob(f"{session_id}*.json"))
            if len(matches) == 1:
                session_path = matches[0]
            elif len(matches) > 1:
                log.warning("Ambiguous session ID: %s (matches %d sessions)", session_id, len(matches))
                return None
            else:
                log.debug("Session not found: %s", session_id)
                return None

        try:
            raw = session_path.read_text(encoding="utf-8", errors="replace")
            data = json.loads(raw)
            # Validate required fields
            if "session_id" not in data or "messages" not in data:
                log.error("Session %s missing required fields", session_id)
                return None
            # Filter to known fields for forward compatibility
            known = {f.name for f in fields(Session)}
            return Session(**{k: v for k, v in data.items() if k in known})
        except json.JSONDecodeError as e:
            log.error("Corrupted session file %s: %s", session_id, e)
            # Try to recover by backing up the corrupted file
            backup = session_path.with_suffix(".json.corrupted")
            try:
                os.replace(str(session_path), str(backup))
                log.info("Moved corrupted session to %s", backup)
            except OSError as oe:
                log.warning("Could not backup corrupted session %s: %s", session_id, oe)
            return None
        except (OSError, TypeError) as e:
            log.error("Failed to load session %s: %s", session_id, e)
            return None

    def list_sessions(self, limit: int = DEFAULT_LIST_LIMIT) -> list[SessionSummary]:
        """List saved sessions, most recent first."""
        limit = min(max(limit, MIN_LIST_LIMIT), MAX_LIST_LIMIT)
        summaries = []

        for session_file in self.sessions_dir.glob("session-*.json"):
            try:
                data = json.loads(session_file.read_text(encoding="utf-8"))
                summaries.append(SessionSummary(
                    session_id=data["session_id"],
                    title=data.get("title", "Untitled"),
                    agent_name=data.get("agent_name", "assistant"),
                    model=data.get("model", ""),
                    message_count=len(data.get("messages", [])),
                    created_at=data.get("created_at", 0),
                    updated_at=data.get("updated_at", 0),
                ))
            except (json.JSONDecodeError, OSError, KeyError) as e:
                log.debug("Skipping corrupted session file %s: %s", session_file.name, e)
                continue

        # Sort by updated_at descending
        summaries.sort(key=lambda s: s.updated_at, reverse=True)
        return summaries[:limit]

    def delete(self, session_id: str) -> bool:
        """Delete a saved session."""
        session_path = self.sessions_dir / f"{session_id}.json"
        if session_path.exists():
            session_path.unlink()
            log.info("Deleted session %s", session_id)
            return True

        # Try partial match
        matches = list(self.sessions_dir.glob(f"{session_id}*.json"))
        if len(matches) == 1:
            matches[0].unlink()
            return True

        return False

    def rename(self, session_id: str, new_title: str) -> bool:
        """Rename a saved session's title.

        Args:
            session_id: Session to rename (supports partial ID match).
            new_title: New title for the session.

        Returns:
            True if renamed successfully, False otherwise.
        """
        session = self.load(session_id)
        if not session:
            return False

        session.title = new_title.strip()[:MAX_TITLE_LENGTH]
        session_path = self.sessions_dir / f"{session.session_id}.json"
        content = json.dumps(asdict(session), indent=2)
        atomic_write(session_path, content)
        log.info("Renamed session %s to '%s'", session.session_id, session.title)
        return True

    def export(self, session_id: str, export_format: str = "markdown") -> str:
        """Export a session as markdown, JSON, or HTML.

        Args:
            session_id: Session to export.
            export_format: "markdown", "json", or "html".

        Returns:
            Formatted string.
        """
        if export_format not in VALID_EXPORT_FORMATS:
            return f"Invalid format '{export_format}'. Must be one of: {', '.join(sorted(VALID_EXPORT_FORMATS))}"

        session = self.load(session_id)
        if not session:
            return f"Session not found: {session_id}"

        if export_format == "json":
            return json.dumps(asdict(session), indent=2)

        if export_format == "html":
            return self._export_html(session)

        # Markdown format
        lines = [
            f"# Chat Session: {session.title}",
            "",
            f"- **Agent**: {session.agent_name}",
            f"- **Model**: {session.model}",
            f"- **Messages**: {session.message_count}",
            "",
            "---",
            "",
        ]

        for msg in session.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lines.append(f"*System: {content[:MAX_SYSTEM_CONTENT_DISPLAY]}...*")
            elif role == "user":
                lines.append(f"**User:** {content}")
            elif role == "assistant":
                lines.append(f"**Assistant:** {content}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _export_html(session: Session) -> str:
        """Export a session as a standalone HTML file."""
        import html as html_mod
        from datetime import datetime

        created = datetime.fromtimestamp(session.created_at).strftime("%Y-%m-%d %H:%M")

        messages_html = []
        for msg in session.messages:
            role = msg.get("role", "user")
            content = html_mod.escape(msg.get("content", ""))
            # Preserve newlines and code blocks
            content = content.replace("\n", "<br>")
            messages_html.append(
                f'<div class="message {role}">'
                f'<span class="role">{role}</span>'
                f'<div class="content">{content}</div>'
                f'</div>'
            )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html_mod.escape(session.title)}</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; background: #1a1a2e; color: #e0e0e0; }}
h1 {{ color: #00bfa6; }}
.meta {{ color: #888; margin-bottom: 20px; }}
.message {{ margin: 16px 0; padding: 12px 16px; border-radius: 8px; }}
.message.user {{ background: #1e3a5f; border-left: 3px solid #4a9eff; }}
.message.assistant {{ background: #1a3330; border-left: 3px solid #00bfa6; }}
.message.system {{ background: #2a2a1a; border-left: 3px solid #ffa726; font-style: italic; }}
.role {{ font-weight: bold; font-size: 0.85em; text-transform: uppercase; color: #888; }}
.content {{ margin-top: 6px; line-height: 1.6; }}
</style>
</head>
<body>
<h1>{html_mod.escape(session.title)}</h1>
<div class="meta">
Agent: {html_mod.escape(session.agent_name)} | Model: {html_mod.escape(session.model)} | {session.message_count} messages | {created}
</div>
{''.join(messages_html)}
</body>
</html>"""

    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        """Search across all sessions for messages matching a query.

        Args:
            query: Text to search for (case-insensitive).
            limit: Maximum results to return.

        Returns:
            List of dicts with session_id, title, role, content, match_line.
        """
        if not query or not query.strip():
            return []
        limit = min(max(limit, 1), MAX_SEARCH_LIMIT)
        query_lower = query.lower()
        results = []

        for session_file in self.sessions_dir.glob("session-*.json"):
            try:
                data = json.loads(session_file.read_text(encoding="utf-8"))
                session_id = data.get("session_id", "")
                title = data.get("title", "Untitled")

                for msg in data.get("messages", []):
                    content = msg.get("content", "")
                    if query_lower in content.lower():
                        # Extract matching line
                        for line in content.splitlines():
                            if query_lower in line.lower():
                                results.append({
                                    "session_id": session_id,
                                    "title": title,
                                    "role": msg.get("role", ""),
                                    "content": line.strip()[:MAX_SEARCH_RESULT_CONTENT],
                                })
                                if len(results) >= limit:
                                    return results
                                break
            except (json.JSONDecodeError, OSError) as e:
                log.debug("Skipping unreadable session file %s during search: %s", session_file.name, e)
                continue

        return results

    def cleanup_old_sessions(self) -> int:
        """Remove oldest sessions if total count exceeds MAX_SESSIONS_ON_DISK.

        Returns number of sessions removed.
        """
        session_files = sorted(
            self.sessions_dir.glob("session-*.json"),
            key=lambda f: f.stat().st_mtime,
        )
        if len(session_files) <= MAX_SESSIONS_ON_DISK:
            return 0

        to_remove = session_files[:len(session_files) - MAX_SESSIONS_ON_DISK]
        for f in to_remove:
            try:
                f.unlink()
            except OSError as e:
                log.debug("Could not remove old session %s: %s", f.name, e)
        log.info("Cleaned up %d old sessions", len(to_remove))
        return len(to_remove)

    def _cleanup_orphaned_files(self) -> int:
        """Remove orphaned .tmp and .corrupted files from the sessions directory.

        Returns number of files removed.
        """
        removed = 0
        for pattern in ("*.json.tmp", "*.json.corrupted"):
            for f in self.sessions_dir.glob(pattern):
                try:
                    f.unlink()
                    removed += 1
                except OSError as e:
                    log.debug("Could not remove orphaned file %s: %s", f.name, e)
        if removed:
            log.info("Cleaned up %d orphaned file(s)", removed)
        return removed

    def get_stats(self) -> dict[str, Any]:
        """Get session storage statistics."""
        session_files = list(self.sessions_dir.glob("session-*.json"))
        total_size = sum(f.stat().st_size for f in session_files if f.exists())
        return {
            "session_count": len(session_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }

    @staticmethod
    def _generate_title(messages: list[dict[str, str]]) -> str:
        """Generate a title from the first user message."""
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "").strip()
                # Take first line, limit to 50 chars
                first_line = content.split("\n")[0][:MAX_TITLE_LENGTH]
                return first_line if first_line else "Untitled"
        return "Untitled"
