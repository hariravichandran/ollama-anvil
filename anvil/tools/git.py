"""Git tool: repository operations for agents.

Includes auto-commit, undo, branch management, and LLM-generated commit messages.
Agents can make changes and auto-commit with descriptive messages, and users
can undo agent commits with a single command.
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from anvil.utils.logging import get_logger

if TYPE_CHECKING:
    from anvil.llm.client import OllamaClient

__all__ = [
    "GitTool",
    "AGENT_COMMIT_TAG",
    "BRANCH_NAME_PATTERN",
    "MAX_DIFF_SIZE",
    "MAX_LOG_COUNT",
    "MIN_LOG_COUNT",
    "LOG_PREVIEW_LENGTH",
    "TIMEOUT_FAST",
    "TIMEOUT_NORMAL",
    "TIMEOUT_SLOW",
]

# Pre-compiled regex patterns
_COMMIT_REF_RE = re.compile(r'^[a-zA-Z0-9_.~^/@{}\-]+$')
_CONFLICT_MARKER_RE = re.compile(r"^(<{7}|={7}|>{7})", re.MULTILINE)

log = get_logger("tools.git")

# Tag prefix for agent-generated commits (used by undo)
AGENT_COMMIT_TAG = "[anvil]"

# Safety limits
MAX_DIFF_SIZE = 500_000  # 500K chars max for diff output
MAX_LOG_COUNT = 500  # Max commits to show in log
MIN_LOG_COUNT = 1
LOG_PREVIEW_LENGTH = 200  # Truncate log previews

# Branch name validation: alphanumeric, hyphens, underscores, dots, slashes
BRANCH_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._/-]{0,99}$")

# Operation-specific timeouts (seconds)
TIMEOUT_FAST = 10    # status, branch
TIMEOUT_NORMAL = 30  # diff, log, commit
TIMEOUT_SLOW = 120   # pull, push, revert


class GitTool:
    """Git operations for agents.

    Features:
    - Basic: status, diff, log, commit
    - Auto-commit: stage + commit with LLM-generated message after edits
    - Undo: revert the last agent commit safely (git revert, not reset)
    - Branch: create/switch branches per task
    - Stash: save/restore work in progress
    """

    name = "git"
    description = "Git repository operations (status, diff, log, commit, undo, branch)"

    def __init__(self, working_dir: str = ".", client: OllamaClient | None = None):
        self.working_dir = working_dir
        self.client = client

    def __repr__(self) -> str:
        return f"GitTool(working_dir={self.working_dir!r})"

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return Ollama tool-calling definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "git_status",
                    "description": "Show the working tree status (modified, staged, untracked files)",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_diff",
                    "description": "Show changes in the working directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "staged": {"type": "boolean", "description": "Show staged changes (default: unstaged)"},
                            "file": {"type": "string", "description": "Specific file to diff (optional)"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_log",
                    "description": "Show recent commit history",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "count": {"type": "integer", "description": "Number of commits (default 10)"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_commit",
                    "description": "Stage and commit changes with a descriptive message",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string", "description": "Commit message"},
                            "files": {"type": "array", "items": {"type": "string"}, "description": "Files to stage (default: all modified)"},
                        },
                        "required": ["message"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_undo",
                    "description": "Undo the last agent-made commit using git revert (safe, creates a new commit)",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_create_branch",
                    "description": "Create and switch to a new branch for a task",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Branch name (e.g., 'fix/auth-bug')"},
                        },
                        "required": ["name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_stash",
                    "description": "Stash or restore work in progress",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["save", "pop", "list"], "description": "Stash action"},
                            "message": {"type": "string", "description": "Stash message (for save)"},
                        },
                        "required": ["action"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_show",
                    "description": "Show details of a specific commit (message, author, changed files, diff)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "commit": {"type": "string", "description": "Commit hash or ref (e.g., 'HEAD', 'abc1234', 'main~3')"},
                            "stat_only": {"type": "boolean", "description": "Show only file change stats, not full diff (default: false)"},
                        },
                        "required": ["commit"],
                    },
                },
            },
        ]

    def execute(self, function_name: str, args: dict[str, Any]) -> str:
        """Execute a git tool function."""
        handlers = {
            "git_status": self._status,
            "git_diff": self._diff,
            "git_log": self._log,
            "git_commit": self._commit,
            "git_undo": self._undo,
            "git_create_branch": self._create_branch,
            "git_stash": self._stash,
            "git_show": self._show,
        }
        handler = handlers.get(function_name)
        if not handler:
            return f"Unknown function: {function_name}"
        try:
            return handler(**args)
        except (TypeError, ValueError, OSError, KeyError) as e:
            return f"Error: {e}"

    def _run_git(self, *args: str, timeout: int = 30) -> str:
        """Run a git command and return output, with timing."""
        cmd = ["git"] + list(args)
        start = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=self.working_dir,
        )
        elapsed = time.time() - start
        if elapsed > 5:
            log.info("Slow git command (%.1fs): %s", elapsed, " ".join(args[:3]))
        output = result.stdout
        if result.stderr and result.returncode != 0:
            output += f"\n[stderr]: {result.stderr}"
        return output.strip() or "(no output)"

    def _validate_file_path(self, path: str) -> str:
        """Validate a file path stays within the working directory.

        Returns error string or empty if valid.
        """
        if not path:
            return ""
        try:
            resolved = Path(self.working_dir).resolve() / path
            resolved = resolved.resolve()
            if not str(resolved).startswith(str(Path(self.working_dir).resolve())):
                return f"Error: path '{path}' resolves outside working directory"
        except (OSError, ValueError):
            return f"Error: invalid path '{path}'"
        return ""

    def _status(self) -> str:
        return self._run_git("status", "--short")

    def _diff(self, staged: bool = False, file: str = "") -> str:
        file = file.strip()
        if file:
            path_err = self._validate_file_path(file)
            if path_err:
                return path_err
        args = ["diff"]
        if staged:
            args.append("--cached")
        if file:
            args.extend(["--", file])
        output = self._run_git(*args)
        # Truncate oversized diffs
        if len(output) > MAX_DIFF_SIZE:
            log.warning("Diff output truncated from %d to %d chars", len(output), MAX_DIFF_SIZE)
            output = output[:MAX_DIFF_SIZE] + "\n... (diff truncated, use file-specific diff for full output)"
        return output

    def _log(self, count: int = 10) -> str:
        count = max(MIN_LOG_COUNT, min(count, MAX_LOG_COUNT))
        return self._run_git("log", "--oneline", f"-{count}")

    def _show(self, commit: str, stat_only: bool = False) -> str:
        """Show details of a specific commit.

        Args:
            commit: Commit hash or ref (HEAD, abc1234, main~3).
            stat_only: If True, show only file change stats instead of full diff.
        """
        # Sanitize commit ref — only allow safe characters
        if not _COMMIT_REF_RE.match(commit):
            return f"Invalid commit reference: {commit}"
        if stat_only:
            output = self._run_git("show", "--stat", commit, timeout=TIMEOUT_NORMAL)
        else:
            output = self._run_git("show", commit, timeout=TIMEOUT_NORMAL)
        # Truncate oversized output
        if len(output) > MAX_DIFF_SIZE:
            log.warning("git show output truncated from %d to %d chars", len(output), MAX_DIFF_SIZE)
            output = output[:MAX_DIFF_SIZE] + "\n... (output truncated, use stat_only=true for summary)"
        return output

    def _commit(self, message: str, files: list[str] | None = None) -> str:
        message = message.strip()
        if not message:
            return "Error: commit message cannot be empty"
        # Check for unresolved merge conflicts before committing
        conflict_check = self._check_conflicts()
        if conflict_check:
            return conflict_check

        if files:
            for f in files:
                path_err = self._validate_file_path(f)
                if path_err:
                    return path_err
                self._run_git("add", f)
        else:
            self._run_git("add", "-A")

        # Tag agent commits for undo tracking
        full_message = f"{AGENT_COMMIT_TAG} {message}"
        return self._run_git("commit", "-m", full_message)

    def _check_conflicts(self) -> str:
        """Check for unresolved merge conflicts in tracked files.

        Checks both git status indicators (UU, AA, DD) and conflict
        markers in file content (<<<<<<<, =======, >>>>>>>).

        Returns an error message if conflicts found, empty string if clean.
        """
        status = self._run_git("status", "--porcelain")
        if status == "(no output)":
            return ""

        conflict_files = []
        modified_files = []
        for line in status.splitlines():
            # 'UU', 'AA', 'DD' etc. indicate unmerged files
            if line[:2] in ("UU", "AA", "DD", "AU", "UA", "DU", "UD"):
                conflict_files.append(line[3:].strip())
            # Track modified files for marker check
            elif line[:2].strip():
                modified_files.append(line[3:].strip())

        # Also check modified files for conflict markers in content
        marker_conflicts = self._check_conflict_markers(modified_files)
        conflict_files.extend(marker_conflicts)

        if conflict_files:
            unique = list(dict.fromkeys(conflict_files))  # dedup preserving order
            files_str = ", ".join(unique[:5])
            return f"Cannot commit: unresolved merge conflicts in {files_str}. Resolve conflicts first."
        return ""

    def _check_conflict_markers(self, files: list[str]) -> list[str]:
        """Check file content for git conflict markers (<<<<<<<, >>>>>>>).

        Returns list of files containing unresolved conflict markers.
        """
        conflict_re = _CONFLICT_MARKER_RE
        marker_files = []
        # Binary extensions to skip (avoid false positives from binary content)
        binary_exts = {".pyc", ".so", ".o", ".exe", ".bin", ".png", ".jpg", ".gif",
                       ".zip", ".tar", ".gz", ".pdf", ".db", ".sqlite"}
        for f in files:
            file_path = Path(self.working_dir) / f
            if not file_path.exists() or not file_path.is_file():
                continue
            if file_path.suffix.lower() in binary_exts:
                continue
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                if conflict_re.search(content):
                    marker_files.append(f)
            except (OSError, PermissionError):
                continue
        return marker_files

    def _undo(self) -> str:
        """Undo the last agent commit using git revert (safe).

        Only reverts commits tagged with AGENT_COMMIT_TAG.
        """
        # Find the last agent commit
        log_output = self._run_git("log", "--oneline", "-20")
        for line in log_output.splitlines():
            if AGENT_COMMIT_TAG in line:
                parts = line.split()
                if not parts:
                    continue
                commit_hash = parts[0]
                if not _COMMIT_REF_RE.match(commit_hash):
                    log.warning("Invalid commit hash format in undo: %s", commit_hash)
                    continue
                result = self._run_git("revert", "--no-edit", commit_hash)
                return f"Reverted agent commit {commit_hash}: {result}"

        return "No agent commits found to undo"

    @staticmethod
    def validate_branch_name(name: str) -> str:
        """Validate a branch name. Returns error string or empty if valid."""
        if not name or not name.strip():
            return "Branch name cannot be empty"
        if name.startswith("-"):
            return "Branch name cannot start with '-'"
        if ".." in name:
            return "Branch name cannot contain '..'"
        if name.endswith(".lock"):
            return "Branch name cannot end with '.lock'"
        if not BRANCH_NAME_PATTERN.match(name):
            return f"Invalid branch name: {name}"
        return ""

    def _create_branch(self, name: str) -> str:
        """Create and switch to a new branch.

        Validates branch name and checks if it already exists.
        """
        # Validate branch name
        validation_err = self.validate_branch_name(name)
        if validation_err:
            return f"Error: {validation_err}"

        # Check if branch already exists
        existing = self._run_git("branch", "--list", name)
        if existing and existing != "(no output)":
            return f"Branch '{name}' already exists. Use 'git checkout {name}' to switch to it."
        result = self._run_git("checkout", "-b", name)
        return result

    def _stash(self, action: str, message: str = "") -> str:
        """Stash operations."""
        message = message.strip()
        if action == "save":
            args = ["stash", "push"]
            if message:
                args.extend(["-m", message])
            return self._run_git(*args)
        if action == "pop":
            return self._run_git("stash", "pop")
        if action == "list":
            return self._run_git("stash", "list")
        return f"Error: unknown stash action: {action}"

    # --- Auto-commit workflow ---

    def auto_commit(self, files_changed: list[str], description: str = "") -> str:
        """Auto-commit changes with an LLM-generated commit message.

        Args:
            files_changed: List of file paths that were modified.
            description: Brief description of what was changed (helps LLM).

        Returns:
            Commit result string.
        """
        # Check if there are actual changes
        status = self._status()
        if "nothing to commit" in status.lower() or status == "(no output)":
            return "Nothing to commit"

        # Generate commit message
        message = self._generate_commit_message(files_changed, description)

        # Stage specific files
        for f in files_changed:
            self._run_git("add", f)

        # Commit with agent tag
        full_message = f"{AGENT_COMMIT_TAG} {message}"
        result = self._run_git("commit", "-m", full_message)
        log.info("Auto-committed: %s", message)
        return result

    def _generate_commit_message(self, files_changed: list[str], description: str) -> str:
        """Generate a commit message using LLM or from description."""
        if not self.client:
            # No LLM — use description or generic message
            if description:
                return description[:72]
            return f"Update {', '.join(f[:30] for f in files_changed[:3])}"

        # Get the diff for context
        diff = self._run_git("diff", "--cached", "--stat")
        if diff == "(no output)":
            # Stage first to get diff
            for f in files_changed:
                self._run_git("add", f)
            diff = self._run_git("diff", "--cached", "--stat")

        prompt = (
            f"Write a concise git commit message (max 72 chars first line) for these changes.\n"
            f"Files: {', '.join(files_changed)}\n"
            f"Description: {description}\n"
            f"Diff stats:\n{diff}\n\n"
            f"Reply with ONLY the commit message, no quotes or explanation."
        )

        try:
            result = self.client.generate(
                prompt=prompt,
                system="You write concise, conventional git commit messages. Use imperative mood (e.g., 'Add feature', 'Fix bug'). Max 72 chars first line.",
                timeout=30,
                temperature=0.3,
            )
            if result.get("error"):
                log.warning("LLM error generating commit message: %s", str(result["error"])[:LOG_PREVIEW_LENGTH])
                return description[:72] if description else "Update files"
            message = result.get("response", "").strip()
            # Clean up: remove quotes, limit length
            message = message.strip('"\'')
            first_line = message.splitlines()[0][:72] if message else ""
            # Validate the generated message
            first_line = self._validate_commit_message(first_line)
            return first_line or description[:72] or "Update files"
        except (OSError, TimeoutError, ValueError) as e:
            log.debug("LLM commit message generation failed: %s", e)
            return description[:72] if description else "Update files"

    @staticmethod
    def _validate_commit_message(message: str) -> str:
        """Validate and clean a commit message.

        Ensures:
        - Non-empty after stripping
        - First line <= 72 chars
        - No excessive repetition
        - Strips trailing periods (conventional commits style)
        """
        message = message.strip()
        if not message:
            return ""

        # Truncate first line
        first_line = message.splitlines()[0][:72]

        # Detect excessive repetition (e.g., "update update update")
        words = first_line.lower().split()
        if words and len(set(words)) == 1 and len(words) > 2:
            return ""  # reject repetitive messages

        # Strip trailing period (conventional commits style)
        first_line = first_line.rstrip(".")

        return first_line

    def get_current_branch(self) -> str:
        """Get the current branch name."""
        return self._run_git("rev-parse", "--abbrev-ref", "HEAD")

    def has_uncommitted_changes(self) -> bool:
        """Check if there are uncommitted changes."""
        status = self._run_git("status", "--porcelain")
        return bool(status and status != "(no output)")

    def get_agent_commits(self, count: int = 10) -> list[str]:
        """Get recent agent-made commits."""
        log_output = self._run_git("log", "--oneline", f"-{count * 2}")
        return [
            line for line in log_output.splitlines()
            if AGENT_COMMIT_TAG in line
        ][:count]
