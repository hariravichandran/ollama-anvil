"""Filesystem tool: read, write, edit, glob, grep files.

Includes fuzzy matching for edit operations — when local LLMs produce
slightly incorrect search strings (whitespace, indentation), the fuzzy
matcher finds the best match instead of failing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from anvil.utils.fileio import atomic_write
from anvil.utils.logging import get_logger

__all__ = [
    "FilesystemTool",
    "FUZZY_MATCH_THRESHOLD",
    "MAX_READ_SIZE",
    "MAX_WRITE_SIZE",
    "BINARY_EXTENSIONS",
    "MAX_SEARCH_RESULTS",
    "MAX_LIST_RESULTS",
    "MAX_REGEX_PATTERN_LENGTH",
    "BINARY_CHECK_SIZE",
]

log = get_logger("tools.filesystem")

# Minimum similarity ratio for fuzzy matching (0.0 to 1.0)
FUZZY_MATCH_THRESHOLD = 0.75

# Maximum file size for reading (10 MB)
MAX_READ_SIZE = 10 * 1024 * 1024

# Maximum file size for writing (10 MB)
MAX_WRITE_SIZE = 10 * 1024 * 1024

# Maximum search results
MAX_SEARCH_RESULTS = 100

# Maximum file listing results
MAX_LIST_RESULTS = 100

# Regex compile timeout safety — reject patterns over this length
MAX_REGEX_PATTERN_LENGTH = 500

# Sample size for binary detection (bytes)
BINARY_CHECK_SIZE = 8192

# Binary file extensions — refuse to read as text
BINARY_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".dll", ".dylib", ".o", ".a",
    ".exe", ".bin", ".dat", ".db", ".sqlite", ".sqlite3",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svg",
    ".mp3", ".mp4", ".avi", ".mkv", ".wav", ".flac", ".ogg",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".woff", ".woff2", ".ttf", ".eot",
    ".class", ".jar", ".war",
}


class FilesystemTool:
    """File operations for agents."""

    name = "filesystem"
    description = "Read, write, edit, search, and list files"

    def __init__(self, working_dir: str = "."):
        self.working_dir = Path(working_dir).resolve()

    def __repr__(self) -> str:
        return f"FilesystemTool(working_dir={str(self.working_dir)!r})"

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return Ollama tool-calling definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path (relative to working directory)"},
                            "start_line": {"type": "integer", "description": "Start line (1-indexed, optional)"},
                            "end_line": {"type": "integer", "description": "End line (1-indexed, optional)"},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file (creates or overwrites)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path"},
                            "content": {"type": "string", "description": "Content to write"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "description": "Replace a specific string in a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path"},
                            "old_string": {"type": "string", "description": "Exact text to find and replace"},
                            "new_string": {"type": "string", "description": "Replacement text"},
                        },
                        "required": ["path", "old_string", "new_string"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files matching a glob pattern",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string", "description": "Glob pattern (e.g., '**/*.py')"},
                        },
                        "required": ["pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": "Search file contents for a regex pattern",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string", "description": "Regex pattern to search for"},
                            "glob": {"type": "string", "description": "File glob to limit search (e.g., '*.py')"},
                            "max_results": {"type": "integer", "description": "Maximum results (default 20)"},
                            "context_lines": {"type": "integer", "description": "Number of lines to show before and after each match (default 0, max 10)"},
                        },
                        "required": ["pattern"],
                    },
                },
            },
        ]

    def execute(self, function_name: str, args: dict[str, Any]) -> str:
        """Execute a filesystem tool function."""
        handlers = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "edit_file": self._edit_file,
            "list_files": self._list_files,
            "search_files": self._search_files,
        }
        handler = handlers.get(function_name)
        if not handler:
            return f"Unknown function: {function_name}"
        try:
            return handler(**args)
        except (TypeError, ValueError, OSError, KeyError) as e:
            return f"Error: {e}"

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to working directory, with safety checks.

        Validates against directory traversal and symlink escapes.
        """
        target = self.working_dir / path
        resolved = target.resolve()
        # Prevent directory traversal outside working directory
        if not str(resolved).startswith(str(self.working_dir)):
            raise ValueError(f"Path escapes working directory: {path}")
        # Symlink safety: if the original path is a symlink, verify its
        # resolved target is still within the working directory
        if target.is_symlink():
            link_target = target.resolve()
            if not str(link_target).startswith(str(self.working_dir)):
                raise ValueError(f"Symlink target escapes working directory: {path}")
        # Hard link safety: detect files with multiple hard links
        # (a hard link could point outside the working directory)
        if resolved.exists() and resolved.is_file():
            try:
                if resolved.stat().st_nlink > 1:
                    log.debug("File has %d hard links: %s", resolved.stat().st_nlink, path)
            except OSError:
                pass  # stat() may fail on special files; safe to skip
        return resolved

    @staticmethod
    def _is_binary(path: Path) -> bool:
        """Check if a file is likely binary based on extension and content.

        Uses extension check first (fast), then falls back to reading
        a small sample and checking for null bytes.
        """
        if path.suffix.lower() in BINARY_EXTENSIONS:
            return True
        # Check first 8KB for null bytes (heuristic for binary content)
        try:
            with open(path, "rb") as f:
                chunk = f.read(BINARY_CHECK_SIZE)
                return b"\x00" in chunk
        except (OSError, PermissionError):
            return False

    def _read_file(self, path: str, start_line: int = 0, end_line: int = 0) -> str:
        resolved = self._resolve_path(path)
        if not resolved.exists():
            return f"File not found: {path}"
        if resolved.is_dir():
            return f"Path is a directory: {path}"

        # Check file size
        file_size = resolved.stat().st_size
        if file_size > MAX_READ_SIZE:
            size_mb = file_size / (1024 * 1024)
            return f"File too large: {path} ({size_mb:.1f} MB, max {MAX_READ_SIZE // (1024*1024)} MB)"

        # Check for binary files
        if self._is_binary(resolved):
            return f"Cannot read binary file: {path} (extension: {resolved.suffix})"

        content = resolved.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()

        if start_line or end_line:
            start = max(0, start_line - 1)
            end = min(end_line, len(lines)) if end_line else len(lines)
            lines = lines[start:end]
            numbered = [f"{i + start + 1:4d}  {line}" for i, line in enumerate(lines)]
            return "\n".join(numbered)

        # Add line numbers
        numbered = [f"{i + 1:4d}  {line}" for i, line in enumerate(lines)]
        return "\n".join(numbered)

    def _write_file(self, path: str, content: str) -> str:
        # Validate path first (consistent with _read_file and _edit_file)
        resolved = self._resolve_path(path)

        # Validate content size
        content_bytes = len(content.encode("utf-8", errors="replace"))
        if content_bytes > MAX_WRITE_SIZE:
            size_mb = content_bytes / (1024 * 1024)
            return f"Content too large to write: {size_mb:.1f} MB (max {MAX_WRITE_SIZE // (1024*1024)} MB)"

        atomic_write(resolved, content)
        return f"Written {len(content)} chars to {path}"

    def _edit_file(self, path: str, old_string: str, new_string: str) -> str:
        resolved = self._resolve_path(path)
        if not resolved.exists():
            return f"File not found: {path}"

        content = resolved.read_text(encoding="utf-8")

        # Try exact match first
        count = content.count(old_string)
        if count == 1:
            new_content = content.replace(old_string, new_string, 1)
            self._atomic_write(resolved, new_content)
            return f"Replaced 1 occurrence in {path}"

        if count > 1:
            return f"String found {count} times in {path} — provide more context to make it unique"

        # Exact match failed — try fuzzy matching
        # This handles whitespace/indentation mismatches from local LLMs
        match_result = self._fuzzy_find(content, old_string)
        if match_result:
            actual_text, similarity = match_result
            new_content = content.replace(actual_text, new_string, 1)
            self._atomic_write(resolved, new_content)
            return f"Replaced 1 occurrence in {path} (fuzzy match, {similarity:.0%} similar)"

        return f"String not found in {path} (exact and fuzzy match failed)"

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        """Write content atomically via temp file + rename."""
        atomic_write(path, content)

    @staticmethod
    def _fuzzy_find(content: str, search: str) -> tuple[str, float] | None:
        """Find the closest match to search string in content using fuzzy matching.

        Returns (actual_text, similarity_ratio) or None if no good match found.
        """
        import difflib
        search_lines = search.splitlines()
        content_lines = content.splitlines()
        n_search = len(search_lines)

        if n_search == 0:
            return None

        best_match = None
        best_ratio = 0.0

        # Slide a window of search_lines length over content_lines
        for i in range(len(content_lines) - n_search + 1):
            window = content_lines[i:i + n_search]
            window_text = "\n".join(window)

            # Quick length check to avoid expensive comparison
            len_ratio = len(search) / max(len(window_text), 1)
            if len_ratio < 0.5 or len_ratio > 2.0:
                continue

            ratio = difflib.SequenceMatcher(None, search, window_text).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = window_text

        if best_match and best_ratio >= FUZZY_MATCH_THRESHOLD:
            return best_match, best_ratio

        return None

    def _list_files(self, pattern: str) -> str:
        matches = sorted(self.working_dir.glob(pattern))
        # Limit results
        total = len(matches)
        if total > MAX_LIST_RESULTS:
            matches = matches[:MAX_LIST_RESULTS]
            suffix = f"\n... and {total - MAX_LIST_RESULTS} more"
        else:
            suffix = ""

        lines = [str(m.relative_to(self.working_dir)) for m in matches]
        return "\n".join(lines) + suffix if lines else "No files found"

    def _search_files(
        self, pattern: str, glob: str = "**/*", max_results: int = 20, context_lines: int = 0,
    ) -> str:
        """Search file contents for a regex pattern.

        Args:
            pattern: Regex pattern to search for.
            glob: File glob to limit search scope.
            max_results: Maximum number of matches to return.
            context_lines: Number of lines to show before and after each match (0-10).
        """
        # Validate regex pattern length to prevent ReDoS
        if len(pattern) > MAX_REGEX_PATTERN_LENGTH:
            return f"Error: search pattern too long ({len(pattern)} chars, max {MAX_REGEX_PATTERN_LENGTH})"
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"Invalid regex pattern: {e}"

        # Clamp max_results and context_lines
        max_results = min(max_results, MAX_SEARCH_RESULTS)
        context_lines = max(0, min(context_lines, 10))
        results: list[str] = []

        for file_path in self.working_dir.glob(glob):
            if not file_path.is_file():
                continue
            # Skip binary files in search
            if self._is_binary(file_path):
                continue
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()
                rel = file_path.relative_to(self.working_dir)
                for i, line in enumerate(lines):
                    if regex.search(line):
                        line_num = i + 1
                        if context_lines > 0:
                            # Show context before and after the match
                            ctx_start = max(0, i - context_lines)
                            ctx_end = min(len(lines), i + context_lines + 1)
                            ctx_parts = []
                            for j in range(ctx_start, ctx_end):
                                marker = ">" if j == i else " "
                                ctx_parts.append(f"  {marker} {j + 1:4d} | {lines[j]}")
                            results.append(f"{rel}:{line_num}:\n" + "\n".join(ctx_parts))
                        else:
                            results.append(f"{rel}:{line_num}: {line.strip()}")
                        if len(results) >= max_results:
                            return "\n".join(results) + f"\n... (limited to {max_results} results)"
            except (PermissionError, IsADirectoryError):
                continue

        return "\n".join(results) if results else "No matches found"
