"""Codebase indexing and semantic search.

Indexes project files for fast retrieval by agents. Extracts symbols (functions,
classes, imports), builds a file-level summary index, and supports fuzzy and
semantic search so agents can quickly find relevant code.

This is the key differentiator for coding assistants — the agent needs to
understand the full codebase, not just the file it's editing.

Index storage: .anvil/index/ in the project root (gitignored).
"""

from __future__ import annotations

import fnmatch as fnm
import hashlib
import json
import os
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from anvil.utils.logging import get_logger

if TYPE_CHECKING:
    from anvil.llm.client import OllamaClient

__all__ = [
    "Symbol",
    "FileIndex",
    "SearchResult",
    "CodebaseIndexer",
    "CodebaseTool",
    "DEFAULT_IGNORE",
    "CODE_EXTENSIONS",
    "MAX_FILE_SIZE",
    "MAX_SEARCH_RESULTS",
    "MAX_SEARCH_QUERY_LENGTH",
    "MAX_SYMBOL_DISPLAY",
    "MAX_IMPORT_DISPLAY",
    "MAX_OVERVIEW_FILES",
    "MAX_DOCSTRING_CONTEXT",
    "MAX_CONTEXT_PREVIEW",
    "MAX_SIGNATURE_LENGTH",
    "MAX_SUMMARY_CONTENT",
    "MAX_SUMMARY_LENGTH",
    "MAX_SUMMARY_FILE_SIZE",
]

log = get_logger("tools.codebase")

# Pre-compiled regex patterns for symbol extraction (avoids recompilation per file)
_PY_CLASS_RE = re.compile(r'^class\s+(\w+)(?:\(([^)]*)\))?\s*:')
_PY_DEF_RE = re.compile(r'^(\s*)def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*(.+))?\s*:')
_JS_FUNC_RE = re.compile(r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)')
_JS_ARROW_RE = re.compile(r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|(\w+))\s*=>')
_JS_CLASS_RE = re.compile(r'(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?\s*\{')
_TS_IFACE_RE = re.compile(r'(?:export\s+)?(?:interface|type)\s+(\w+)')
_GO_FUNC_RE = re.compile(r'func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(([^)]*)\)')
_GO_TYPE_RE = re.compile(r'type\s+(\w+)\s+(struct|interface)')
_RUST_FN_RE = re.compile(r'(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)')
_RUST_TYPE_RE = re.compile(r'(?:pub\s+)?(?:struct|enum|trait)\s+(\w+)')
_GENERIC_FUNC_RE = re.compile(r'(?:function|def|fn|func|sub|proc)\s+(\w+)')
_GENERIC_CLASS_RE = re.compile(r'(?:class|struct|interface|trait|enum|type)\s+(\w+)')

# Default ignore patterns (gitignore-style)
DEFAULT_IGNORE = {
    ".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv",
    ".env", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", ".eggs", "*.egg-info", ".anvil",
}

# File extensions to index
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".c", ".cpp",
    ".h", ".hpp", ".cs", ".rb", ".php", ".swift", ".kt", ".scala", ".lua",
    ".sh", ".bash", ".zsh", ".yaml", ".yml", ".toml", ".json", ".md",
    ".html", ".css", ".scss", ".sql", ".r", ".R", ".jl",
}

# Max file size to index (500KB)
MAX_FILE_SIZE = 500_000

# Search limits
MAX_SEARCH_RESULTS = 100  # hard cap on search results
MAX_SEARCH_QUERY_LENGTH = 500  # max query string length
MAX_SYMBOL_DISPLAY = 20  # symbols shown in file summary
MAX_IMPORT_DISPLAY = 15  # imports shown in file summary
MAX_OVERVIEW_FILES = 200  # max files in project overview
MAX_DOCSTRING_CONTEXT = 200  # max docstring chars in search results
MAX_CONTEXT_PREVIEW = 100  # max context preview chars
MAX_SIGNATURE_LENGTH = 120  # max signature length for display
MAX_SUMMARY_CONTENT = 3000  # max content chars for LLM summary
MAX_SUMMARY_LENGTH = 100  # max summary output chars
MAX_SUMMARY_FILE_SIZE = 10000  # max file size (chars) eligible for LLM summary


@dataclass(slots=True, frozen=True)
class Symbol:
    """A code symbol (function, class, variable, import)."""

    name: str
    kind: str  # "function", "class", "method", "import", "variable", "constant"
    file: str
    line: int
    signature: str = ""  # e.g., "def foo(x: int, y: str) -> bool"
    docstring: str = ""
    parent: str = ""  # parent class for methods


@dataclass(slots=True)
class FileIndex:
    """Index entry for a single file."""

    path: str
    language: str
    size: int
    lines: int
    hash: str  # content hash for staleness detection
    symbols: list[Symbol] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    summary: str = ""  # LLM-generated one-line summary
    indexed_at: float = 0.0


@dataclass(slots=True, frozen=True)
class SearchResult:
    """A search result from the codebase index."""

    file: str
    line: int
    content: str
    symbol: str = ""
    score: float = 0.0
    context: str = ""  # surrounding lines


class CodebaseIndexer:
    """Indexes and searches a codebase for agent context retrieval.

    Usage:
        indexer = CodebaseIndexer("/path/to/project")
        indexer.build_index()  # Full index (first time)
        indexer.update_index()  # Incremental update

        # Search
        results = indexer.search("authentication middleware")
        results = indexer.find_symbol("UserAuth")
        summary = indexer.get_file_summary("src/auth.py")
    """

    def __init__(
        self,
        project_dir: str,
        client: OllamaClient | None = None,
        ignore_patterns: set[str] | None = None,
    ):
        self.project_dir = Path(project_dir).resolve()
        self.client = client
        self.ignore_patterns = ignore_patterns or DEFAULT_IGNORE
        self.index_dir = self.project_dir / ".anvil" / "index"
        self._file_index: dict[str, FileIndex] = {}
        self._symbol_index: dict[str, list[Symbol]] = {}  # name -> symbols
        self._loaded = False
        self._gitignore_cache: list[str] | None = None  # cached gitignore patterns
        self._gitignore_lock = threading.Lock()

    def __repr__(self) -> str:
        files = len(self._file_index)
        symbols = sum(len(v) for v in self._symbol_index.values())
        return f"CodebaseIndexer(dir={self.project_dir!r}, files={files}, symbols={symbols})"

    def build_index(self, generate_summaries: bool = False) -> dict[str, Any]:
        """Build the full codebase index from scratch.

        Args:
            generate_summaries: If True and client is available, generate
                one-line LLM summaries for each file (slower but useful).

        Returns:
            Stats dict with file count, symbol count, duration.
        """
        start = time.time()
        self._file_index.clear()
        self._symbol_index.clear()

        files = self._discover_files()
        for file_path in files:
            try:
                entry = self._index_file(file_path, generate_summaries)
                if entry:
                    self._file_index[entry.path] = entry
                    for sym in entry.symbols:
                        self._symbol_index.setdefault(sym.name.lower(), []).append(sym)
            except (OSError, SyntaxError, UnicodeDecodeError, ValueError) as e:
                log.debug("Skipping %s: %s", file_path, e)

        self._save_index()
        elapsed = time.time() - start

        stats = {
            "files": len(self._file_index),
            "symbols": sum(len(s) for s in self._symbol_index.values()),
            "duration_s": round(elapsed, 2),
        }
        log.info("Indexed %d files, %d symbols in %.1fs", stats["files"], stats["symbols"], elapsed)
        self._loaded = True
        return stats

    def update_index(self) -> dict[str, Any]:
        """Incremental index update — only re-index changed files.

        Returns stats dict.
        """
        if not self._loaded:
            self._load_index()

        files = self._discover_files()
        current_paths = {str(f.relative_to(self.project_dir)) for f in files}

        updated = 0
        removed = 0
        added = 0

        # Remove deleted files
        for path in list(self._file_index.keys()):
            if path not in current_paths:
                self._remove_from_index(path)
                removed += 1

        # Update changed files, add new files
        for file_path in files:
            rel_path = str(file_path.relative_to(self.project_dir))
            content_hash = self._hash_file(file_path)

            existing = self._file_index.get(rel_path)
            if existing and existing.hash == content_hash:
                continue  # unchanged

            # Remove old entry's symbols
            if existing:
                self._remove_from_index(rel_path)
                updated += 1
            else:
                added += 1

            try:
                entry = self._index_file(file_path, generate_summaries=False)
                if entry:
                    self._file_index[entry.path] = entry
                    for sym in entry.symbols:
                        self._symbol_index.setdefault(sym.name.lower(), []).append(sym)
            except (OSError, SyntaxError, UnicodeDecodeError, ValueError) as e:
                log.debug("Skipping %s: %s", file_path, e)

        if updated or removed or added:
            self._save_index()

        return {"updated": updated, "removed": removed, "added": added}

    def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Search the codebase for a query string.

        Searches file paths, symbol names, and file content.
        Returns results sorted by relevance score.
        """
        if not query or not query.strip():
            return []
        if len(query) > MAX_SEARCH_QUERY_LENGTH:
            query = query[:MAX_SEARCH_QUERY_LENGTH]
        max_results = min(max(1, max_results), MAX_SEARCH_RESULTS)

        if not self._loaded:
            self._load_index()

        query_lower = query.lower()
        query_words = set(query_lower.split())
        results: list[SearchResult] = []

        # 1. Symbol name matches (highest priority)
        for name_lower, symbols in self._symbol_index.items():
            score = 0.0
            if name_lower == query_lower:
                score = 10.0
            elif query_lower in name_lower:
                score = 5.0
            elif any(w in name_lower for w in query_words):
                score = 3.0
            else:
                continue

            for sym in symbols:
                results.append(SearchResult(
                    file=sym.file,
                    line=sym.line,
                    content=sym.signature or sym.name,
                    symbol=sym.name,
                    score=score,
                    context=sym.docstring[:MAX_DOCSTRING_CONTEXT] if sym.docstring else "",
                ))

        # 2. File path matches
        for path, entry in self._file_index.items():
            path_lower = path.lower()
            if query_lower in path_lower:
                score = 4.0 if query_lower in Path(path_lower).stem else 2.0
                results.append(SearchResult(
                    file=path,
                    line=1,
                    content=f"[file] {path} ({entry.lines} lines, {entry.language})",
                    score=score,
                    context=entry.summary,
                ))

        # 3. Content search (grep-like, lower priority)
        if len(results) < max_results:
            content_results = self._search_content(query, max_results - len(results))
            results.extend(content_results)

        # Sort by score descending, deduplicate
        results.sort(key=lambda r: r.score, reverse=True)
        seen = set()
        deduped = []
        for r in results:
            key = (r.file, r.line)
            if key not in seen:
                seen.add(key)
                deduped.append(r)
                if len(deduped) >= max_results:
                    break

        return deduped

    def find_symbol(self, name: str) -> list[Symbol]:
        """Find all symbols matching a name (case-insensitive)."""
        if not self._loaded:
            self._load_index()
        return self._symbol_index.get(name.lower(), [])

    def get_file_summary(self, path: str) -> str:
        """Get a summary of a file from the index."""
        if not self._loaded:
            self._load_index()
        entry = self._file_index.get(path)
        if not entry:
            return f"File not indexed: {path}"

        parts = [f"**{path}** ({entry.language}, {entry.lines} lines)"]
        if entry.summary:
            parts.append(f"Summary: {entry.summary}")
        if entry.symbols:
            sym_list = []
            for sym in entry.symbols[:MAX_SYMBOL_DISPLAY]:
                prefix = f"  {sym.kind}: "
                sig = sym.signature if sym.signature else sym.name
                sym_list.append(f"{prefix}{sig}")
            parts.append("Symbols:\n" + "\n".join(sym_list))
        if entry.imports:
            parts.append(f"Imports: {', '.join(entry.imports[:MAX_IMPORT_DISPLAY])}")

        return "\n".join(parts)

    def get_project_overview(self, max_files: int = 50) -> str:
        """Get a high-level overview of the project structure.

        Returns a formatted string with file tree and key symbols.
        """
        if not self._loaded:
            self._load_index()

        if not self._file_index:
            return "No files indexed. Run build_index() first."

        # Group by directory
        dirs: dict[str, list[str]] = {}
        for path in sorted(self._file_index.keys()):
            parent = str(Path(path).parent)
            dirs.setdefault(parent, []).append(Path(path).name)

        lines = [f"Project: {self.project_dir.name}"]
        lines.append(f"Files: {len(self._file_index)}, Symbols: {sum(len(s) for s in self._symbol_index.values())}")
        lines.append("")

        # Language breakdown
        lang_counts = Counter(entry.language for entry in self._file_index.values())
        if lang_counts:
            lang_str = ", ".join(f"{lang}: {count}" for lang, count in
                                 sorted(lang_counts.items(), key=lambda x: -x[1])[:5])
            lines.append(f"Languages: {lang_str}")
            lines.append("")

        # File tree (abbreviated)
        shown = 0
        for dir_path in sorted(dirs.keys()):
            if shown >= max_files:
                lines.append(f"... and {len(self._file_index) - shown} more files")
                break
            files = dirs[dir_path]
            dir_display = dir_path if dir_path != "." else "(root)"
            lines.append(f"{dir_display}/")
            for f in sorted(files):
                if shown >= max_files:
                    break
                entry = self._file_index.get(str(Path(dir_path) / f))
                sym_count = len(entry.symbols) if entry else 0
                lines.append(f"  {f} ({sym_count} symbols)")
                shown += 1

        return "\n".join(lines)

    # --- File discovery ---

    def _discover_files(self) -> list[Path]:
        """Discover all indexable files in the project."""
        files = []
        gitignore_patterns = self._load_gitignore()

        for root, dirs, filenames in os.walk(self.project_dir):
            root_path = Path(root)

            # Skip ignored directories
            dirs[:] = [
                d for d in dirs
                if d not in self.ignore_patterns
                and not d.startswith(".")
                and not self._matches_gitignore(str(root_path / d), gitignore_patterns)
            ]

            for fname in filenames:
                file_path = root_path / fname
                ext = file_path.suffix.lower()

                if ext not in CODE_EXTENSIONS:
                    continue
                if file_path.stat().st_size > MAX_FILE_SIZE:
                    continue
                if self._matches_gitignore(str(file_path), gitignore_patterns):
                    continue

                files.append(file_path)

        return files

    def _load_gitignore(self) -> list[str]:
        """Load .gitignore patterns if present (cached after first call)."""
        with self._gitignore_lock:
            if self._gitignore_cache is not None:
                return self._gitignore_cache
            gitignore = self.project_dir / ".gitignore"
            if not gitignore.exists():
                self._gitignore_cache = []
                return self._gitignore_cache
            patterns = []
            for line in gitignore.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)
            self._gitignore_cache = patterns
            return self._gitignore_cache

    def _matches_gitignore(self, path: str, patterns: list[str]) -> bool:
        """Check if a path matches any gitignore pattern (simplified)."""
        rel = str(Path(path).relative_to(self.project_dir))
        for pattern in patterns:
            # Simple matching: exact name, or fnmatch
            if pattern.rstrip("/") in rel.split(os.sep):
                return True
            if fnm.fnmatch(rel, pattern) or fnm.fnmatch(Path(rel).name, pattern):
                return True
        return False

    # --- File indexing ---

    def _index_file(self, file_path: Path, generate_summaries: bool = False) -> FileIndex | None:
        """Index a single file: extract symbols, imports, metadata."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeDecodeError) as e:
            log.debug("Cannot read %s for indexing: %s", file_path, e)
            return None

        rel_path = str(file_path.relative_to(self.project_dir))
        ext = file_path.suffix.lower()
        language = self._detect_language(ext)
        lines = content.splitlines()

        entry = FileIndex(
            path=rel_path,
            language=language,
            size=len(content),
            lines=len(lines),
            hash=hashlib.md5(content.encode(), usedforsecurity=False).hexdigest(),
            indexed_at=time.time(),
        )

        # Extract symbols based on language
        if language == "python":
            entry.symbols = self._extract_python_symbols(content, rel_path)
            entry.imports = self._extract_python_imports(content)
        elif language in ("javascript", "typescript"):
            entry.symbols = self._extract_js_symbols(content, rel_path)
        elif language == "go":
            entry.symbols = self._extract_go_symbols(content, rel_path)
        elif language == "rust":
            entry.symbols = self._extract_rust_symbols(content, rel_path)
        else:
            # Generic: just extract function-like patterns
            entry.symbols = self._extract_generic_symbols(content, rel_path)

        # LLM summary (optional, slow)
        if generate_summaries and self.client and len(content) < MAX_SUMMARY_FILE_SIZE:
            entry.summary = self._generate_summary(rel_path, content)

        return entry

    @staticmethod
    def _hash_file(file_path: Path) -> str:
        """Compute content hash for change detection."""
        try:
            content = file_path.read_bytes()
            return hashlib.md5(content, usedforsecurity=False).hexdigest()
        except OSError as e:
            log.debug("Cannot hash %s: %s", file_path, e)
            return ""

    # Class-level constant: extension → language mapping (avoid recreating per call)
    _LANG_MAP = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".jsx": "javascript", ".tsx": "typescript",
        ".go": "go", ".rs": "rust", ".java": "java",
        ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp",
        ".cs": "csharp", ".rb": "ruby", ".php": "php",
        ".swift": "swift", ".kt": "kotlin", ".scala": "scala",
        ".lua": "lua", ".sh": "shell", ".bash": "shell", ".zsh": "shell",
        ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
        ".json": "json", ".md": "markdown",
        ".html": "html", ".css": "css", ".scss": "scss",
        ".sql": "sql", ".r": "r", ".R": "r", ".jl": "julia",
    }

    def _detect_language(self, ext: str) -> str:
        """Map file extension to language name."""
        return self._LANG_MAP.get(ext, "unknown")

    # --- Symbol extraction ---

    def _extract_python_symbols(self, content: str, file_path: str) -> list[Symbol]:
        """Extract Python functions, classes, and methods."""
        symbols = []
        lines = content.splitlines()
        current_class = ""

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Class definitions
            match = _PY_CLASS_RE.match(line)
            if match:
                name = match.group(1)
                bases = match.group(2) or ""
                sig = f"class {name}({bases})" if bases else f"class {name}"
                docstring = self._get_python_docstring(lines, i)
                symbols.append(Symbol(
                    name=name, kind="class", file=file_path, line=i,
                    signature=sig, docstring=docstring,
                ))
                current_class = name
                continue

            # Function/method definitions
            match = _PY_DEF_RE.match(line)
            if match:
                indent = match.group(1)
                name = match.group(2)
                params = match.group(3).strip()
                return_type = (match.group(4) or "").strip().rstrip(":")
                # Detect methods: indented (spaces or tabs) inside a class
                is_indented = bool(indent) and indent[0] in (" ", "\t")
                kind = "method" if is_indented and current_class else "function"
                parent = current_class if kind == "method" else ""

                sig = f"def {name}({params})"
                if return_type:
                    sig += f" -> {return_type}"
                docstring = self._get_python_docstring(lines, i)

                symbols.append(Symbol(
                    name=name, kind=kind, file=file_path, line=i,
                    signature=sig, docstring=docstring, parent=parent,
                ))
                continue

            # Top-level if unindented, reset class context
            if stripped and not line[0:1].isspace() and not stripped.startswith(("class ", "def ", "@", "#", "\"", "'")):
                current_class = ""

        return symbols

    @staticmethod
    def _get_python_docstring(lines: list[str], def_line: int) -> str:
        """Extract the docstring following a def/class line."""
        if def_line >= len(lines):
            return ""
        next_line = lines[def_line].strip() if def_line < len(lines) else ""
        if next_line.startswith(('"""', "'''")):
            quote = next_line[:3]
            # Single-line docstring
            if next_line.count(quote) >= 2:
                return next_line.strip(quote).strip()
            # Multi-line: collect until closing quote
            parts = [next_line[3:]]
            for j in range(def_line + 1, min(def_line + 20, len(lines))):
                line = lines[j].strip()
                if quote in line:
                    parts.append(line.replace(quote, "").strip())
                    break
                parts.append(line)
            return " ".join(parts).strip()
        return ""

    @staticmethod
    def _extract_python_imports(content: str) -> list[str]:
        """Extract import statements from Python code."""
        imports = []
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("import "):
                parts = stripped.split()
                if len(parts) >= 2:
                    module = parts[1].split(".")[0]
                    if module not in imports:
                        imports.append(module)
            elif stripped.startswith("from "):
                parts = stripped.split()
                if len(parts) >= 2:
                    module = parts[1].split(".")[0]
                    if module not in imports:
                        imports.append(module)
        return imports

    @staticmethod
    def _extract_js_symbols(content: str, file_path: str) -> list[Symbol]:
        """Extract JavaScript/TypeScript symbols."""
        symbols = []
        lines = content.splitlines()

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Function declarations
            match = _JS_FUNC_RE.match(stripped)
            if match:
                symbols.append(Symbol(
                    name=match.group(1), kind="function", file=file_path, line=i,
                    signature=f"function {match.group(1)}({match.group(2)})",
                ))
                continue

            # Arrow functions / const declarations
            match = _JS_ARROW_RE.match(stripped)
            if match:
                symbols.append(Symbol(
                    name=match.group(1), kind="function", file=file_path, line=i,
                    signature=f"const {match.group(1)} = (...) =>",
                ))
                continue

            # Class declarations
            match = _JS_CLASS_RE.match(stripped)
            if match:
                sig = f"class {match.group(1)}"
                if match.group(2):
                    sig += f" extends {match.group(2)}"
                symbols.append(Symbol(
                    name=match.group(1), kind="class", file=file_path, line=i,
                    signature=sig,
                ))
                continue

            # Interface/type declarations (TypeScript)
            match = _TS_IFACE_RE.match(stripped)
            if match:
                symbols.append(Symbol(
                    name=match.group(1), kind="class", file=file_path, line=i,
                    signature=f"interface {match.group(1)}",
                ))

        return symbols

    @staticmethod
    def _extract_go_symbols(content: str, file_path: str) -> list[Symbol]:
        """Extract Go symbols."""
        symbols = []
        lines = content.splitlines()

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Function declarations
            match = _GO_FUNC_RE.match(stripped)
            if match:
                symbols.append(Symbol(
                    name=match.group(1), kind="function", file=file_path, line=i,
                    signature=stripped.split("{")[0].strip(),
                ))

            # Type declarations
            match = _GO_TYPE_RE.match(stripped)
            if match:
                symbols.append(Symbol(
                    name=match.group(1), kind="class", file=file_path, line=i,
                    signature=f"type {match.group(1)} {match.group(2)}",
                ))

        return symbols

    @staticmethod
    def _extract_rust_symbols(content: str, file_path: str) -> list[Symbol]:
        """Extract Rust symbols."""
        symbols = []
        lines = content.splitlines()

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Function declarations
            match = _RUST_FN_RE.match(stripped)
            if match:
                symbols.append(Symbol(
                    name=match.group(1), kind="function", file=file_path, line=i,
                    signature=stripped.split("{")[0].strip(),
                ))

            # Struct/enum/trait
            match = _RUST_TYPE_RE.match(stripped)
            if match:
                symbols.append(Symbol(
                    name=match.group(1), kind="class", file=file_path, line=i,
                    signature=stripped.split("{")[0].strip(),
                ))

        return symbols

    @staticmethod
    def _extract_generic_symbols(content: str, file_path: str) -> list[Symbol]:
        """Extract symbols from unknown languages using generic patterns."""
        symbols = []
        lines = content.splitlines()

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Match common function patterns
            match = _GENERIC_FUNC_RE.match(stripped)
            if match:
                symbols.append(Symbol(
                    name=match.group(1), kind="function", file=file_path, line=i,
                    signature=stripped[:MAX_SIGNATURE_LENGTH],
                ))
            # Match class-like patterns
            match = _GENERIC_CLASS_RE.match(stripped)
            if match:
                symbols.append(Symbol(
                    name=match.group(1), kind="class", file=file_path, line=i,
                    signature=stripped[:MAX_SIGNATURE_LENGTH],
                ))

        return symbols

    # --- Content search ---

    def _search_content(self, query: str, max_results: int) -> list[SearchResult]:
        """Grep-like content search across indexed files."""
        results = []
        query_lower = query.lower()

        for path, _entry in self._file_index.items():
            file_path = self.project_dir / path
            if not file_path.exists():
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()
                for i, line in enumerate(lines, 1):
                    if query_lower in line.lower():
                        # Get context (2 lines before and after)
                        ctx_start = max(0, i - 3)
                        ctx_end = min(len(lines), i + 2)
                        context_lines = lines[ctx_start:ctx_end]

                        results.append(SearchResult(
                            file=path,
                            line=i,
                            content=line.strip(),
                            score=1.0,
                            context="\n".join(context_lines),
                        ))
                        if len(results) >= max_results:
                            return results
            except (OSError, UnicodeDecodeError) as e:
                log.debug("Cannot search %s: %s", path, e)
                continue

        return results

    # --- LLM summaries ---

    def _generate_summary(self, path: str, content: str) -> str:
        """Generate a one-line LLM summary of a file."""
        if not self.client:
            return ""

        # Truncate for summary prompt
        truncated = content[:MAX_SUMMARY_CONTENT]
        prompt = (
            f"Write a one-line summary (max 80 chars) of what this file does.\n"
            f"File: {path}\n\n```\n{truncated}\n```"
        )

        try:
            result = self.client.generate(
                prompt=prompt,
                system="You are a code summarizer. Respond with ONLY a single line summary, no quotes or punctuation.",
                timeout=30,
                temperature=0.1,
            )
            return result.get("response", "").strip()[:MAX_SUMMARY_LENGTH]
        except (OSError, TimeoutError, ValueError) as e:
            log.debug("Cannot generate summary for %s: %s", path, e)
            return ""

    # --- Persistence ---

    def _save_index(self) -> None:
        """Save index to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Save file index
        data = {}
        for path, entry in self._file_index.items():
            data[path] = {
                "path": entry.path,
                "language": entry.language,
                "size": entry.size,
                "lines": entry.lines,
                "hash": entry.hash,
                "summary": entry.summary,
                "indexed_at": entry.indexed_at,
                "imports": entry.imports,
                "symbols": [
                    {
                        "name": s.name, "kind": s.kind, "file": s.file,
                        "line": s.line, "signature": s.signature,
                        "docstring": s.docstring, "parent": s.parent,
                    }
                    for s in entry.symbols
                ],
            }

        index_file = self.index_dir / "files.json"
        # Atomic write: write to temp file then rename to prevent corruption
        tmp_file = index_file.with_suffix(".json.tmp")
        try:
            tmp_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(str(tmp_file), str(index_file))
        except OSError:
            # Clean up temp file on failure
            tmp_file.unlink(missing_ok=True)
            raise
        log.debug("Saved index: %d files to %s", len(data), index_file)

    def _load_index(self) -> None:
        """Load index from disk."""
        index_file = self.index_dir / "files.json"
        if not index_file.exists():
            self._loaded = True
            return

        try:
            data = json.loads(index_file.read_text(encoding="utf-8"))
            self._file_index.clear()
            self._symbol_index.clear()

            for path, entry_data in data.items():
                symbols = []
                for s in entry_data.get("symbols", []):
                    try:
                        symbols.append(Symbol(**s))
                    except (TypeError, KeyError) as e:
                        log.debug("Skipping malformed symbol in index for %s: %s", path, e)
                entry = FileIndex(
                    path=entry_data["path"],
                    language=entry_data["language"],
                    size=entry_data["size"],
                    lines=entry_data["lines"],
                    hash=entry_data["hash"],
                    summary=entry_data.get("summary", ""),
                    indexed_at=entry_data.get("indexed_at", 0),
                    imports=entry_data.get("imports", []),
                    symbols=symbols,
                )
                self._file_index[path] = entry
                for sym in symbols:
                    self._symbol_index.setdefault(sym.name.lower(), []).append(sym)

            self._loaded = True
            log.debug("Loaded index: %d files from %s", len(self._file_index), index_file)
        except (json.JSONDecodeError, OSError, KeyError, TypeError) as e:
            log.warning("Could not load index: %s", e)
            self._loaded = True

    def _remove_from_index(self, path: str) -> None:
        """Remove a file and its symbols from the index."""
        entry = self._file_index.pop(path, None)
        if entry:
            for sym in entry.symbols:
                key = sym.name.lower()
                if key in self._symbol_index:
                    self._symbol_index[key] = [
                        s for s in self._symbol_index[key] if s.file != path
                    ]
                    if not self._symbol_index[key]:
                        del self._symbol_index[key]


class CodebaseTool:
    """Agent-callable tool wrapper around CodebaseIndexer.

    Exposes codebase search, symbol lookup, and project overview as tool
    functions that agents can invoke through Ollama tool calling.

    Includes staleness detection — if files have changed since the index
    was built, the stale entries are automatically refreshed.
    """

    name = "codebase"
    description = "Search code, find symbols, and explore project structure"

    # Re-check for staleness every 5 minutes
    STALENESS_CHECK_INTERVAL = 300

    def __init__(self, working_dir: str = ".", client: OllamaClient | None = None):
        self._indexer = CodebaseIndexer(project_dir=working_dir, client=client)
        self._indexed = False
        self._last_staleness_check: float = 0

    def _ensure_indexed(self) -> None:
        """Build or load the index on first use, and refresh stale entries."""
        if not self._indexed:
            self._indexer._load_index()
            if not self._indexer._file_index:
                self._indexer.build_index()
            self._indexed = True
            self._last_staleness_check = time.time()
        elif time.time() - self._last_staleness_check > self.STALENESS_CHECK_INTERVAL:
            self._refresh_stale_entries()
            self._last_staleness_check = time.time()

    def _refresh_stale_entries(self) -> None:
        """Check for files that changed since last index and re-index them."""
        stale_count = 0
        for path, entry in list(self._indexer._file_index.items()):
            full_path = self._indexer.project_dir / path
            if not full_path.exists():
                self._indexer._remove_from_index(path)
                stale_count += 1
                continue
            try:
                current_hash = hashlib.md5(full_path.read_bytes(), usedforsecurity=False).hexdigest()
                if current_hash != entry.hash:
                    self._indexer._remove_from_index(path)
                    new_entry = self._indexer._index_file(full_path)
                    if new_entry:
                        self._indexer._file_index[new_entry.path] = new_entry
                        for sym in new_entry.symbols:
                            self._indexer._symbol_index.setdefault(
                                sym.name.lower(), []
                            ).append(sym)
                    stale_count += 1
            except (OSError, PermissionError):
                continue
        if stale_count:
            self._indexer._save_index()
            log.info("Refreshed %d stale index entries", stale_count)

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return Ollama tool-calling definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "codebase_search",
                    "description": "Search the codebase for a query (files, symbols, code content)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query (e.g., 'authentication', 'UserModel', 'handle_request')"},
                            "max_results": {"type": "integer", "description": "Maximum results to return (default 10)"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "find_symbol",
                    "description": "Find a specific symbol (function, class, method) by name",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Symbol name to find"},
                        },
                        "required": ["name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "project_overview",
                    "description": "Get a high-level overview of the project structure, files, and languages",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_summary",
                    "description": "Get a summary of a specific file (symbols, imports, structure)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path relative to project root"},
                        },
                        "required": ["path"],
                    },
                },
            },
        ]

    def execute(self, function_name: str, args: dict[str, Any]) -> str:
        """Execute a codebase tool function."""
        self._ensure_indexed()

        handlers = {
            "codebase_search": self._search,
            "find_symbol": self._find_symbol,
            "project_overview": self._overview,
            "file_summary": self._file_summary,
        }
        handler = handlers.get(function_name)
        if not handler:
            return f"Unknown function: {function_name}"
        try:
            return handler(**args)
        except (TypeError, ValueError, OSError, KeyError) as e:
            return f"Error: {e}"

    def _search(self, query: str, max_results: int = 10) -> str:
        """Search the codebase."""
        if not query or not query.strip():
            return "Error: search query cannot be empty"
        max_results = min(max(1, max_results), MAX_SEARCH_RESULTS)
        results = self._indexer.search(query, max_results=max_results)
        if not results:
            return f"No results found for: {query}"

        lines = [f"Found {len(results)} results for '{query}':\n"]
        for r in results:
            lines.append(f"  {r.file}:{r.line}  {r.content}")
            if r.context:
                lines.append(f"    {r.context[:MAX_CONTEXT_PREVIEW]}")
        return "\n".join(lines)

    def _find_symbol(self, name: str) -> str:
        """Find a symbol by name."""
        name = name.strip()
        if not name:
            return "Error: symbol name cannot be empty"
        symbols = self._indexer.find_symbol(name)
        if not symbols:
            return f"No symbol found: {name}"

        lines = [f"Found {len(symbols)} matches for '{name}':\n"]
        for s in symbols:
            lines.append(f"  {s.kind}: {s.signature or s.name}")
            lines.append(f"    {s.file}:{s.line}")
            if s.docstring:
                lines.append(f"    {s.docstring[:MAX_CONTEXT_PREVIEW]}")
        return "\n".join(lines)

    def _overview(self) -> str:
        """Get project overview."""
        return self._indexer.get_project_overview()

    def _file_summary(self, path: str) -> str:
        """Get file summary."""
        path = path.strip()
        if not path:
            return "Error: file path cannot be empty"
        return self._indexer.get_file_summary(path)
