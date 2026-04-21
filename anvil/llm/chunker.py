"""Document chunking for retrieval-augmented generation (RAG).

Splits text into overlapping chunks optimized for embedding and retrieval.
Supports multiple strategies: fixed-size, sentence-based, and paragraph-based.

Usage::

    from anvil.llm.chunker import chunk_text, chunk_file

    # Simple chunking
    chunks = chunk_text("Long document text...", chunk_size=500, overlap=50)

    # Sentence-aware chunking (better for semantic search)
    chunks = chunk_text(text, strategy="sentence", chunk_size=500)

    # Chunk a file directly
    chunks = chunk_file("paper.txt", strategy="paragraph")

    # Feed into EmbeddingStore
    from anvil.llm.embeddings import EmbeddingStore
    store = EmbeddingStore(client, model="nomic-embed-text")
    docs = [{"doc_id": c.chunk_id, "text": c.text, "metadata": c.metadata}
            for c in chunks]
    store.add_batch(docs)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anvil.utils.logging import get_logger

log = get_logger("llm.chunker")

__all__ = [
    "Chunk",
    "chunk_text",
    "chunk_file",
    "STRATEGIES",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_OVERLAP",
    "MAX_CHUNK_SIZE",
    "MIN_CHUNK_SIZE",
    "SENTENCE_ENDINGS",
]

# Chunking parameters
DEFAULT_CHUNK_SIZE = 500   # characters per chunk
DEFAULT_OVERLAP = 50       # overlap between consecutive chunks
MAX_CHUNK_SIZE = 50_000    # safety limit
MIN_CHUNK_SIZE = 10        # minimum meaningful chunk

# Sentence boundary pattern
SENTENCE_ENDINGS = re.compile(r"(?<=[.!?])\s+")

# Available strategies
STRATEGIES = frozenset({"fixed", "sentence", "paragraph", "code"})

# Max bytes of Python source we'll try to parse with ast. Above this we
# fall back to regex-based splitting. Keeps the chunker reasonably fast
# on very large generated files.
MAX_AST_PARSE_BYTES = 1_000_000

# Regex for language-agnostic "top-level definition" starts. Matches the
# common ``def/function/class/interface/struct/fn`` idioms so we can
# split non-Python sources without a real parser. Kept intentionally
# simple — this is a retrieval chunker, not a compiler front-end.
_CODE_BLOCK_START_RE = re.compile(
    r"^(?:"
    r"\s*(?:async\s+)?def\s+\w+"                 # Python def
    r"|\s*class\s+\w+"                            # Python class
    r"|\s*(?:export\s+)?(?:async\s+)?function\s+\w+"  # JS/TS function
    r"|\s*(?:export\s+)?(?:default\s+)?class\s+\w+"   # JS/TS class
    r"|\s*interface\s+\w+"                        # TS interface
    r"|\s*(?:pub\s+)?struct\s+\w+"                # Rust/Go struct
    r"|\s*(?:pub\s+)?(?:async\s+)?fn\s+\w+"       # Rust fn
    r"|\s*func\s+\w+"                             # Go func
    r"|\s*(?:public|private|protected)\s+(?:static\s+)?[\w<>,\s\[\]]+\s+\w+\s*\("  # Java/C# method
    r")",
    re.MULTILINE,
)


@dataclass(slots=True)
class Chunk:
    """A text chunk with position metadata."""

    chunk_id: str
    text: str
    index: int          # position in the sequence (0-based)
    start_char: int     # character offset in original text
    end_char: int       # character offset end in original text
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    strategy: str = "sentence",
    source: str = "",
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Split text into overlapping chunks.

    Args:
        text: The text to chunk.
        chunk_size: Target size per chunk in characters.
        overlap: Number of overlapping characters between chunks.
        strategy: Chunking strategy — 'fixed', 'sentence', or 'paragraph'.
        source: Optional source identifier (included in chunk metadata).
        metadata: Additional metadata to attach to each chunk.

    Returns list of Chunk objects.
    """
    if not text or not text.strip():
        return []

    # Validate parameters
    chunk_size = max(MIN_CHUNK_SIZE, min(chunk_size, MAX_CHUNK_SIZE))
    overlap = max(0, min(overlap, chunk_size // 2))

    if strategy not in STRATEGIES:
        log.warning("Unknown strategy '%s', falling back to 'sentence'", strategy)
        strategy = "sentence"

    if strategy == "fixed":
        raw_chunks = _chunk_fixed(text, chunk_size, overlap)
    elif strategy == "sentence":
        raw_chunks = _chunk_sentence(text, chunk_size, overlap)
    elif strategy == "paragraph":
        raw_chunks = _chunk_paragraph(text, chunk_size, overlap)
    elif strategy == "code":
        # Language auto-detection from the source filename when available.
        raw_chunks = _chunk_code(text, chunk_size, overlap, source=source)
    else:
        raw_chunks = _chunk_fixed(text, chunk_size, overlap)

    # Build Chunk objects
    base_meta = metadata.copy() if metadata else {}
    if source:
        base_meta["source"] = source

    chunks = []
    for i, (chunk_text_val, start, end) in enumerate(raw_chunks):
        chunk_meta = {**base_meta, "chunk_index": i, "total_chunks": len(raw_chunks)}
        chunk_id = f"{source or 'chunk'}_{i}" if source else f"chunk_{i}"
        chunks.append(Chunk(
            chunk_id=chunk_id,
            text=chunk_text_val,
            index=i,
            start_char=start,
            end_char=end,
            metadata=chunk_meta,
        ))

    log.debug(
        "Chunked %d chars into %d chunks (strategy=%s, size=%d, overlap=%d)",
        len(text), len(chunks), strategy, chunk_size, overlap,
    )
    return chunks


def chunk_file(
    path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    strategy: str = "sentence",
    encoding: str = "utf-8",
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Read a file and chunk its contents.

    Args:
        path: Path to the text file.
        chunk_size: Target size per chunk in characters.
        overlap: Overlap between consecutive chunks.
        strategy: Chunking strategy.
        encoding: File encoding.
        metadata: Additional metadata for each chunk.

    Returns list of Chunk objects with source set to the filename.
    """
    path = Path(path)
    if not path.exists():
        log.error("File not found: %s", path)
        return []

    text = path.read_text(encoding=encoding)
    source = path.name
    file_meta = {"file_path": str(path), "file_size": len(text)}
    if metadata:
        file_meta.update(metadata)

    return chunk_text(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
        strategy=strategy,
        source=source,
        metadata=file_meta,
    )


# ─── Internal chunking strategies ───────────────────────────────────────────


def _chunk_fixed(
    text: str, chunk_size: int, overlap: int,
) -> list[tuple[str, int, int]]:
    """Fixed-size chunking with overlap. Fast but may split mid-word."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((chunk, start, end))
        start += chunk_size - overlap
    return chunks


def _chunk_sentence(
    text: str, chunk_size: int, overlap: int,
) -> list[tuple[str, int, int]]:
    """Sentence-aware chunking. Splits on sentence boundaries.

    Accumulates sentences until chunk_size is reached, then starts a new
    chunk with overlap from the previous chunk's ending sentences.
    """
    # Split into sentences
    sentences = SENTENCE_ENDINGS.split(text)
    if not sentences:
        return [(text.strip(), 0, len(text))] if text.strip() else []

    chunks = []
    current_sentences: list[str] = []
    current_len = 0
    current_start = 0
    _text_pos = 0  # track position in original text

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_len = len(sentence)

        if current_len + sentence_len > chunk_size and current_sentences:
            # Emit current chunk
            chunk_text_val = " ".join(current_sentences)
            chunk_end = current_start + len(chunk_text_val)
            chunks.append((chunk_text_val, current_start, chunk_end))

            # Calculate overlap: keep last sentences that fit in overlap
            overlap_sentences: list[str] = []
            overlap_len = 0
            for s in reversed(current_sentences):
                if overlap_len + len(s) > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s) + 1  # +1 for space

            current_sentences = overlap_sentences
            current_len = sum(len(s) for s in current_sentences) + max(0, len(current_sentences) - 1)
            current_start = chunk_end - current_len

        current_sentences.append(sentence)
        current_len += sentence_len + (1 if current_len > 0 else 0)

    # Emit remaining
    if current_sentences:
        chunk_text_val = " ".join(current_sentences)
        chunks.append((chunk_text_val, current_start, current_start + len(chunk_text_val)))

    return chunks


def _chunk_paragraph(
    text: str, chunk_size: int, overlap: int,
) -> list[tuple[str, int, int]]:
    """Paragraph-aware chunking. Splits on double newlines.

    Keeps paragraphs intact where possible. Long paragraphs are split
    using sentence chunking as a fallback.
    """
    # Split on paragraph boundaries (2+ newlines)
    paragraphs = re.split(r"\n\s*\n", text)

    chunks = []
    current_paras: list[str] = []
    current_len = 0
    current_start = 0
    text_pos = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_len = len(para)

        # If single paragraph exceeds chunk_size, split it with sentences
        if para_len > chunk_size and not current_paras:
            sub_chunks = _chunk_sentence(para, chunk_size, overlap)
            for sub_text, sub_start, sub_end in sub_chunks:
                chunks.append((sub_text, text_pos + sub_start, text_pos + sub_end))
            text_pos += para_len + 2  # +2 for paragraph separator
            continue

        if current_len + para_len > chunk_size and current_paras:
            # Emit current chunk
            chunk_text_val = "\n\n".join(current_paras)
            chunks.append((chunk_text_val, current_start, current_start + len(chunk_text_val)))

            # Overlap: keep last paragraph if it fits
            if overlap > 0 and current_paras:
                last = current_paras[-1]
                if len(last) <= overlap:
                    current_paras = [last]
                    current_len = len(last)
                    current_start = current_start + len(chunk_text_val) - len(last)
                else:
                    current_paras = []
                    current_len = 0
                    current_start = text_pos
            else:
                current_paras = []
                current_len = 0
                current_start = text_pos

        current_paras.append(para)
        current_len += para_len + (2 if current_len > 0 else 0)
        text_pos += para_len + 2

    # Emit remaining
    if current_paras:
        chunk_text_val = "\n\n".join(current_paras)
        chunks.append((chunk_text_val, current_start, current_start + len(chunk_text_val)))

    return chunks


def _chunk_code(
    text: str,
    chunk_size: int,
    overlap: int,
    source: str = "",
) -> list[tuple[str, int, int]]:
    """Code-aware chunking: split by top-level definitions.

    Python sources (detected via ``.py`` filename or heuristic) use the
    stdlib :mod:`ast` to find exact class/function boundaries. Other
    languages fall back to a regex that matches the common
    ``def / class / function / fn / func / struct / interface`` idioms.

    Boundaries that produce oversized chunks (single classes > chunk_size)
    are re-split using the fixed-size chunker so we still obey the cap.
    Small sibling boundaries are coalesced into a single chunk up to
    ``chunk_size`` so we don't emit a flurry of one-line stubs.
    """
    if not text.strip():
        return []

    use_ast = _looks_like_python(source, text)
    if use_ast and len(text.encode("utf-8")) <= MAX_AST_PARSE_BYTES:
        boundaries = _python_ast_boundaries(text)
        if boundaries:
            return _coalesce_by_size(text, boundaries, chunk_size, overlap)

    # Regex fallback for non-Python or when AST parse failed.
    boundaries = _regex_code_boundaries(text)
    if not boundaries:
        return _chunk_fixed(text, chunk_size, overlap)
    return _coalesce_by_size(text, boundaries, chunk_size, overlap)


# ─── Code-chunking helpers ──────────────────────────────────────────


def _looks_like_python(source: str, text: str) -> bool:
    """Heuristic: is ``text`` Python source?"""
    if source.endswith(".py"):
        return True
    # Short-circuit check — presence of ``def ``/``class `` at line start.
    head = text[:2000]
    return bool(re.search(r"(?m)^\s*(def|class|async def)\s+\w+", head))


def _python_ast_boundaries(text: str) -> list[tuple[int, int]]:
    """Return (start_char, end_char) spans for every top-level Python def/class.

    Falls through to ``[]`` on SyntaxError so the caller can degrade to
    the regex path.
    """
    import ast

    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []

    # Convert line/col to char offsets. ast line numbers are 1-based.
    lines = text.splitlines(keepends=True)
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line))

    def _offset(line: int, col: int) -> int:
        if line <= 0 or line > len(line_offsets):
            return 0
        return line_offsets[line - 1] + col

    boundaries: list[tuple[int, int]] = []
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start = _offset(node.lineno, node.col_offset)
        end_line = getattr(node, "end_lineno", node.lineno)
        end_col = getattr(node, "end_col_offset", 0)
        end = _offset(end_line, end_col)
        if end <= start:
            continue
        boundaries.append((start, end))

    return boundaries


def _regex_code_boundaries(text: str) -> list[tuple[int, int]]:
    """Regex-based fallback: find definition starts, span to the next start."""
    matches = list(_CODE_BLOCK_START_RE.finditer(text))
    if not matches:
        return []
    boundaries: list[tuple[int, int]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        if end > start:
            boundaries.append((start, end))
    return boundaries


def _coalesce_by_size(
    text: str,
    boundaries: list[tuple[int, int]],
    chunk_size: int,
    overlap: int,
) -> list[tuple[str, int, int]]:
    """Merge small boundaries into chunk-sized groups; split oversized ones.

    Preserves character offsets so downstream code can still cite
    locations in the original text.
    """
    if not boundaries:
        return []
    chunks: list[tuple[str, int, int]] = []
    cur_start: int | None = None
    cur_end: int | None = None

    def _flush() -> None:
        nonlocal cur_start, cur_end
        if cur_start is None or cur_end is None:
            return
        span_text = text[cur_start:cur_end]
        chunks.append((span_text, cur_start, cur_end))
        cur_start = None
        cur_end = None

    for start, end in boundaries:
        size = end - start
        # Oversized single boundary: split with the fixed chunker but
        # keep the char offsets relative to the outer text.
        if size > chunk_size:
            _flush()
            sub = _chunk_fixed(text[start:end], chunk_size, overlap)
            for sub_text, sub_start, sub_end in sub:
                chunks.append((sub_text, start + sub_start, start + sub_end))
            continue

        if cur_start is None:
            cur_start, cur_end = start, end
            continue

        # Coalesce if the combined span fits; otherwise flush + start anew.
        if (end - cur_start) <= chunk_size:
            cur_end = end
        else:
            _flush()
            cur_start, cur_end = start, end

    _flush()
    return chunks
