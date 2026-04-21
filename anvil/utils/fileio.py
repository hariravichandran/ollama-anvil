"""File I/O utilities: atomic writes and safe file operations."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from anvil.utils.logging import get_logger

log = get_logger("utils.fileio")

__all__ = ["atomic_write"]


def atomic_write(path: str | Path, content: str, *, encoding: str = "utf-8") -> None:
    """Write content to a file atomically via temp file + os.replace.

    Creates a temporary file in the same directory as the target, writes the
    content, then atomically replaces the target. This prevents partial writes
    and data corruption on crash.

    Args:
        path: Destination file path.
        content: Text content to write.
        encoding: Text encoding (default: utf-8).

    Raises:
        OSError: If the write or replace fails (temp file is cleaned up).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: str | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp",
        )
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
        os.replace(tmp_path, str(path))
    except OSError:
        if tmp_path and Path(tmp_path).exists():
            Path(tmp_path).unlink()
        raise
