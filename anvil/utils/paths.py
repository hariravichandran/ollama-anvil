"""Path sandbox enforcement — jail a path inside a root directory.

Extracted from the same ``resolve`` + ``relative_to`` + try/except pattern
in :mod:`anvil.tools.scratchpad`, :mod:`anvil.tools.vision_ocr`,
:mod:`anvil.tools.fuzzy_edit`, and :mod:`anvil.tools.image_prep`. All four
wanted: "given a user-supplied path, return its absolute form iff it
stays inside my working directory — otherwise refuse."

Usage::

    from anvil.utils.paths import jail_path, PathEscape

    try:
        abs_path = jail_path(user_input, root=self.working_dir)
    except PathEscape as e:
        return f"error: {e}"

Relative paths are resolved against ``root``; absolute paths are checked
as-is (and rejected if they escape). Symlink resolution is applied to both
sides so a symlink inside the root that points *out* of the root is
rejected.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["PathEscape", "jail_path", "is_inside"]


class PathEscape(ValueError):
    """Raised when a path resolves outside the allowed root."""


def jail_path(path: str | Path, root: str | Path) -> Path:
    """Resolve ``path`` against ``root`` and ensure it stays inside.

    * Relative paths: resolved as ``root / path``.
    * Absolute paths: resolved as-is.

    Raises :class:`PathEscape` if the final resolved path is not a
    descendant of (or equal to) the resolved ``root``.
    """
    root_abs = Path(root).resolve()
    p = Path(path)
    candidate = (root_abs / p).resolve() if not p.is_absolute() else p.resolve()
    if not is_inside(candidate, root_abs):
        raise PathEscape(f"path escapes root: {path!r} (root={root_abs})")
    return candidate


def is_inside(path: Path, root: Path) -> bool:
    """True when ``path`` is ``root`` or a descendant."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True
