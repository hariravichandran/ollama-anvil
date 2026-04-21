"""Config-path discovery — project → XDG → ~/.config/ollama-anvil.

Extracted from the repeated three-tier search in
:mod:`anvil.agents.skills`, :mod:`anvil.agents.prompt_library`, and
:mod:`anvil.ui.statusline`. The convention:

1. ``<working_dir>/.anvil/<subdir>``  — project-local, wins by default
2. ``$XDG_CONFIG_HOME/ollama-anvil/<subdir>``  — when that env var is set
3. ``~/.config/ollama-anvil/<subdir>``         — user-wide fallback

Usage::

    from anvil.utils.config_paths import config_roots, user_config_root

    roots = config_roots("skills")           # list[Path], project first
    user = user_config_root("skills")        # Path, for writes
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "PROJECT_CONFIG_DIR",
    "USER_CONFIG_DIR",
    "config_roots",
    "user_config_root",
    "project_config_root",
]

PROJECT_CONFIG_DIR = ".anvil"
USER_CONFIG_DIR = "ollama-anvil"


def config_roots(subdir: str, working_dir: str | Path = ".") -> list[Path]:
    """Return all config roots for ``subdir`` in priority order.

    First entry wins for reads; writes should use :func:`user_config_root`
    (or explicitly choose the project root when the file is project-scoped).
    """
    roots: list[Path] = [project_config_root(subdir, working_dir)]
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        roots.append(Path(xdg) / USER_CONFIG_DIR / subdir)
    roots.append(user_config_root(subdir))
    return roots


def project_config_root(subdir: str, working_dir: str | Path = ".") -> Path:
    """``<working_dir>/.anvil/<subdir>``."""
    return Path(working_dir) / PROJECT_CONFIG_DIR / subdir


def user_config_root(subdir: str) -> Path:
    """``~/.config/ollama-anvil/<subdir>``."""
    return Path.home() / ".config" / USER_CONFIG_DIR / subdir
