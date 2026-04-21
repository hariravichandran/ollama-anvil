"""Structured logging setup for ollama-anvil."""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the anvil root logger with rich output."""
    logger = logging.getLogger("anvil")
    if logger.handlers:
        return logger  # already configured

    handler = RichHandler(
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the anvil namespace."""
    return logging.getLogger(f"anvil.{name}")
