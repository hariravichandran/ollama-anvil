"""Streaming token counter — real-time token estimation during generation.

Provides fast character-based token estimation for streaming responses
without requiring a tokenizer. Tracks tokens in flight and provides
running statistics during generation.

Usage::

    from anvil.llm.tokencount import StreamingCounter

    counter = StreamingCounter()
    for chunk in stream:
        counter.add(chunk)
        print(f"\\r{counter.tokens_so_far} tokens, {counter.elapsed_tps:.1f} tok/s", end="")
    print(counter.summary())
"""

from __future__ import annotations

import time

from anvil.utils.logging import get_logger

log = get_logger("llm.tokencount")

__all__ = [
    "StreamingCounter",
    "estimate_tokens",
    "CHARS_PER_TOKEN",
    "CODE_CHARS_PER_TOKEN",
]

# Average characters per token (English text)
CHARS_PER_TOKEN = 4.0

# Average characters per token (code, tends to be shorter tokens)
CODE_CHARS_PER_TOKEN = 3.2


def estimate_tokens(text: str, is_code: bool = False) -> int:
    """Estimate token count from text without a tokenizer.

    Args:
        text: Input text.
        is_code: If True, use code-optimized ratio.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    ratio = CODE_CHARS_PER_TOKEN if is_code else CHARS_PER_TOKEN
    return max(1, int(len(text) / ratio))


class StreamingCounter:
    """Count tokens in real-time during streaming generation.

    Uses character-based estimation for fast, dependency-free counting.
    """

    def __init__(self, is_code: bool = False) -> None:
        self._chars = 0
        self._chunks = 0
        self._is_code = is_code
        self._start_time: float = 0.0
        self._first_token_time: float = 0.0
        self._ratio = CODE_CHARS_PER_TOKEN if is_code else CHARS_PER_TOKEN

    def __repr__(self) -> str:
        return f"StreamingCounter(tokens≈{self.tokens_so_far}, chunks={self._chunks})"

    def start(self) -> None:
        """Mark the start of generation (call before streaming begins)."""
        self._start_time = time.monotonic()

    def add(self, chunk: str) -> None:
        """Add a streamed text chunk.

        Args:
            chunk: Text chunk from the stream.
        """
        if not chunk:
            return
        if self._start_time == 0.0:
            self._start_time = time.monotonic()
        if self._first_token_time == 0.0 and self._chars == 0:
            self._first_token_time = time.monotonic()
        self._chars += len(chunk)
        self._chunks += 1

    @property
    def tokens_so_far(self) -> int:
        """Estimated tokens generated so far."""
        return max(0, int(self._chars / self._ratio))

    @property
    def chars_so_far(self) -> int:
        """Total characters received."""
        return self._chars

    @property
    def elapsed_s(self) -> float:
        """Elapsed time since start."""
        if self._start_time == 0.0:
            return 0.0
        return time.monotonic() - self._start_time

    @property
    def time_to_first_token(self) -> float:
        """Time from start() to first chunk."""
        if self._start_time == 0.0 or self._first_token_time == 0.0:
            return 0.0
        return self._first_token_time - self._start_time

    @property
    def elapsed_tps(self) -> float:
        """Current tokens per second."""
        elapsed = self.elapsed_s
        if elapsed <= 0:
            return 0.0
        return self.tokens_so_far / elapsed

    def summary(self) -> str:
        """Generate final summary of streaming stats."""
        return (
            f"~{self.tokens_so_far} tokens in {self.elapsed_s:.1f}s "
            f"({self.elapsed_tps:.1f} tok/s, "
            f"TTFT: {self.time_to_first_token:.2f}s, "
            f"{self._chunks} chunks)"
        )

    def reset(self) -> None:
        """Reset counter for reuse."""
        self._chars = 0
        self._chunks = 0
        self._start_time = 0.0
        self._first_token_time = 0.0
