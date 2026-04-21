"""Prompt caching utilities — optimize repeated inference patterns.

Ollama automatically caches the KV state for prompt prefixes that match
previous requests. This module helps users maximize cache hits by:

1. Tracking system prompt changes (same prompt = KV cache reuse)
2. Estimating cache savings
3. Providing cache-aware prompt construction

Usage::

    from anvil.llm.cache import PromptCacheTracker

    tracker = PromptCacheTracker()
    tracker.set_system_prompt("You are a helpful assistant.")
    savings = tracker.estimate_savings(message_count=10)
"""

from __future__ import annotations

import hashlib

from anvil.utils.logging import get_logger

log = get_logger("llm.cache")

__all__ = [
    "PromptCacheTracker",
    "TOKENS_PER_CHAR",
    "CACHE_HIT_SPEEDUP",
]

# Approximate tokens per character (for English text)
TOKENS_PER_CHAR = 0.25

# Approximate speedup factor from KV cache hit on system prompt
# (no need to reprocess cached prefix)
CACHE_HIT_SPEEDUP = 2.0


class PromptCacheTracker:
    """Track system prompt changes to estimate KV cache effectiveness.

    Ollama's KV cache works by caching the processed state of prompt
    prefixes. If the system prompt (which comes first) stays the same
    between requests, the KV cache for that prefix is reused, saving
    significant computation.
    """

    def __init__(self) -> None:
        self._current_hash: str = ""
        self._current_prompt: str = ""
        self._prompt_tokens: int = 0
        self._total_requests: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    def __repr__(self) -> str:
        hit_rate = self.hit_rate
        return f"PromptCacheTracker(hits={self._cache_hits}, misses={self._cache_misses}, rate={hit_rate:.0f}%)"

    def set_system_prompt(self, prompt: str) -> bool:
        """Set the current system prompt and track changes.

        Args:
            prompt: The system prompt text.

        Returns:
            True if prompt changed (cache miss), False if same (cache hit).
        """
        new_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

        if new_hash == self._current_hash:
            self._cache_hits += 1
            self._total_requests += 1
            return False  # Cache hit

        self._current_hash = new_hash
        self._current_prompt = prompt
        self._prompt_tokens = int(len(prompt) * TOKENS_PER_CHAR)
        self._cache_misses += 1
        self._total_requests += 1
        log.debug("System prompt changed (hash=%s, ~%d tokens)", new_hash, self._prompt_tokens)
        return True  # Cache miss

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a percentage."""
        if self._total_requests == 0:
            return 0.0
        return (self._cache_hits / self._total_requests) * 100

    @property
    def prompt_tokens(self) -> int:
        """Estimated tokens in the current system prompt."""
        return self._prompt_tokens

    def estimate_savings(self, message_count: int | None = None) -> dict[str, float]:
        """Estimate token savings from prompt caching.

        Args:
            message_count: Override message count. If None, uses tracked count.

        Returns:
            Dict with 'cached_tokens', 'total_tokens', 'savings_pct',
            'estimated_speedup'.
        """
        count = message_count if message_count is not None else self._total_requests
        if count == 0:
            return {
                "cached_tokens": 0,
                "total_tokens": 0,
                "savings_pct": 0.0,
                "estimated_speedup": 1.0,
            }

        total_tokens = self._prompt_tokens * count
        # All but the first request benefit from caching
        cached_tokens = self._prompt_tokens * max(0, count - 1)
        savings_pct = (cached_tokens / max(1, total_tokens)) * 100

        # Speedup from not reprocessing cached prefix
        speedup = 1.0 + (cached_tokens / max(1, total_tokens)) * (CACHE_HIT_SPEEDUP - 1.0)

        return {
            "cached_tokens": cached_tokens,
            "total_tokens": total_tokens,
            "savings_pct": round(savings_pct, 1),
            "estimated_speedup": round(speedup, 2),
        }

    def reset(self) -> None:
        """Reset all tracking statistics."""
        self._current_hash = ""
        self._current_prompt = ""
        self._prompt_tokens = 0
        self._total_requests = 0
        self._cache_hits = 0
        self._cache_misses = 0

    def stats(self) -> dict[str, int | float]:
        """Get cache tracking statistics."""
        return {
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": round(self.hit_rate, 1),
            "prompt_tokens": self._prompt_tokens,
        }
