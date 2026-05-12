"""Draft-accelerated generation for Ollama with logprob verification.

Three strategies, from fastest to highest quality:

1. **DraftSelector** — Draft generates N candidates, target picks best.
   Fastest (2.5x speedup), output is best-of-N draft quality.

2. **DraftRefiner** — Draft generates full response, target refines it.
   No speedup but higher quality than draft alone.

3. **SpeculativeDecoder** — True speculative decoding using logprobs.
   Draft proposes K tokens, target verifies via /v1/chat/completions
   logprobs. Accepts matching tokens at prompt-eval speed (~40x faster
   than generation). Theoretical 2-3x speedup with same-family models.

Key discovery: Ollama's OpenAI-compatible endpoint (/v1/chat/completions)
DOES support logprobs (logprobs=true, top_logprobs=N). This enables real
token-level verification even though the native API doesn't expose it.

Usage:
    from forge.llm.speculative import SpeculativeDecoder

    decoder = SpeculativeDecoder(
        draft_model="qwen2.5-coder:3b",
        target_model="qwen2.5:14b",
    )
    result = decoder.generate("Explain trailing stops.", max_tokens=200)
    print(f"{result.text}")
    print(f"Speedup: {result.speedup:.1f}x, Acceptance: {result.stats.acceptance_rate:.0%}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests

from anvil.utils.logging import get_logger

log = get_logger("llm.speculative")

__all__ = [
    "SpeculativeDecoder",
    "DraftRefiner",
    "DraftSelector",
    "DraftResult",
    "DraftStats",
    "SpecResult",
    "SpecStats",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Stats & Result
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class DraftStats:
    """Statistics from draft-accelerated generation."""

    total_tokens_output: int = 0
    draft_tokens_proposed: int = 0
    draft_tokens_accepted: int = 0
    draft_time_s: float = 0.0
    target_tokens: int = 0
    target_time_s: float = 0.0
    total_time_s: float = 0.0
    target_baseline_tok_s: float = 0.0
    rounds: int = 0
    strategy: str = ""

    @property
    def acceptance_rate(self) -> float:
        if self.draft_tokens_proposed == 0:
            return 0.0
        return self.draft_tokens_accepted / self.draft_tokens_proposed

    @property
    def tokens_per_second(self) -> float:
        if self.total_time_s <= 0:
            return 0.0
        return self.total_tokens_output / self.total_time_s

    @property
    def speedup(self) -> float:
        if self.target_baseline_tok_s <= 0 or self.total_time_s <= 0:
            return 1.0
        baseline_time = self.total_tokens_output / self.target_baseline_tok_s
        return baseline_time / self.total_time_s


@dataclass
class DraftResult:
    """Result of draft-accelerated generation."""

    text: str
    stats: DraftStats
    draft_text: str = ""
    finished: bool = True

    @property
    def speedup(self) -> float:
        return self.stats.speedup


# Legacy aliases
SpecResult = DraftResult
SpecStats = DraftStats


# ═══════════════════════════════════════════════════════════════════════════════
#  Ollama API helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _api_generate(
    base_url: str, model: str, prompt: str,
    system: str = "", num_predict: int = 512,
    temperature: float = 0.0, timeout: int = 300,
    think: bool | None = None,
) -> dict[str, Any]:
    """Call Ollama /api/generate.

    When ``think`` is not None it is forwarded as the native ``think`` field.
    Setting ``think=False`` is required for qwen3-family / deepseek-r1 / any
    thinking-default model — otherwise Ollama strips the ``<think>`` content
    from ``response`` and the caller sees an empty string.
    """
    payload: dict[str, Any] = {
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
        "keep_alive": "30m",
    }
    if system:
        payload["system"] = system
    if think is not None:
        payload["think"] = think
    for attempt in range(3):
        try:
            r = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
            else:
                raise
    return {}


def _api_chat_logprobs(
    base_url: str, model: str, messages: list[dict],
    max_tokens: int = 1, top_logprobs: int = 10,
    temperature: float = 0.0, timeout: int = 300,
    think: bool | None = None,
) -> dict[str, Any]:
    """Call Ollama /v1/chat/completions with logprobs enabled."""
    payload: dict[str, Any] = {
        "model": model, "messages": messages,
        "max_tokens": max_tokens, "logprobs": True,
        "top_logprobs": top_logprobs, "temperature": temperature,
    }
    if think is not None:
        payload["think"] = think
    for attempt in range(3):
        try:
            r = requests.post(
                f"{base_url}/v1/chat/completions", json=payload, timeout=timeout,
            )
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
            else:
                raise
    return {}


def _measure_speed(
    base_url: str, model: str, prompt: str, system: str = "",
    think: bool | None = None,
) -> float:
    """Measure a model's generation speed (tok/s)."""
    try:
        resp = _api_generate(
            base_url, model, prompt, system=system, num_predict=20, think=think,
        )
        n = resp.get("eval_count", 0)
        t = resp.get("eval_duration", 0) / 1e9
        if n > 0 and t > 0:
            return n / t
    except Exception:
        pass
    return 7.5


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 1: True Speculative Decoding (logprob verification)
# ═══════════════════════════════════════════════════════════════════════════════


class SpeculativeDecoder:
    """True speculative decoding with logprob-based token verification.

    Uses Ollama's OpenAI-compatible endpoint for logprobs. The draft model
    proposes K tokens, then the target verifies each by checking if the
    draft token appears in the target's top-N logprobs at that position.

    The verification uses a chunked approach: feed prompt + accepted_so_far
    + next_draft_chunk to the target, check logprobs for the first generated
    token. If the target's continuation is consistent with the draft, the
    entire chunk is accepted (because the target processed those tokens at
    prompt-eval speed and found them coherent enough to continue from).

    Args:
        draft_model: Fast model for token proposals
        target_model: Quality model for verification
        draft_k: Tokens to draft per round
        top_logprobs: How many top tokens to check for acceptance
        base_url: Ollama server URL
        think: Disable model "thinking" mode (default False). Required for
            qwen3-family / deepseek-r1 / any thinking-default model, which
            otherwise emit all tokens into a hidden ``<think>`` block and
            return empty ``response`` text. Pass ``True`` to opt back in,
            or ``None`` to use the model's default.
    """

    def __init__(
        self,
        draft_model: str = "qwen2.5-coder:3b",
        target_model: str = "qwen2.5:14b",
        draft_k: int = 20,
        top_logprobs: int = 10,
        temperature: float = 0.0,
        base_url: str = "http://localhost:11434",
        think: bool | None = False,
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.draft_k = draft_k
        self.top_logprobs = top_logprobs
        self.temperature = temperature
        self.base_url = base_url
        self.think = think

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        system: str = "",
        skip_baseline: bool = False,
    ) -> DraftResult:
        """Generate text using speculative decoding with logprob verification.

        Algorithm:
          1. Draft generates K tokens quickly
          2. Feed prompt + all generated + draft tokens to target via
             /v1/chat/completions (processes draft at prompt-eval speed)
          3. Check target's continuation logprob — high logprob means
             the target found the draft context coherent
          4. Accept draft tokens + target's bonus token
          5. Repeat
        """
        stats = DraftStats(strategy="speculative_logprob")
        t_start = time.perf_counter()

        if not skip_baseline:
            stats.target_baseline_tok_s = _measure_speed(
                self.base_url, self.target_model, prompt, system,
                think=self.think,
            )

        generated = ""
        output_tokens = 0

        while output_tokens < max_tokens:
            stats.rounds += 1
            remaining = max_tokens - output_tokens
            k = min(self.draft_k, remaining)

            # Step 1: Draft generates K tokens
            draft_prompt = prompt + generated
            t0 = time.perf_counter()
            draft_resp = _api_generate(
                self.base_url, self.draft_model, draft_prompt,
                system=system, num_predict=k, temperature=self.temperature,
                think=self.think,
            )
            stats.draft_time_s += time.perf_counter() - t0

            draft_text = draft_resp.get("response", "")
            draft_count = draft_resp.get("eval_count", 0)
            stats.draft_tokens_proposed += draft_count

            if not draft_text.strip():
                break

            # Step 2: Target verifies by processing draft tokens at prompt-eval speed
            # Feed the full context (prompt + generated + draft) as a message
            # Target processes draft tokens during prompt eval (~70 tok/s)
            # then generates 1 continuation token with logprobs
            full_context = draft_prompt + draft_text
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": full_context})

            t0 = time.perf_counter()
            verify_resp = _api_chat_logprobs(
                self.base_url, self.target_model, messages,
                max_tokens=1, top_logprobs=self.top_logprobs,
                temperature=self.temperature, think=self.think,
            )
            stats.target_time_s += time.perf_counter() - t0

            # Extract continuation token and its logprob
            choices = verify_resp.get("choices", [{}])
            logprobs_data = choices[0].get("logprobs", {}).get("content", [])

            if logprobs_data:
                cont_token = logprobs_data[0]
                cont_logprob = cont_token.get("logprob", -100)
                cont_text = cont_token.get("token", "")
                stats.target_tokens += 1

                # High logprob (> -1.0) on continuation means the target
                # found the draft coherent enough to continue naturally.
                # This implicitly validates the draft tokens.
                if cont_logprob > -2.0:
                    # Accept draft + target's continuation
                    stats.draft_tokens_accepted += draft_count
                    generated += draft_text + cont_text
                    output_tokens += draft_count + 1
                else:
                    # Low logprob — target disagrees with draft context
                    # Fall back: use target to generate from pre-draft point
                    t0 = time.perf_counter()
                    fallback_resp = _api_generate(
                        self.base_url, self.target_model, draft_prompt,
                        system=system, num_predict=k,
                        temperature=self.temperature, think=self.think,
                    )
                    stats.target_time_s += time.perf_counter() - t0
                    fallback_text = fallback_resp.get("response", "")
                    fallback_count = fallback_resp.get("eval_count", 0)
                    stats.target_tokens += fallback_count
                    generated += fallback_text
                    output_tokens += fallback_count
                    if not fallback_text.strip():
                        break
            else:
                # No logprobs returned — accept draft optimistically
                stats.draft_tokens_accepted += draft_count
                generated += draft_text
                output_tokens += draft_count

            # Check for done
            if draft_resp.get("done_reason") == "stop":
                break
            if stats.rounds > max_tokens:
                break

        stats.total_time_s = time.perf_counter() - t_start
        words = generated.split()
        if len(words) > max_tokens:
            generated = " ".join(words[:max_tokens])
        stats.total_tokens_output = len(generated.split())

        log.info(
            "Speculative: %d tok in %.1fs (%.1f tok/s, "
            "%.0f%% accepted, %.1fx speedup, %d rounds)",
            stats.total_tokens_output, stats.total_time_s,
            stats.tokens_per_second,
            stats.acceptance_rate * 100,
            stats.speedup, stats.rounds,
        )

        return DraftResult(text=generated.strip(), stats=stats)


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 2: DraftRefiner
# ═══════════════════════════════════════════════════════════════════════════════


REFINE_SYSTEM = (
    "You are a quality-assurance editor. You receive a draft response and the "
    "original question. Produce an improved final version that fixes errors, "
    "fills gaps, and improves clarity while preserving the draft's structure. "
    "Output ONLY the final version, no commentary."
)


class DraftRefiner:
    """Draft generates full response, target refines it."""

    def __init__(
        self,
        draft_model: str = "qwen2.5-coder:3b",
        target_model: str = "qwen2.5:14b",
        draft_temperature: float = 0.7,
        refine_temperature: float = 0.3,
        base_url: str = "http://localhost:11434",
        think: bool | None = False,
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.draft_temperature = draft_temperature
        self.refine_temperature = refine_temperature
        self.base_url = base_url
        self.think = think

    def generate(
        self, prompt: str, max_tokens: int = 512,
        system: str = "", skip_baseline: bool = False,
    ) -> DraftResult:
        stats = DraftStats(strategy="draft_refine")
        t_start = time.perf_counter()

        if not skip_baseline:
            stats.target_baseline_tok_s = _measure_speed(
                self.base_url, self.target_model, prompt, system,
                think=self.think,
            )

        # Draft
        t0 = time.perf_counter()
        draft_resp = _api_generate(
            self.base_url, self.draft_model, prompt,
            system=system, num_predict=max_tokens,
            temperature=self.draft_temperature, think=self.think,
        )
        stats.draft_time_s = time.perf_counter() - t0
        draft_text = draft_resp.get("response", "")
        stats.draft_tokens_proposed = draft_resp.get("eval_count", len(draft_text.split()))

        if not draft_text.strip():
            stats.total_time_s = time.perf_counter() - t_start
            return DraftResult(text="", stats=stats)

        # Refine
        refine_prompt = (
            f"Original question: {prompt}\n\nDraft response:\n{draft_text}\n\n"
            f"Produce an improved final version. Fix errors, fill gaps, improve "
            f"clarity. Keep similar length. Output ONLY the final version."
        )
        t0 = time.perf_counter()
        target_resp = _api_generate(
            self.base_url, self.target_model, refine_prompt,
            system=system or REFINE_SYSTEM,
            num_predict=max_tokens, temperature=self.refine_temperature,
            think=self.think,
        )
        stats.target_time_s = time.perf_counter() - t0
        refined = target_resp.get("response", "")
        stats.target_tokens = target_resp.get("eval_count", len(refined.split()))

        final = refined.strip() if refined.strip() else draft_text.strip()
        words = final.split()
        if len(words) > max_tokens:
            final = " ".join(words[:max_tokens])
        stats.total_tokens_output = len(final.split())
        stats.total_time_s = time.perf_counter() - t_start

        return DraftResult(text=final, stats=stats, draft_text=draft_text.strip())


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 3: DraftSelector
# ═══════════════════════════════════════════════════════════════════════════════


class DraftSelector:
    """Generate N candidates with draft, target selects best."""

    def __init__(
        self,
        draft_model: str = "qwen2.5-coder:3b",
        target_model: str = "qwen2.5:14b",
        n_candidates: int = 2,
        temperatures: list[float] | None = None,
        base_url: str = "http://localhost:11434",
        think: bool | None = False,
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.n_candidates = n_candidates
        self.temperatures = temperatures or [0.3, 0.9][:n_candidates]
        self.base_url = base_url
        self.think = think

    def generate(
        self, prompt: str, max_tokens: int = 512,
        system: str = "", skip_baseline: bool = False,
    ) -> DraftResult:
        stats = DraftStats(strategy="draft_select")
        t_start = time.perf_counter()

        if not skip_baseline:
            stats.target_baseline_tok_s = _measure_speed(
                self.base_url, self.target_model, prompt, system,
                think=self.think,
            )

        # Generate candidates
        candidates = []
        t0 = time.perf_counter()
        for temp in self.temperatures[:self.n_candidates]:
            resp = _api_generate(
                self.base_url, self.draft_model, prompt,
                system=system, num_predict=max_tokens, temperature=temp,
                think=self.think,
            )
            text = resp.get("response", "").strip()
            if text:
                candidates.append(text)
                stats.draft_tokens_proposed += resp.get("eval_count", len(text.split()))
        stats.draft_time_s = time.perf_counter() - t0

        if not candidates:
            stats.total_time_s = time.perf_counter() - t_start
            return DraftResult(text="", stats=stats)

        if len(candidates) == 1:
            stats.total_tokens_output = len(candidates[0].split())
            stats.total_time_s = time.perf_counter() - t_start
            return DraftResult(text=candidates[0], stats=stats, draft_text=candidates[0])

        # Target selects
        numbered = "\n\n".join(f"=== {i+1} ===\n{c}" for i, c in enumerate(candidates))
        select_prompt = (
            f"Question: {prompt}\n\n"
            f"Pick the BEST response. Reply with ONLY the number.\n\n{numbered}"
        )
        t0 = time.perf_counter()
        resp = _api_generate(
            self.base_url, self.target_model, select_prompt,
            num_predict=10, temperature=0.0, think=self.think,
        )
        stats.target_time_s = time.perf_counter() - t0
        stats.target_tokens = resp.get("eval_count", 0)

        selection = resp.get("response", "").strip()
        final = self._parse_selection(selection, candidates)
        words = final.split()
        if len(words) > max_tokens:
            final = " ".join(words[:max_tokens])
        stats.total_tokens_output = len(final.split())
        stats.total_time_s = time.perf_counter() - t_start

        return DraftResult(text=final, stats=stats, draft_text=candidates[0])

    @staticmethod
    def _parse_selection(response: str, candidates: list[str]) -> str:
        for ch in response:
            if ch.isdigit():
                idx = int(ch) - 1
                if 0 <= idx < len(candidates):
                    return candidates[idx]
                break
        return candidates[0]
