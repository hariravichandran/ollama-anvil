"""Model health check — verify models are functional and diagnose issues.

Runs a quick diagnostic on one or all locally-installed models to ensure
they can generate responses, support expected features, and perform
within acceptable latency bounds.

Usage::

    from anvil.llm.client import OllamaClient
    from anvil.llm.health import HealthChecker

    client = OllamaClient()
    checker = HealthChecker(client)

    # Check a specific model
    result = checker.check("qwen2.5-coder:7b")
    print(result.summary())
    # qwen2.5-coder:7b: HEALTHY (0.8s latency, 45.2 tok/s)

    # Check all local models
    results = checker.check_all()
    for r in results:
        print(r.summary())
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from anvil.utils.logging import get_logger

log = get_logger("llm.health")

__all__ = [
    "HealthChecker",
    "HealthResult",
    "HealthStatus",
    "HEALTH_PROMPT",
    "HEALTH_TIMEOUT",
    "LATENCY_WARN_THRESHOLD",
    "LATENCY_ERROR_THRESHOLD",
    "MIN_TOKENS_EXPECTED",
]

# Health check configuration
HEALTH_PROMPT = "Say 'hello' in one word."
HEALTH_TIMEOUT = 30            # seconds
LATENCY_WARN_THRESHOLD = 10.0  # seconds — warn if slower
LATENCY_ERROR_THRESHOLD = 25.0 # seconds — error if slower
MIN_TOKENS_EXPECTED = 1        # minimum tokens in valid response


class HealthStatus:
    """Health check result statuses."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    ERROR = "error"
    NOT_FOUND = "not_found"


@dataclass(frozen=True, slots=True)
class HealthResult:
    """Result of a model health check."""

    model: str
    status: str
    latency_s: float
    tokens_generated: int
    tokens_per_sec: float
    response_preview: str
    error: str = ""
    warnings: tuple[str, ...] = ()

    def summary(self) -> str:
        """Human-readable health summary."""
        status_icon = {
            HealthStatus.HEALTHY: "OK",
            HealthStatus.DEGRADED: "WARN",
            HealthStatus.UNHEALTHY: "FAIL",
            HealthStatus.ERROR: "ERR",
            HealthStatus.NOT_FOUND: "N/A",
        }.get(self.status, "???")

        base = f"{self.model}: {status_icon}"
        if self.status == HealthStatus.ERROR:
            return f"{base} — {self.error}"
        if self.status == HealthStatus.NOT_FOUND:
            return f"{base} — model not installed"

        parts = [base]
        parts.append(f"({self.latency_s:.1f}s latency, {self.tokens_per_sec:.1f} tok/s)")
        if self.warnings:
            parts.append(f"Warnings: {'; '.join(self.warnings)}")
        return " ".join(parts)

    @property
    def is_ok(self) -> bool:
        """Whether the model is healthy or only mildly degraded."""
        return self.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)


class HealthChecker:
    """Checks model health by running a quick generation test.

    Args:
        client: OllamaClient instance.
        timeout: Maximum time to wait for a response.
    """

    def __init__(self, client: Any, timeout: int = HEALTH_TIMEOUT):
        self._client = client
        self._timeout = timeout

    def __repr__(self) -> str:
        return f"HealthChecker(timeout={self._timeout}s)"

    def check(self, model: str) -> HealthResult:
        """Run a health check on a specific model.

        Args:
            model: Model name to check.

        Returns:
            HealthResult with status and diagnostics.
        """
        warnings: list[str] = []

        # Verify model exists locally
        try:
            models = self._client.list_models()
            local_names = [m.get("name", "") for m in models]
            if model not in local_names:
                return HealthResult(
                    model=model,
                    status=HealthStatus.NOT_FOUND,
                    latency_s=0.0,
                    tokens_generated=0,
                    tokens_per_sec=0.0,
                    response_preview="",
                    error="Model not installed locally",
                )
        except (OSError, ValueError, RuntimeError) as exc:
            return HealthResult(
                model=model,
                status=HealthStatus.ERROR,
                latency_s=0.0,
                tokens_generated=0,
                tokens_per_sec=0.0,
                response_preview="",
                error=f"Cannot connect to Ollama: {exc}",
            )

        # Run generation test
        start = time.time()
        try:
            result = self._client.generate(
                prompt=HEALTH_PROMPT,
                model=model,
                timeout=self._timeout,
            )
            elapsed = time.time() - start
        except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
            return HealthResult(
                model=model,
                status=HealthStatus.ERROR,
                latency_s=time.time() - start,
                tokens_generated=0,
                tokens_per_sec=0.0,
                response_preview="",
                error=f"Generation failed: {exc}",
            )

        # Analyze result
        response_text = result.get("response", "")
        tokens = result.get("eval_count", len(response_text.split()))
        tps = tokens / max(0.01, elapsed)

        # Check response quality
        if tokens < MIN_TOKENS_EXPECTED:
            warnings.append("Empty or very short response")

        # Check latency
        if elapsed > LATENCY_ERROR_THRESHOLD:
            return HealthResult(
                model=model,
                status=HealthStatus.UNHEALTHY,
                latency_s=round(elapsed, 2),
                tokens_generated=tokens,
                tokens_per_sec=round(tps, 1),
                response_preview=response_text[:100],
                warnings=tuple(warnings),
                error=f"Latency {elapsed:.1f}s exceeds {LATENCY_ERROR_THRESHOLD}s threshold",
            )

        if elapsed > LATENCY_WARN_THRESHOLD:
            warnings.append(f"High latency: {elapsed:.1f}s")

        status = HealthStatus.DEGRADED if warnings else HealthStatus.HEALTHY

        return HealthResult(
            model=model,
            status=status,
            latency_s=round(elapsed, 2),
            tokens_generated=tokens,
            tokens_per_sec=round(tps, 1),
            response_preview=response_text[:100],
            warnings=tuple(warnings),
        )

    def check_all(self) -> list[HealthResult]:
        """Check all locally installed models.

        Returns:
            List of HealthResult, one per local model.
        """
        try:
            models = self._client.list_models()
        except (OSError, ValueError, RuntimeError) as exc:
            log.error("Cannot list models: %s", exc)
            return [HealthResult(
                model="(unknown)",
                status=HealthStatus.ERROR,
                latency_s=0.0,
                tokens_generated=0,
                tokens_per_sec=0.0,
                response_preview="",
                error=f"Cannot connect to Ollama: {exc}",
            )]

        results = []
        for model_info in models:
            name = model_info.get("name", "")
            if not name:
                continue
            # Skip embedding models (they can't generate text)
            if "embed" in name.lower():
                continue
            log.info("Checking model: %s", name)
            results.append(self.check(name))

        return results
