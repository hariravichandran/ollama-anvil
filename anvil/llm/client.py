"""Ollama API client with streaming, tool calling, and model management.

Features:
- Non-streaming and streaming chat/generate
- Tool calling (function calling)
- Structured output (JSON schema)
- Multi-modal (image) input
- Prompt caching via keep_alive
- Model management (pull, delete, switch, list)
"""

from __future__ import annotations

import base64
import json
import re
import threading
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any
from urllib.parse import urlparse

import requests

from anvil.utils.logging import get_logger

log = get_logger("llm.client")

__all__ = [
    "OllamaClient",
    "LLMStats",
    "MIN_NUM_CTX",
    "MAX_NUM_CTX",
    "MODEL_NAME_PATTERN",
    "MAX_RESPONSE_SIZE",
    "LOG_PREVIEW_LENGTH",
    "TIMEOUT_HEALTH",
    "TIMEOUT_LIST",
    "TIMEOUT_DELETE",
    "TIMEOUT_WARMUP",
    "TIMEOUT_PULL",
    "TIMEOUT_MODEL_INFO",
    "TIMEOUT_EMBED",
    "MAX_EMBED_BATCH",
]

# Context window bounds
MIN_NUM_CTX = 128
MAX_NUM_CTX = 131072  # 128K

# Model name validation pattern: alphanumeric, dots, colons, hyphens, underscores, slashes
MODEL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,99}$")

# Maximum response JSON size to parse (50 MB)
MAX_RESPONSE_SIZE = 50 * 1024 * 1024
LOG_PREVIEW_LENGTH = 200  # Truncate log previews

# HTTP timeout values (seconds) for Ollama API calls
TIMEOUT_HEALTH = 5        # Quick checks: version, availability, running list
TIMEOUT_LIST = 10         # Listing models
TIMEOUT_DELETE = 30       # Deleting a model
TIMEOUT_WARMUP = 60       # Warming KV cache
TIMEOUT_PULL = 600        # Pulling a model (large downloads)
TIMEOUT_MODEL_INFO = 10   # Fetching model details
TIMEOUT_EMBED = 120       # Embedding requests (batch of texts)
MAX_EMBED_BATCH = 512     # Max texts per embed call


@dataclass(slots=True)
class LLMStats:
    """Tracks LLM usage statistics."""

    total_calls: int = 0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_time_s: float = 0.0
    errors: int = 0

    @property
    def avg_time_s(self) -> float:
        """Average time per LLM call in seconds."""
        return self.total_time_s / max(1, self.total_calls)

    @property
    def avg_tokens_per_sec(self) -> float:
        """Average tokens generated per second across all calls."""
        return self.total_tokens / max(0.01, self.total_time_s)

    def __repr__(self) -> str:
        return (
            f"LLMStats(calls={self.total_calls}, tokens={self.total_tokens}, "
            f"avg_time={self.avg_time_s:.2f}s, errors={self.errors})"
        )


class OllamaClient:
    """Client for the Ollama REST API."""

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        num_ctx: int = 8192,
        num_thread: int | None = None,
        num_batch: int = 2048,
        keep_alive: str = "30m",
        max_retries: int = 2,
        usage_tracker: Any | None = None,
        performance_monitor: Any | None = None,
        num_gpu: int | None = None,
        num_predict: int | None = None,
    ):
        # Validate base URL
        self.base_url = self._validate_base_url(base_url)
        # Validate model name
        self.model = model  # validated on use, not init (may be placeholder)
        self.temperature = temperature
        # Clamp context window to reasonable bounds
        self.num_ctx = max(MIN_NUM_CTX, min(num_ctx, MAX_NUM_CTX))
        if num_ctx != self.num_ctx:
            log.warning("num_ctx clamped from %d to %d", num_ctx, self.num_ctx)
        self.num_thread = num_thread
        self.num_batch = num_batch
        self.keep_alive = keep_alive  # Keep model in memory between requests
        # GPU layer offloading: number of layers to run on GPU (None = Ollama default)
        self.num_gpu = num_gpu
        # Max output tokens per response (None = unlimited)
        self.num_predict = num_predict
        self.max_retries = max_retries
        self.stats = LLMStats()
        # Optional usage tracker for persistent token accounting
        self._usage_tracker = usage_tracker
        # Optional performance monitor for real-time metrics
        self._performance_monitor = performance_monitor
        # Thread safety locks
        self._stats_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._session_lock = threading.Lock()
        # Persistent HTTP session with connection pooling
        self._session = self._create_session()
        # Model list cache (TTL-based)
        self._models_cache: list[dict[str, Any]] = []
        self._models_cache_time: float = 0
        self._models_cache_ttl: float = 300  # 5 minutes
        # Session lifecycle tracking — recreate session if too old
        self._session_created: float = time.time()
        self._session_max_age: float = 1800  # 30 minutes

    def __repr__(self) -> str:
        num_gpu = getattr(self, "num_gpu", None)
        gpu = f", num_gpu={num_gpu}" if num_gpu is not None else ""
        return f"OllamaClient(model={self.model!r}, base_url={self.base_url!r}, ctx={self.num_ctx}{gpu})"

    def __enter__(self) -> OllamaClient:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> bool:
        """Exit context manager, closing HTTP session."""
        self.close()
        return False

    def close(self) -> None:
        """Close the HTTP session and release resources."""
        with self._session_lock:
            if self._session:
                self._session.close()
                log.debug("Closed HTTP session for OllamaClient")

    def adapt_to_pressure(self) -> dict[str, Any]:
        """Adjust context window and batch size based on current RAM pressure.

        Reads real-time memory pressure from ResourceManager and reduces
        num_ctx and num_batch if the system is under high or critical pressure.
        Returns a dict with the old and new values plus the pressure level.
        """
        from anvil.llm.resource import ResourceManager
        mgr = ResourceManager()
        status = mgr.check_pressure(target_context=self.num_ctx)

        old_ctx = self.num_ctx
        old_batch = self.num_batch

        if status.level in ("high", "critical"):
            self.num_ctx = min(self.num_ctx, status.recommended_context)
            self.num_batch = min(self.num_batch, status.recommended_batch)
            log.warning(
                "RAM pressure %s — reduced ctx %d→%d, batch %d→%d",
                status.level, old_ctx, self.num_ctx, old_batch, self.num_batch,
            )

        return {
            "pressure": status.level,
            "old_ctx": old_ctx,
            "new_ctx": self.num_ctx,
            "old_batch": old_batch,
            "new_batch": self.num_batch,
            "free_ram_mb": status.free_ram_mb,
            "changed": old_ctx != self.num_ctx or old_batch != self.num_batch,
        }

    @staticmethod
    def _validate_base_url(url: str) -> str:
        """Validate and normalize the base URL."""
        url = url.rstrip("/")
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            log.warning("Invalid URL scheme '%s', defaulting to http://localhost:11434", parsed.scheme)
            return "http://localhost:11434"
        if not parsed.hostname:
            log.warning("No hostname in URL '%s', defaulting to http://localhost:11434", url)
            return "http://localhost:11434"
        return url

    @staticmethod
    def validate_model_name(name: str) -> str:
        """Validate a model name. Returns error string or empty if valid."""
        if not name or not name.strip():
            return "Model name cannot be empty"
        if len(name) > 100:
            return f"Model name too long ({len(name)} chars, max 100)"
        if not MODEL_NAME_PATTERN.match(name):
            return f"Invalid model name format: {name}"
        if ".." in name:
            return "Model name cannot contain '..'"
        return ""

    def _record_usage(self, prompt_tokens: int, completion_tokens: int, time_s: float = 0.0) -> None:
        """Forward metrics to the usage tracker and performance monitor."""
        if self._usage_tracker is not None:
            try:
                self._usage_tracker.record(
                    model=self.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            except (AttributeError, TypeError, ValueError) as exc:
                log.debug("Usage tracker record failed: %s", exc)
        if self._performance_monitor is not None:
            try:
                self._performance_monitor.record(
                    tokens=completion_tokens,
                    prompt_tokens=prompt_tokens,
                    time_s=time_s,
                    model=self.model,
                )
            except (AttributeError, TypeError, ValueError) as exc:
                log.debug("Performance monitor record failed: %s", exc)

    @staticmethod
    def _safe_json(response: requests.Response) -> dict[str, Any]:
        """Safely parse JSON response, handling malformed data."""
        try:
            if len(response.content) > MAX_RESPONSE_SIZE:
                log.warning("Response too large to parse (%d bytes)", len(response.content))
                return {"error": "Response too large"}
            return response.json()
        except (json.JSONDecodeError, ValueError) as e:
            log.error("Malformed JSON response: %s", str(e)[:LOG_PREVIEW_LENGTH])
            return {"error": f"Malformed JSON: {e}"}

    @staticmethod
    def _backoff_delay(attempt: int, base: float = 1.0, max_delay: float = 10.0) -> float:
        """Calculate exponential backoff delay: base * 2^attempt, capped at max_delay."""
        delay = min(base * (2 ** attempt), max_delay)
        return delay

    def _get_session(self) -> requests.Session:
        """Get HTTP session, recreating if stale to prevent connection issues."""
        with self._session_lock:
            if time.time() - self._session_created > self._session_max_age:
                self._session.close()
                self._session = self._create_session()
                self._session_created = time.time()
                log.debug("Recreated HTTP session (max age exceeded)")
            return self._session

    @staticmethod
    def _create_session() -> requests.Session:
        """Create a new HTTP session with connection pooling."""
        session = requests.Session()
        from requests.adapters import HTTPAdapter
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=8)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            r = self._get_session().get(f"{self.base_url}/api/version", timeout=TIMEOUT_HEALTH)
            return r.status_code == 200
        except (requests.ConnectionError, requests.exceptions.RequestException, OSError):
            return False

    def reconnect(self) -> bool:
        """Force-recreate the HTTP session and verify connection.

        Useful when Ollama has been restarted or the connection was lost.
        Closes the old session and creates a fresh one.

        Returns:
            True if reconnection succeeded, False otherwise.
        """
        with self._session_lock:
            self._session.close()
            self._session = self._create_session()
            self._session_created = time.time()
            log.info("Recreated HTTP session (manual reconnect)")
        return self.is_available()

    def ensure_connection(self, max_attempts: int = 3, base_delay: float = 1.0) -> bool:
        """Ensure connection to Ollama, retrying with backoff if needed.

        Tries to connect to Ollama server, recreating the session and
        retrying with exponential backoff on failure. This is useful when
        Ollama might be restarting or temporarily unavailable.

        Args:
            max_attempts: Maximum number of connection attempts.
            base_delay: Base delay in seconds between retries.

        Returns:
            True if connection is established, False if all attempts fail.
        """
        for attempt in range(max_attempts):
            if self.is_available():
                return True
            if attempt < max_attempts - 1:
                delay = self._backoff_delay(attempt, base_delay, max_delay=10.0)
                log.info("Connection attempt %d/%d failed, retrying in %.1fs",
                         attempt + 1, max_attempts, delay)
                time.sleep(delay)
                self.reconnect()
        return False

    def get_version(self) -> str:
        """Get Ollama server version."""
        try:
            r = self._get_session().get(f"{self.base_url}/api/version", timeout=TIMEOUT_HEALTH)
            if r.status_code == 200:
                return r.json().get("version", "unknown")
        except requests.ConnectionError as e:
            log.debug("Could not get Ollama version: %s", e)
        return "unavailable"

    def list_models(self) -> list[dict[str, Any]]:
        """List locally available models (cached with 5-minute TTL)."""
        with self._cache_lock:
            now = time.time()
            if self._models_cache and (now - self._models_cache_time) < self._models_cache_ttl:
                return self._models_cache
            try:
                r = self._get_session().get(f"{self.base_url}/api/tags", timeout=TIMEOUT_LIST)
                if r.status_code == 200:
                    self._models_cache = r.json().get("models", [])
                    self._models_cache_time = now
                    return self._models_cache
            except requests.ConnectionError:
                log.error("Cannot connect to Ollama at %s", self.base_url)
            return self._models_cache or []

    def list_running(self) -> list[dict[str, Any]]:
        """List models currently loaded in memory."""
        try:
            r = self._get_session().get(f"{self.base_url}/api/ps", timeout=TIMEOUT_HEALTH)
            if r.status_code == 200:
                return r.json().get("models", [])
        except requests.ConnectionError as e:
            log.debug("Could not list running models: %s", e)
        return []

    def pull_model(
        self,
        model: str,
        progress_cb: Callable[[str], None] | None = None,
        max_retries: int = 3,
    ) -> bool:
        """Pull a model from the Ollama registry with automatic retry.

        Ollama's pull API supports resume — if a download is interrupted,
        re-issuing the same pull request continues from where it left off.
        This method retries on network errors with exponential backoff.

        Args:
            model: Model name (e.g., 'qwen2.5-coder:7b').
            progress_cb: Called with progress string for each update.
                The string includes download percentage when available.
            max_retries: Maximum number of retry attempts on network failure.

        Returns:
            True if pull succeeded.
        """
        # Validate model name
        validation_err = self.validate_model_name(model)
        if validation_err:
            log.error("Invalid model name for pull: %s", validation_err)
            return False

        for attempt in range(1 + max_retries):
            if attempt > 0:
                delay = self._backoff_delay(attempt - 1, base=2.0, max_delay=30.0)
                log.info(
                    "Retrying pull for %s (attempt %d/%d) in %.0fs...",
                    model, attempt + 1, 1 + max_retries, delay,
                )
                if progress_cb:
                    progress_cb(f"Connection lost — retrying in {delay:.0f}s (attempt {attempt + 1}/{1 + max_retries})")
                time.sleep(delay)

            log.info("Pulling model: %s", model)
            r = None
            try:
                r = self._get_session().post(
                    f"{self.base_url}/api/pull",
                    json={"name": model, "stream": True},
                    stream=True,
                    timeout=TIMEOUT_PULL,
                )
                for line in r.iter_lines():
                    if line:
                        data = json.loads(line)
                        status = data.get("status", "")

                        # Format progress with percentage if available
                        total = data.get("total", 0)
                        completed = data.get("completed", 0)
                        if total and completed:
                            pct = int(completed / total * 100)
                            size_gb = total / (1024 ** 3)
                            progress_str = f"{status} — {pct}% of {size_gb:.1f} GB"
                        else:
                            progress_str = status

                        if progress_cb:
                            progress_cb(progress_str)

                        if "error" in data:
                            log.error("Pull error: %s", data["error"])
                            return False
                log.info("Successfully pulled %s", model)
                # Invalidate models cache so list_models() reflects the new model
                with self._cache_lock:
                    self._models_cache_time = 0
                return True
            except (requests.ConnectionError, requests.Timeout) as e:
                log.warning("Pull attempt %d failed for %s: %s", attempt + 1, model, e)
                if attempt >= max_retries:
                    log.error("Failed to pull %s after %d attempts", model, 1 + max_retries)
                    return False
            finally:
                if r is not None:
                    r.close()

        return False  # unreachable, but satisfies type checker

    def delete_model(self, model: str) -> bool:
        """Delete a locally stored model."""
        validation_err = self.validate_model_name(model)
        if validation_err:
            log.error("Invalid model name for delete: %s", validation_err)
            return False

        try:
            r = self._get_session().delete(
                f"{self.base_url}/api/delete",
                json={"name": model},
                timeout=TIMEOUT_DELETE,
            )
            if r.status_code == 200:
                # Invalidate models cache so list_models() reflects the deletion
                with self._cache_lock:
                    self._models_cache_time = 0
                return True
            log.warning("Delete model %s returned HTTP %d", model, r.status_code)
            return False
        except requests.ConnectionError as e:
            log.error("Connection error deleting model %s: %s", model, e)
            return False

    def generate(
        self,
        prompt: str,
        system: str = "",
        json_mode: bool = False,
        json_schema: dict | None = None,
        timeout: int = 300,
        temperature: float | None = None,
        model: str | None = None,
        think: bool = False,
    ) -> dict[str, Any]:
        """Generate a completion (non-streaming).

        Args:
            prompt: The prompt text.
            system: Optional system prompt.
            json_mode: If True, forces JSON output.
            json_schema: If provided, forces output to match the given
                JSON schema (Ollama structured output). Takes precedence
                over json_mode.
            timeout: Request timeout in seconds.
            temperature: Override temperature for this call.
            model: Override model for this call.
            think: If True, enable thinking/reasoning mode for models
                that support it (e.g., deepseek-r1, qwq). The model's
                chain-of-thought is returned in the 'thinking' key.

        Returns dict with keys: response, tokens, time_s, tokens_per_sec,
            and optionally 'thinking' (str) when think=True.
        """
        payload: dict[str, Any] = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_ctx": self.num_ctx,
                "num_batch": self.num_batch,
            },
        }
        if self.num_thread:
            payload["options"]["num_thread"] = self.num_thread
        if self.num_gpu is not None:
            payload["options"]["num_gpu"] = self.num_gpu
        if self.num_predict is not None:
            payload["options"]["num_predict"] = self.num_predict
        if system:
            payload["system"] = system
        if json_schema:
            # Ollama structured output: pass the full JSON schema as format
            payload["format"] = json_schema
        elif json_mode:
            payload["format"] = "json"
        if think:
            payload["think"] = True

        last_error = ""
        for attempt in range(1 + self.max_retries):
            start = time.time()
            try:
                r = self._get_session().post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=(5, timeout),
                )
                elapsed = time.time() - start

                if r.status_code != 200:
                    with self._stats_lock:
                        self.stats.errors += 1
                    last_error = r.text
                    if attempt < self.max_retries:
                        delay = self._backoff_delay(attempt)
                        log.warning("Generate failed (attempt %d), retrying in %.1fs: %s", attempt + 1, delay, r.text[:LOG_PREVIEW_LENGTH])
                        time.sleep(delay)
                        continue
                    return {"response": "", "tokens": 0, "time_s": elapsed, "tokens_per_sec": 0, "error": r.text}

                data = self._safe_json(r)
                if "error" in data and "response" not in data:
                    with self._stats_lock:
                        self.stats.errors += 1
                    return {"response": "", "tokens": 0, "time_s": elapsed, "tokens_per_sec": 0, "error": data["error"]}
                tokens = data.get("eval_count", 0)
                prompt_tokens = data.get("prompt_eval_count", 0)
                with self._stats_lock:
                    self.stats.total_calls += 1
                    self.stats.total_tokens += tokens
                    self.stats.total_prompt_tokens += prompt_tokens
                    self.stats.total_time_s += elapsed
                self._record_usage(prompt_tokens, tokens, elapsed)

                result = {
                    "response": data.get("response", ""),
                    "tokens": tokens,
                    "prompt_tokens": prompt_tokens,
                    "time_s": elapsed,
                    "tokens_per_sec": tokens / max(0.01, elapsed),
                }
                # Include thinking content when present (reasoning models)
                if data.get("thinking"):
                    result["thinking"] = data["thinking"]
                return result
            except requests.Timeout:
                with self._stats_lock:
                    self.stats.errors += 1
                last_error = "timeout"
                if attempt < self.max_retries:
                    delay = self._backoff_delay(attempt)
                    log.warning("Generate timed out (attempt %d), retrying in %.1fs", attempt + 1, delay)
                    time.sleep(delay)
                    continue
                return {"response": "", "tokens": 0, "time_s": time.time() - start, "tokens_per_sec": 0, "error": "timeout"}
            except (requests.ConnectionError, requests.exceptions.RequestException, OSError) as e:
                with self._stats_lock:
                    self.stats.errors += 1
                last_error = str(e)
                if attempt < self.max_retries:
                    delay = self._backoff_delay(attempt)
                    log.warning("Generate error (attempt %d), retrying in %.1fs: %s", attempt + 1, delay, e)
                    time.sleep(delay)
                    continue
                return {"response": "", "tokens": 0, "time_s": 0, "tokens_per_sec": 0, "error": str(e)}

        return {"response": "", "tokens": 0, "time_s": 0, "tokens_per_sec": 0, "error": last_error}

    def chat(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        tools: list[dict] | None = None,
        json_mode: bool = False,
        json_schema: dict | None = None,
        images: list[str] | None = None,
        timeout: int = 300,
        temperature: float | None = None,
        model: str | None = None,
        think: bool = False,
    ) -> dict[str, Any]:
        """Chat completion (non-streaming).

        Args:
            messages: list of {"role": "user"|"assistant"|"system", "content": "..."}
            tools: Ollama tool definitions for function calling
            json_mode: If True, forces JSON output
            json_schema: If provided, forces output to match the JSON schema
            images: List of image paths or base64 strings for vision models.
                    Paths are auto-converted to base64.
            timeout: Request timeout in seconds
            temperature: Override temperature for this call
            model: Override model for this call
            think: If True, enable thinking/reasoning mode. The model's
                chain-of-thought is returned in the 'thinking' key.

        Returns dict with: response, tokens, time_s, tool_calls (if any),
            and optionally 'thinking' (str) when think=True.
        """
        # If images provided, add them to the last user message
        if images:
            messages = self._inject_images(messages, images)

        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_ctx": self.num_ctx,
                "num_batch": self.num_batch,
            },
        }
        if self.num_thread:
            payload["options"]["num_thread"] = self.num_thread
        if self.num_gpu is not None:
            payload["options"]["num_gpu"] = self.num_gpu
        if self.num_predict is not None:
            payload["options"]["num_predict"] = self.num_predict
        if tools:
            payload["tools"] = tools
        if json_schema:
            payload["format"] = json_schema
        elif json_mode:
            payload["format"] = "json"
        if think:
            payload["think"] = True

        last_error = ""
        for attempt in range(1 + self.max_retries):
            start = time.time()
            try:
                r = self._get_session().post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=(5, timeout),
                )
                elapsed = time.time() - start

                if r.status_code != 200:
                    with self._stats_lock:
                        self.stats.errors += 1
                    last_error = r.text
                    if attempt < self.max_retries:
                        delay = self._backoff_delay(attempt)
                        log.warning("Chat failed (attempt %d), retrying in %.1fs: %s", attempt + 1, delay, r.text[:LOG_PREVIEW_LENGTH])
                        time.sleep(delay)
                        continue
                    return {"response": "", "tokens": 0, "time_s": elapsed, "error": r.text}

                data = self._safe_json(r)
                if "error" in data and "message" not in data:
                    with self._stats_lock:
                        self.stats.errors += 1
                    return {"response": "", "tokens": 0, "time_s": elapsed, "error": data["error"]}
                msg = data.get("message", {})
                tokens = data.get("eval_count", 0)
                prompt_tokens = data.get("prompt_eval_count", 0)

                with self._stats_lock:
                    self.stats.total_calls += 1
                    self.stats.total_tokens += tokens
                    self.stats.total_prompt_tokens += prompt_tokens
                    self.stats.total_time_s += elapsed
                self._record_usage(prompt_tokens, tokens, elapsed)

                result: dict[str, Any] = {
                    "response": msg.get("content", ""),
                    "tokens": tokens,
                    "prompt_tokens": prompt_tokens,
                    "time_s": elapsed,
                    "tokens_per_sec": tokens / max(0.01, elapsed),
                }
                if msg.get("tool_calls"):
                    result["tool_calls"] = msg["tool_calls"]
                # Include thinking content when present (reasoning models)
                if msg.get("thinking"):
                    result["thinking"] = msg["thinking"]

                return result
            except requests.Timeout:
                with self._stats_lock:
                    self.stats.errors += 1
                last_error = "timeout"
                if attempt < self.max_retries:
                    delay = self._backoff_delay(attempt)
                    log.warning("Chat timed out (attempt %d), retrying in %.1fs", attempt + 1, delay)
                    time.sleep(delay)
                    continue
                return {"response": "", "tokens": 0, "time_s": time.time() - start, "error": "timeout"}
            except (requests.ConnectionError, requests.exceptions.RequestException, OSError) as e:
                with self._stats_lock:
                    self.stats.errors += 1
                last_error = str(e)
                if attempt < self.max_retries:
                    delay = self._backoff_delay(attempt)
                    log.warning("Chat error (attempt %d), retrying in %.1fs: %s", attempt + 1, delay, e)
                    time.sleep(delay)
                    continue
                return {"response": "", "tokens": 0, "time_s": 0, "error": str(e)}

        return {"response": "", "tokens": 0, "time_s": 0, "error": last_error}

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        tools: list[dict] | None = None,
        json_mode: bool = False,
        json_schema: dict | None = None,
        images: list[str] | None = None,
        timeout: int = 300,
        temperature: float | None = None,
        model: str | None = None,
        think: bool = False,
    ) -> Generator[dict[str, Any], None, None]:
        """Streaming chat — yields event dicts as they arrive.

        Each yielded dict has a "type" key:
        - {"type": "text", "content": "..."} — text token
        - {"type": "thinking", "content": "..."} — reasoning token (think=True)
        - {"type": "tool_call", "tool_calls": [...]} — tool call request
        - {"type": "done", "tokens": N, "time_s": N} — generation complete
        - {"type": "error", "error": "..."} — error

        Args:
            json_mode: If True, forces JSON output.
            json_schema: If provided, forces output to match the JSON schema.
            think: If True, enable thinking mode. Reasoning tokens are
                yielded as {"type": "thinking", "content": "..."} events.
        """
        if images:
            messages = self._inject_images(messages, images)

        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_ctx": self.num_ctx,
                "num_batch": self.num_batch,
            },
        }
        if self.num_thread:
            payload["options"]["num_thread"] = self.num_thread
        if self.num_gpu is not None:
            payload["options"]["num_gpu"] = self.num_gpu
        if self.num_predict is not None:
            payload["options"]["num_predict"] = self.num_predict
        if tools:
            payload["tools"] = tools
        if json_schema:
            payload["format"] = json_schema
        elif json_mode:
            payload["format"] = "json"
        if think:
            payload["think"] = True

        for attempt in range(1 + self.max_retries):
            start = time.time()
            r = None
            try:
                r = self._get_session().post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    stream=True,
                    timeout=(5, timeout),
                )
                for line in r.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                        except (json.JSONDecodeError, ValueError):
                            log.debug("Skipping malformed stream line: %s", line[:LOG_PREVIEW_LENGTH])
                            continue
                        msg = data.get("message", {})

                        # Thinking content (reasoning models with think=True)
                        thinking = msg.get("thinking", "")
                        if thinking:
                            yield {"type": "thinking", "content": thinking}

                        # Text content
                        content = msg.get("content", "")
                        if content:
                            yield {"type": "text", "content": content}

                        # Tool calls
                        if msg.get("tool_calls"):
                            yield {"type": "tool_call", "tool_calls": msg["tool_calls"]}

                        # Done signal
                        if data.get("done"):
                            total_tokens = data.get("eval_count", 0)
                            prompt_tokens = data.get("prompt_eval_count", 0)
                            elapsed = time.time() - start
                            with self._stats_lock:
                                self.stats.total_calls += 1
                                self.stats.total_tokens += total_tokens
                                self.stats.total_prompt_tokens += prompt_tokens
                                self.stats.total_time_s += elapsed
                            self._record_usage(prompt_tokens, total_tokens, elapsed)
                            yield {
                                "type": "done",
                                "tokens": total_tokens,
                                "time_s": round(elapsed, 2),
                                "tokens_per_sec": round(total_tokens / max(0.01, elapsed), 1),
                            }
                return  # Stream completed successfully
            except (requests.ConnectionError, requests.Timeout, OSError) as e:
                with self._stats_lock:
                    self.stats.errors += 1
                if attempt < self.max_retries:
                    delay = self._backoff_delay(attempt)
                    log.warning("Stream chat error (attempt %d), retrying in %.1fs: %s", attempt + 1, delay, e)
                    time.sleep(delay)
                    continue
                yield {"type": "error", "error": str(e)}
            finally:
                if r is not None:
                    r.close()

    def show_model(self, model: str | None = None) -> dict[str, Any]:
        """Get model details (parameters, template, capabilities).

        Useful for checking if a model supports vision, tools, etc.
        """
        try:
            r = self._get_session().post(
                f"{self.base_url}/api/show",
                json={"name": model or self.model},
                timeout=TIMEOUT_MODEL_INFO,
            )
            if r.status_code == 200:
                return r.json()
        except (requests.ConnectionError, requests.Timeout) as e:
            log.debug("Could not fetch model info: %s", e)
        return {}

    def warmup(self, system: str = "") -> bool:
        """Pre-warm the KV cache by sending the system prompt with no generation.

        This pre-computes the KV cache for the system prompt so subsequent
        requests that share the same prefix are faster.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": system or "You are a helpful assistant.",
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "num_predict": 1,  # Generate minimal tokens — just warm the cache
                "num_ctx": self.num_ctx,
            },
        }
        try:
            r = self._get_session().post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=TIMEOUT_WARMUP,
            )
            if r.status_code == 200:
                log.info("Warmed KV cache for %s", self.model)
                return True
        except (requests.ConnectionError, requests.Timeout) as e:
            log.debug("Could not warm KV cache: %s", e)
        return False

    # Maximum image file size (20 MB) — prevents sending huge files to Ollama
    MAX_IMAGE_SIZE = 20 * 1024 * 1024

    # Supported image MIME types (by file extension)
    SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

    def _inject_images(
        self, messages: list[dict[str, Any]], images: list[str],
    ) -> list[dict[str, Any]]:
        """Inject base64-encoded images into the last user message.

        Accepts file paths or raw base64 strings.
        Validates file existence, size, and extension before encoding.
        """
        encoded = []
        for img in images:
            img_path = Path(img)
            if img_path.exists():
                # Validate extension
                if img_path.suffix.lower() not in self.SUPPORTED_IMAGE_EXTENSIONS:
                    log.warning(
                        "Skipping unsupported image type: %s (supported: %s)",
                        img_path.suffix, ", ".join(sorted(self.SUPPORTED_IMAGE_EXTENSIONS)),
                    )
                    continue
                # Validate size
                file_size = img_path.stat().st_size
                if file_size > self.MAX_IMAGE_SIZE:
                    log.warning(
                        "Skipping oversized image: %s (%.1f MB, max %.0f MB)",
                        img_path.name, file_size / (1024 * 1024),
                        self.MAX_IMAGE_SIZE / (1024 * 1024),
                    )
                    continue
                if file_size == 0:
                    log.warning("Skipping empty image file: %s", img_path.name)
                    continue
                # File path — read and encode
                encoded.append(base64.b64encode(img_path.read_bytes()).decode("utf-8"))
            else:
                # Assume already base64
                encoded.append(img)

        if not encoded:
            return messages

        # Find last user message and add images
        messages = [dict(m) for m in messages]  # shallow copy
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                messages[i]["images"] = encoded
                break

        return messages

    def switch_model(self, model: str) -> bool:
        """Switch to a different model, pulling it if necessary."""
        validation_err = self.validate_model_name(model)
        if validation_err:
            log.error("Invalid model name: %s", validation_err)
            return False

        available = [m.get("name", "") for m in self.list_models()]
        # Normalize names (strip :latest)
        available_base = [n.split(":")[0] for n in available]
        model_base = model.split(":")[0]

        if model not in available and model_base not in available_base:
            log.info("Model %s not found locally, pulling...", model)
            if not self.pull_model(model):
                return False

        self.model = model
        log.info("Switched to model: %s", model)
        return True

    def embed(
        self,
        texts: str | list[str],
        model: str | None = None,
        timeout: int = TIMEOUT_EMBED,
    ) -> dict[str, Any]:
        """Generate embeddings for one or more texts using Ollama's /api/embed.

        Args:
            texts: A single string or list of strings to embed.
            model: Embedding model to use (e.g., 'nomic-embed-text').
                Defaults to the client's current model.
            timeout: Request timeout in seconds.

        Returns dict with keys:
            - embeddings: list of float vectors (one per input text)
            - model: the model used
            - error: error string (only on failure)

        Example::

            result = client.embed("Hello world", model="nomic-embed-text")
            vector = result["embeddings"][0]  # list[float]
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return {"embeddings": [], "model": model or self.model}

        if len(texts) > MAX_EMBED_BATCH:
            log.warning(
                "Embed batch too large (%d texts), truncating to %d",
                len(texts), MAX_EMBED_BATCH,
            )
            texts = texts[:MAX_EMBED_BATCH]

        payload: dict[str, Any] = {
            "model": model or self.model,
            "input": texts,
        }

        try:
            r = self._get_session().post(
                f"{self.base_url}/api/embed",
                json=payload,
                timeout=(5, timeout),
            )
            if r.status_code != 200:
                log.error("Embed failed with HTTP %d: %s", r.status_code, r.text[:LOG_PREVIEW_LENGTH])
                return {"embeddings": [], "model": payload["model"], "error": r.text}

            data = self._safe_json(r)
            if "error" in data:
                return {"embeddings": [], "model": payload["model"], "error": data["error"]}

            embeddings = data.get("embeddings", [])
            log.debug("Generated %d embeddings (dim=%d)", len(embeddings), len(embeddings[0]) if embeddings else 0)
            return {
                "embeddings": embeddings,
                "model": data.get("model", payload["model"]),
            }
        except requests.Timeout:
            log.error("Embed request timed out after %ds", timeout)
            return {"embeddings": [], "model": payload["model"], "error": "timeout"}
        except (requests.ConnectionError, requests.exceptions.RequestException, OSError) as e:
            log.error("Embed connection error: %s", e)
            return {"embeddings": [], "model": payload["model"], "error": str(e)}
