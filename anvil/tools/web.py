"""Web tool: search and fetch web content."""

from __future__ import annotations

import ipaddress
import json
import re
import threading
import time
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    import requests

from anvil.utils.fileio import atomic_write
from anvil.utils.logging import get_logger

log = get_logger("tools.web")

__all__ = [
    "WebTool",
    "MAX_RETRIES",
    "RETRY_BACKOFF_BASE",
    "RETRYABLE_STATUS_CODES",
    "ALLOWED_URL_SCHEMES",
    "BLOCKED_HOSTS",
    "MAX_REDIRECTS",
    "MAX_RESPONSE_BYTES",
    "MAX_CONTENT_LENGTH",
    "FETCH_TIMEOUT",
    "RATE_LIMIT_SECONDS",
    "MAX_CACHE_ENTRIES",
    "MAX_RATE_LIMIT_DOMAINS",
    "HTML_DETECT_PREFIX",
]

# Retry configuration for transient network errors
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.5  # seconds — exponential backoff: 1.5, 2.25, 3.375...

# Transient HTTP status codes that warrant a retry
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# URL safety: allowed schemes
ALLOWED_URL_SCHEMES = {"http", "https"}

# URL safety: blocked hostnames (internal/local)
BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "[::1]"}

# Maximum redirects to follow
MAX_REDIRECTS = 5

# Maximum response body size (5 MB)
MAX_RESPONSE_BYTES = 5 * 1024 * 1024

# Maximum content length to return from fetched pages (characters)
MAX_CONTENT_LENGTH = 15000

# HTTP fetch timeout (seconds)
FETCH_TIMEOUT = 15

# Rate limiting: minimum seconds between requests to the same domain
RATE_LIMIT_SECONDS = 2.0

# Cache limits
MAX_CACHE_ENTRIES = 500  # evict oldest when exceeded
MAX_RATE_LIMIT_DOMAINS = 1000  # max tracked domains for rate limiting

# HTML detection: check this many chars of content for "<html" tag
HTML_DETECT_PREFIX = 500

# Pre-compiled HTML-to-text regex patterns (avoids recompilation per fetch)
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_PARAGRAPH_BREAK_RE = re.compile(r"\s{2,}")


class WebTool:
    """Web search and fetch operations."""

    name = "web"
    description = "Search the web and fetch content from URLs"

    CACHE_TTL = 6 * 3600  # 6 hours

    def __init__(self, working_dir: str = ".", cache_dir: str = ""):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(working_dir) / ".anvil_state"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "web_search_cache.json"
        self._cache: dict[str, Any] = self._load_cache()
        # Thread safety locks
        self._cache_lock = threading.Lock()
        self._session_lock = threading.Lock()
        self._rate_limit_lock = threading.Lock()
        # Persistent HTTP session for connection reuse
        self._http_session: requests.Session | None = None
        # Rate limiting: track last request time per domain
        self._domain_last_request: dict[str, float] = {}
        # Cache whether duckduckgo-search is available
        self._ddgs_available: bool | None = None

    def __repr__(self) -> str:
        return f"WebTool(cache_entries={len(self._cache)})"

    def __enter__(self) -> WebTool:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> bool:
        """Exit context manager, closing HTTP session."""
        self.close()
        return False

    def close(self) -> None:
        """Close HTTP session and release resources."""
        with self._session_lock:
            if self._http_session is not None:
                try:
                    self._http_session.close()
                except OSError as e:
                    log.debug("Error closing HTTP session: %s", e)
                self._http_session = None

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return Ollama tool-calling definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web using DuckDuckGo and return results",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "max_results": {"type": "integer", "description": "Max results (default 5)"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_fetch",
                    "description": "Fetch and extract text content from a URL",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL to fetch"},
                        },
                        "required": ["url"],
                    },
                },
            },
        ]

    def execute(self, function_name: str, args: dict[str, Any]) -> str:
        """Execute a web tool function."""
        if function_name == "web_search":
            try:
                max_results = int(args.get("max_results", 5))
            except (TypeError, ValueError):
                max_results = 5
            return self._search(args.get("query", "").strip(), max_results)
        if function_name == "web_fetch":
            return self._fetch(args.get("url", "").strip())
        return f"Unknown function: {function_name}"

    def _search(self, query: str, max_results: int = 5) -> str:
        """Search the web using DuckDuckGo with retry on transient failures."""
        if not query:
            return "Error: empty search query"

        # Clamp max_results to reasonable bounds
        max_results = max(1, min(max_results, 20))

        # Check cache
        cache_key = f"search:{query}:{max_results}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        # Fast path: skip retries if duckduckgo-search is known to be missing
        if self._ddgs_available is False:
            return "Error: duckduckgo-search not installed. Run: pip install duckduckgo-search"

        last_error = ""
        for attempt in range(MAX_RETRIES):
            try:
                from duckduckgo_search import DDGS
                self._ddgs_available = True
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=max_results))

                if not results:
                    return f"No results found for: {query}"

                lines = [f"Search results for: {query}\n"]
                for r in results:
                    title = r.get("title", "")
                    href = r.get("href", "")
                    body = r.get("body", "")
                    lines.append(f"  {title}")
                    lines.append(f"  {href}")
                    lines.append(f"  {body}")
                    lines.append("")

                output = "\n".join(lines)
                self._set_cached(cache_key, output)
                return output

            except ImportError:
                self._ddgs_available = False
                log.warning("duckduckgo-search not installed. Web search disabled.")
                return "Error: duckduckgo-search not installed. Run: pip install duckduckgo-search"
            except (OSError, TimeoutError, ValueError) as e:
                last_error = str(e)
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BACKOFF_BASE ** (attempt + 1)
                    log.warning("Search attempt %d failed (%s), retrying in %.1fs", attempt + 1, e, delay)
                    time.sleep(delay)
                else:
                    log.error("Search failed after %d attempts: %s", MAX_RETRIES, e)

        return f"Search error (after {MAX_RETRIES} attempts): {last_error}"

    def _fetch(self, url: str) -> str:
        """Fetch text content from a URL with retry on transient failures."""
        if not url:
            return "Error: empty URL"

        # Validate URL safety
        url_error = self._validate_url(url)
        if url_error:
            return url_error

        # Rate limiting per domain
        self._apply_rate_limit(url)

        # Check cache
        cache_key = f"fetch:{url}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        last_error = ""
        for attempt in range(MAX_RETRIES):
            try:
                import requests
                with self._session_lock:
                    if self._http_session is None:
                        self._http_session = requests.Session()
                        self._http_session.headers["User-Agent"] = "ollama-anvil/0.1 (local AI assistant)"
                        self._http_session.max_redirects = MAX_REDIRECTS
                    session = self._http_session
                r = session.get(url, timeout=FETCH_TIMEOUT, stream=True)

                # Retry on transient HTTP errors
                if r.status_code in RETRYABLE_STATUS_CODES:
                    last_error = f"HTTP {r.status_code}"
                    r.close()
                    if attempt < MAX_RETRIES - 1:
                        delay = RETRY_BACKOFF_BASE ** (attempt + 1)
                        log.warning("Fetch got HTTP %d (attempt %d), retrying in %.1fs", r.status_code, attempt + 1, delay)
                        time.sleep(delay)
                        continue

                r.raise_for_status()

                # Check Content-Length before reading full body
                content_length = r.headers.get("Content-Length")
                if content_length:
                    try:
                        content_size = int(content_length)
                    except (ValueError, OverflowError):
                        content_size = 0
                    if content_size > MAX_RESPONSE_BYTES:
                        r.close()
                        return f"Error: response too large ({content_size} bytes, max {MAX_RESPONSE_BYTES})"

                # Read with size limit
                content = r.text[:MAX_RESPONSE_BYTES]
                r.close()

                # Check Content-Type for HTML detection (more reliable than content sniffing)
                content_type = r.headers.get("Content-Type", "").lower()
                if "text/html" in content_type or "<html" in content[:HTML_DETECT_PREFIX].lower():
                    content = self._html_to_text(content)

                # Truncate long content
                if len(content) > MAX_CONTENT_LENGTH:
                    content = content[:MAX_CONTENT_LENGTH] + "\n... (truncated)"

                self._set_cached(cache_key, content)
                return content

            except (OSError, TimeoutError, ValueError) as e:
                last_error = str(e)
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BACKOFF_BASE ** (attempt + 1)
                    log.warning("Fetch attempt %d failed (%s), retrying in %.1fs", attempt + 1, e, delay)
                    time.sleep(delay)
                else:
                    log.error("Fetch failed after %d attempts: %s", MAX_RETRIES, e)

        return f"Fetch error (after {MAX_RETRIES} attempts): {last_error}"

    @staticmethod
    def _validate_url(url: str) -> str:
        """Validate a URL for safety. Returns error message or empty string if valid."""
        try:
            parsed = urlparse(url)
        except ValueError as e:
            log.debug("URL parse error for %r: %s", url, e)
            return "Error: malformed URL"

        # Check scheme
        if parsed.scheme not in ALLOWED_URL_SCHEMES:
            return f"Error: URL scheme '{parsed.scheme}' not allowed (use http or https)"

        if not parsed.hostname:
            return "Error: URL has no hostname"

        hostname = parsed.hostname.lower()

        # Block known internal hostnames
        if hostname in BLOCKED_HOSTS:
            return f"Error: cannot fetch from internal host '{hostname}'"

        # Block private/reserved IP ranges
        try:
            addr = ipaddress.ip_address(hostname)
            if addr.is_private or addr.is_loopback or addr.is_reserved or addr.is_link_local:
                return f"Error: cannot fetch from private/reserved IP '{hostname}'"
        except ValueError:
            pass  # Not an IP literal — hostname is fine

        return ""

    def _apply_rate_limit(self, url: str) -> None:
        """Apply per-domain rate limiting to avoid hammering servers."""
        try:
            domain = urlparse(url).hostname or ""
        except ValueError as e:
            log.debug("Could not parse URL for rate limiting: %s", e)
            return
        with self._rate_limit_lock:
            now = time.time()
            last = self._domain_last_request.get(domain, 0)
            wait = max(0.0, RATE_LIMIT_SECONDS - (now - last))
            if wait > 0:
                log.debug("Rate limiting %s: waiting %.1fs", domain, wait)
                time.sleep(wait)
            self._domain_last_request[domain] = time.time()
            # Evict stale entries to prevent unbounded growth
            if len(self._domain_last_request) > MAX_RATE_LIMIT_DOMAINS:
                cutoff = now - 3600  # Remove entries older than 1 hour
                self._domain_last_request = {
                    k: v for k, v in self._domain_last_request.items() if v > cutoff
                }

    @staticmethod
    def _html_to_text(html: str) -> str:
        """Basic HTML to text conversion."""
        # Remove script and style tags
        text = _SCRIPT_STYLE_RE.sub("", html)
        # Remove HTML tags
        text = _HTML_TAG_RE.sub(" ", text)
        # Decode common entities
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("&nbsp;", " ").replace("&quot;", '"')
        # Collapse whitespace
        text = _WHITESPACE_RE.sub(" ", text).strip()
        # Re-add paragraph breaks
        text = _PARAGRAPH_BREAK_RE.sub("\n\n", text)
        return text

    def _load_cache(self) -> dict[str, Any]:
        if self.cache_file.exists():
            try:
                return json.loads(self.cache_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                log.debug("Could not load web cache: %s", e)
        return {}

    def _save_cache(self) -> None:
        try:
            atomic_write(self.cache_file, json.dumps(self._cache))
        except OSError as e:
            log.debug("Could not save web cache: %s", e)

    def _get_cached(self, key: str) -> str | None:
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry and time.time() - entry.get("ts", 0) < self.CACHE_TTL:
                return entry.get("data")
        return None

    def _set_cached(self, key: str, data: str) -> None:
        with self._cache_lock:
            # Evict oldest entries if cache is full
            if len(self._cache) >= MAX_CACHE_ENTRIES:
                oldest_key = min(self._cache, key=lambda k: self._cache[k].get("ts", 0))
                del self._cache[oldest_key]
            self._cache[key] = {"data": data, "ts": time.time()}
            self._save_cache()
