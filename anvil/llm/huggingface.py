"""Hugging Face model discovery and pull helpers.

Ollama natively pulls GGUF models from Hugging Face via ``hf.co/<org>/<repo>[:<quant>]``.
This module adds two things forge-side:

1. Search HF's public API for GGUF repos (with optional domain filters).
2. Enumerate the quantization variants published in a given repo, so the user
   can pick one that fits their hardware before pulling.

No ``huggingface_hub`` dependency — we use the public REST API over ``requests``.
Private/gated repos require an ``HF_TOKEN`` env var.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

import requests

from anvil.utils.logging import get_logger

log = get_logger("llm.huggingface")

__all__ = [
    "HFModel",
    "HFQuant",
    "search_models",
    "list_quants",
    "to_ollama_ref",
    "DOMAIN_FILTERS",
]

HF_API = "https://huggingface.co/api"
TIMEOUT_SEARCH = 15
TIMEOUT_FILES = 15

# Matches quantization tokens in GGUF filenames: Q4_K_M, IQ3_XXS, Q8_0, F16, BF16, etc.
# Case-insensitive; we normalize to uppercase.
QUANT_PATTERN = re.compile(
    r"(?P<quant>(?:IQ|Q)[0-9]+(?:_[A-Z0-9]+)*|F16|BF16|F32)",
    re.IGNORECASE,
)

# Keyword bundles per domain. HF's ``search`` param ANDs tokens within a
# single request, so combining a user query like "70b" with "finance" into
# one request returns nothing (no repo name has both tokens). Instead we
# issue one request per keyword and merge results — giving the domain flag
# OR semantics across synonyms and well-known model-family tokens.
DOMAIN_FILTERS: dict[str, list[str]] = {
    "finance": ["finance", "palmyra-fin", "plutus", "fingpt", "finma", "sujet-finance"],
    "medical": ["medical", "meditron", "clinical", "openbiollm", "palmyra-med"],
    "legal": ["legal", "saul", "law"],
    "code": ["coder", "code-llama", "deepseek-coder", "qwen-coder"],
    "math": ["math", "mathstral", "deepseek-math"],
    "science": ["scientific", "galactica", "sciglm"],
}


@dataclass(slots=True)
class HFQuant:
    """A single GGUF file within a repo."""

    quant: str              # "Q4_K_M", "IQ3_XXS", "F16", ...
    filename: str
    size_bytes: int = 0

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)


@dataclass(slots=True)
class HFModel:
    """A GGUF repository on Hugging Face."""

    repo_id: str            # "bartowski/Llama-3.2-3B-Instruct-GGUF"
    downloads: int = 0
    likes: int = 0
    tags: list[str] = field(default_factory=list)
    # Populated only if the caller explicitly requests file details.
    quants: list[HFQuant] = field(default_factory=list)

    @property
    def author(self) -> str:
        return self.repo_id.split("/", 1)[0] if "/" in self.repo_id else ""


def _auth_headers() -> dict[str, str]:
    """Return Authorization header if HF_TOKEN is set, else empty."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def search_models(
    query: str,
    *,
    domain: str | None = None,
    limit: int = 20,
    gguf_only: bool = True,
) -> list[HFModel]:
    """Search Hugging Face for models matching ``query``.

    Args:
        query: Free-text search string (e.g., "llama 70b", "plutus").
        domain: Optional domain filter ("finance", "medical", ...). Appends
            curated keywords to the query so the user does not have to
            remember tag names. See ``DOMAIN_FILTERS`` for the full list.
        limit: Maximum results to return. HF caps at 100.
        gguf_only: If True (default), restrict to repos tagged ``gguf``.

    Returns:
        List of ``HFModel`` ordered by downloads (descending). ``quants`` is
        empty — call ``list_quants`` for a specific repo to enumerate files.
    """
    user_q = query.strip() if query else ""

    # Build the list of searches to issue. Each becomes one HF API request.
    searches: list[str] = []
    if domain:
        keywords = DOMAIN_FILTERS.get(domain.lower())
        if keywords is None:
            raise ValueError(
                f"Unknown domain '{domain}'. Known: {sorted(DOMAIN_FILTERS)}"
            )
        # When both are given, searching the user query alone would surface
        # generic (non-domain) models. We search domain keywords and then
        # post-filter by the user query substring. When only domain is given,
        # issue one search per domain keyword.
        searches.extend(keywords)
    elif user_q:
        searches.append(user_q)
    else:
        # No query, no domain — fall back to "gguf" as a mild prefilter.
        searches.append("gguf")

    per_request_limit = max(1, min(limit * 3, 100)) if domain else max(1, min(limit, 100))
    seen: dict[str, HFModel] = {}
    first_match_rank: dict[str, int] = {}

    user_q_lower = user_q.lower()

    for idx, term in enumerate(searches):
        params: dict[str, Any] = {
            "search": term,
            "sort": "downloads",
            "direction": -1,
            "limit": per_request_limit,
        }
        if gguf_only:
            params["filter"] = "gguf"
        log.debug("HF search params: %s", params)
        try:
            r = requests.get(
                f"{HF_API}/models",
                params=params,
                headers=_auth_headers(),
                timeout=TIMEOUT_SEARCH,
            )
            r.raise_for_status()
            raw = r.json()
        except (requests.RequestException, ValueError) as e:
            log.error("HF search '%s' failed: %s", term, e)
            continue

        for item in raw if isinstance(raw, list) else []:
            repo_id = item.get("id") or item.get("modelId") or ""
            if not repo_id or repo_id in seen:
                continue
            # When the user supplied both a query and a domain, require the
            # repo id to contain the query substring — the domain search
            # gave us topical relevance, the query narrows it further.
            if domain and user_q_lower and user_q_lower not in repo_id.lower():
                continue
            seen[repo_id] = HFModel(
                repo_id=repo_id,
                downloads=int(item.get("downloads", 0) or 0),
                likes=int(item.get("likes", 0) or 0),
                tags=list(item.get("tags", []) or []),
            )
            first_match_rank[repo_id] = idx

        if len(seen) >= limit:
            break

    # Rank: earlier-matched search terms first (user query wins), then by downloads.
    ordered = sorted(
        seen.values(),
        key=lambda m: (first_match_rank.get(m.repo_id, 999), -m.downloads),
    )
    return ordered[:limit]


def list_quants(repo_id: str) -> list[HFQuant]:
    """Enumerate GGUF files in ``repo_id`` and parse quantization labels.

    Args:
        repo_id: ``org/repo`` identifier on Hugging Face.

    Returns:
        List of ``HFQuant``, one per ``.gguf`` file in the repo root. Sharded
        GGUFs (``*-00001-of-00003.gguf``) are collapsed to a single entry per
        quant whose ``size_bytes`` is the sum of all shards, since Ollama
        treats them as one model. Empty list on failure.
    """
    # The /tree/main endpoint returns file sizes (siblings in /api/models/<repo>
    # does not). Pagination isn't an issue for typical GGUF repos (<100 files).
    try:
        r = requests.get(
            f"{HF_API}/models/{repo_id}/tree/main",
            headers=_auth_headers(),
            timeout=TIMEOUT_FILES,
            params={"recursive": "true"},
        )
        r.raise_for_status()
        entries = r.json()
    except (requests.RequestException, ValueError) as e:
        log.error("HF tree fetch failed for %s: %s", repo_id, e)
        return []

    # Group shards by quant label; keep representative filename (first shard or whole file).
    by_quant: dict[str, HFQuant] = {}
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or entry.get("type") != "file":
            continue
        fname = entry.get("path", "")
        if not fname.lower().endswith(".gguf"):
            continue
        size = int(entry.get("size") or 0)
        quant = _parse_quant(fname)
        if quant is None:
            continue
        existing = by_quant.get(quant)
        if existing is None:
            by_quant[quant] = HFQuant(quant=quant, filename=fname, size_bytes=size)
        else:
            existing.size_bytes += size
            if "00001-of-" in fname and "00001-of-" not in existing.filename:
                existing.filename = fname

    return sorted(by_quant.values(), key=lambda q: q.size_bytes)


def _parse_quant(filename: str) -> str | None:
    """Extract a quantization label from a GGUF filename.

    Heuristic: take the last quant-shaped token in the filename, since model
    names can contain tokens like ``Q8`` in unrelated positions. Returns
    uppercase label, or None if no match.
    """
    matches = QUANT_PATTERN.findall(filename)
    if not matches:
        return None
    return matches[-1].upper()


def to_ollama_ref(repo_id: str, quant: str | None = None) -> str:
    """Translate an HF repo + quant into Ollama's pull syntax.

    >>> to_ollama_ref("bartowski/Llama-3.2-3B-Instruct-GGUF", "Q4_K_M")
    'hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M'
    """
    if "/" not in repo_id:
        raise ValueError(f"Expected 'org/repo' format, got: {repo_id}")
    base = f"hf.co/{repo_id}"
    return f"{base}:{quant}" if quant else base
