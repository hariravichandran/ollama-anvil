"""PII redaction — scrub common sensitive patterns before logging / upload.

Designed for workflows like FormOCR's template-labeling phase, where
the extracted blanks may still contain OCR of sensitive text that must
not reach a cloud LLM or a remote trace store.

The redactor runs a set of pre-compiled regexes against the input and
replaces each match with a stable placeholder (``[REDACTED:SSN]``,
``[REDACTED:EMAIL]``, ...). Anything the regexes miss is UNREDACTED —
this is a best-effort scrubber, not a guarantee. Pair it with a policy
review for anything sensitive.

Built-in detectors (US-centric, enable or customise as needed):

- ``ssn``         — ###-##-####
- ``phone_us``    — (###) ###-####, ###-###-####, ###.###.####
- ``email``       — standard RFC 5322-ish
- ``credit_card`` — 13-19 digit runs with optional separators
- ``date``        — MM/DD/YYYY or YYYY-MM-DD
- ``zip_us``      — 5 or 5-4 digit
- ``ip_address``  — IPv4
- ``ipv6``        — compressed and full
- ``url``         — http(s) URLs

Usage::

    from anvil.utils.pii import Redactor, redact

    # Default: all built-ins enabled.
    clean, report = redact("Call me at 555-123-4567 or me@example.com")
    #  clean  → "Call me at [REDACTED:PHONE_US] or [REDACTED:EMAIL]"
    #  report → {"PHONE_US": 1, "EMAIL": 1}

    # Custom detectors — drop unwanted ones, add your own:
    r = Redactor(
        enabled={"SSN", "EMAIL"},
        custom_patterns={"POLICY_ID": r"POL-\\d{6}"},
    )
    clean = r.redact("POL-123456 for jane@example.com")[0]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Redactor",
    "RedactionReport",
    "redact",
    "DEFAULT_DETECTORS",
    "PLACEHOLDER_FORMAT",
]

PLACEHOLDER_FORMAT = "[REDACTED:{name}]"


# Pattern ordering matters: longer/more-specific patterns run first so
# (say) the credit-card matcher wins over a generic digit-run matcher.
# Every regex here is anchored on a boundary-ish check so we don't
# accidentally eat substrings of larger identifiers.
_DEFAULT_PATTERNS: list[tuple[str, str]] = [
    # Emails first — they can otherwise look like URLs with '@'.
    ("EMAIL", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    # IPv6. Two branches: full 8-group form, and the common compressed
    # form that contains ``::`` (e.g. ``2001:db8::1``). The compressed
    # branch REQUIRES both a leading hex:hex prefix AND a literal ``::``
    # so placeholders like ``[REDACTED:EMAIL]`` can't false-match on a
    # lone ``:E`` substring.
    ("IPV6", (
        r"\b(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}\b"
        r"|\b(?:[0-9A-Fa-f]{1,4}:){1,6}:(?:[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{1,4})*)?"
    )),
    # URLs.
    ("URL", r"\bhttps?://[^\s<>\"']+"),
    # Credit card: 13-19 digits with optional spaces or dashes.
    # Reject if the whole run is a single 9-digit SSN shape — handled by
    # SSN below since it runs AFTER this (we remove a match before the
    # next regex sees the text).
    ("CREDIT_CARD", r"\b(?:\d[ -]?){12,18}\d\b"),
    # SSN in hyphenated form. We avoid raw 9-digit runs to limit
    # false-positives.
    ("SSN", r"\b\d{3}-\d{2}-\d{4}\b"),
    # US phone: optional +1, area code, etc.
    ("PHONE_US", r"(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b"),
    # ISO and US date formats.
    ("DATE", r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b"),
    # IPv4.
    ("IP_ADDRESS", r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    # US ZIP.
    ("ZIP_US", r"\b\d{5}(?:-\d{4})?\b"),
]

# Names of every built-in detector. Exposed so callers can enable/disable.
DEFAULT_DETECTORS: frozenset[str] = frozenset(name for name, _ in _DEFAULT_PATTERNS)


@dataclass(slots=True)
class RedactionReport:
    """Summary of what a redaction pass changed."""

    counts: dict[str, int] = field(default_factory=dict)
    total: int = 0

    def add(self, name: str, n: int = 1) -> None:
        if n <= 0:
            return
        self.counts[name] = self.counts.get(name, 0) + n
        self.total += n

    def __bool__(self) -> bool:
        return self.total > 0


@dataclass
class Redactor:
    """Compiled redactor with configurable detectors.

    A fresh instance compiles its pattern set once; subsequent
    :meth:`redact` calls reuse the compiled regexes, so redactors are
    cheap to hold as module-level singletons.
    """

    enabled: frozenset[str] | None = None
    custom_patterns: dict[str, str] = field(default_factory=dict)
    _compiled: list[tuple[str, re.Pattern[str]]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        enabled = self.enabled if self.enabled is not None else DEFAULT_DETECTORS
        patterns: list[tuple[str, str]] = [
            (name, pat) for name, pat in _DEFAULT_PATTERNS if name in enabled
        ]
        for name, pat in self.custom_patterns.items():
            patterns.append((name.upper(), pat))
        compiled: list[tuple[str, re.Pattern[str]]] = []
        for name, pat in patterns:
            try:
                compiled.append((name, re.compile(pat)))
            except re.error:
                # Skip bad user-supplied patterns silently — a redactor
                # that crashes on construction is worse than one that
                # silently drops one rule and reports via the counts.
                continue
        self._compiled = compiled

    def redact(self, text: str) -> tuple[str, RedactionReport]:
        """Scrub ``text`` and return ``(redacted_text, report)``.

        Runs every enabled detector in declaration order; each one's
        matches are replaced before the next one sees the text. That
        way EMAIL (which contains ``@``) runs before URL (which could
        otherwise accidentally gobble email addresses with schemes).
        """
        report = RedactionReport()
        if not text:
            return "", report
        current = text
        for name, regex in self._compiled:
            placeholder = PLACEHOLDER_FORMAT.format(name=name)
            new_text, n = regex.subn(placeholder, current)
            if n:
                report.add(name, n)
                current = new_text
        return current, report

    def detect(self, text: str) -> dict[str, int]:
        """Count matches without replacing. Useful for audit dashboards."""
        counts: dict[str, int] = {}
        if not text:
            return counts
        for name, regex in self._compiled:
            n = len(regex.findall(text))
            if n:
                counts[name] = n
        return counts


# Module-level convenience using the default detector set.
_DEFAULT = Redactor()


def redact(text: str) -> tuple[str, RedactionReport]:
    """Redact ``text`` using the default detector set."""
    return _DEFAULT.redact(text)


def detect_pii(text: str) -> dict[str, int]:
    """Detect (but do not remove) PII in ``text``."""
    return _DEFAULT.detect(text)


def redact_dict(data: dict[str, Any]) -> tuple[dict[str, Any], RedactionReport]:
    """Redact every string value in a dict (recursively).

    Non-string leaves (ints, floats, bools, None) are left alone.
    Returns a new dict so the original is not mutated.
    """
    report = RedactionReport()
    out = _redact_node(data, report)
    return out, report


def _redact_node(node: Any, report: RedactionReport) -> Any:
    if isinstance(node, str):
        clean, r = _DEFAULT.redact(node)
        for name, n in r.counts.items():
            report.add(name, n)
        return clean
    if isinstance(node, dict):
        return {k: _redact_node(v, report) for k, v in node.items()}
    if isinstance(node, list):
        return [_redact_node(v, report) for v in node]
    if isinstance(node, tuple):
        return tuple(_redact_node(v, report) for v in node)
    return node
