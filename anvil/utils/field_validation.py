"""Field-level schema validator — declarative rules for OCR / form output.

Goes beyond JSON Schema type checks: validates the *content* of
extracted values against real-world formats (SSN, DOB, ZIP, currency,
email, phone, US state) and emits ``needs_review`` hints when values
look wrong but aren't obviously malformed.

Generalises FormOCR's ``extract/field_validators.py`` so any anvil
user can declare a field map and run it against an extracted record.

Usage::

    from anvil.utils.field_validation import validate_fields

    spec = {
        "ssn": {"type": "ssn"},
        "dob": {"type": "date", "format": "MM/DD/YYYY"},
        "zip": {"type": "zip_us"},
        "premium": {"type": "currency", "min": 0, "max": 10_000},
        "state": {"type": "state_us"},
        "notes": {"type": "text", "max_len": 500, "required": False},
    }
    report = validate_fields(spec, record)
    if report.needs_review:
        ...

The report carries per-field ``FieldResult`` entries with ``ok``,
``message``, ``normalized_value`` (what the validator would use
going forward), and ``severity`` (``error`` | ``warning`` | ``info``).

Built-in field types (extensible via ``register_validator``):

- ``text``       — string, optional min_len/max_len/pattern.
- ``integer``    — int, optional min/max.
- ``number``     — int or float, optional min/max.
- ``ssn``        — ###-##-#### or 9 digits (auto-normalized to hyphenated form).
- ``date``       — configurable format (MM/DD/YYYY, YYYY-MM-DD, MM/DD/YY).
- ``zip_us``     — 5 or 5-4 digits.
- ``phone_us``   — 10 digits in any common format → normalized to (###) ###-####.
- ``email``      — RFC-5322-ish.
- ``currency``   — accepts "$1,234.56" / "1234.56" / "1,234" → float; optional min/max.
- ``state_us``   — 2-letter postal code (auto-uppercased).
- ``choice``     — value must be in ``choices`` list.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "FieldResult",
    "ValidationReport",
    "validate_fields",
    "validate_field",
    "register_validator",
    "BUILT_IN_VALIDATORS",
    "US_STATES",
]

# Two-letter US postal codes.
US_STATES: frozenset[str] = frozenset({
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC", "PR", "VI", "GU", "AS", "MP",
})


@dataclass(slots=True)
class FieldResult:
    """Outcome of validating a single field."""

    field_name: str
    ok: bool
    severity: str = "info"              # 'error' | 'warning' | 'info'
    message: str = ""
    normalized_value: Any = None
    raw_value: Any = None

    @property
    def needs_review(self) -> bool:
        return not self.ok or self.severity == "warning"


@dataclass(slots=True)
class ValidationReport:
    """Aggregate result across a whole record."""

    fields: list[FieldResult] = field(default_factory=list)

    def by_field(self) -> dict[str, FieldResult]:
        return {r.field_name: r for r in self.fields}

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.fields)

    @property
    def needs_review(self) -> bool:
        return any(r.needs_review for r in self.fields)

    @property
    def errors(self) -> list[FieldResult]:
        return [r for r in self.fields if r.severity == "error"]

    @property
    def warnings(self) -> list[FieldResult]:
        return [r for r in self.fields if r.severity == "warning"]


# A validator is a callable that takes (value, spec) and returns
# (ok, severity, message, normalized_value).
Validator = Callable[[Any, dict[str, Any]], tuple[bool, str, str, Any]]


def validate_field(
    field_name: str,
    value: Any,
    spec: dict[str, Any],
) -> FieldResult:
    """Validate one field against its spec."""
    required = spec.get("required", True)
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            return FieldResult(
                field_name=field_name, ok=False, severity="error",
                message="required field is missing",
                normalized_value=None, raw_value=value,
            )
        return FieldResult(
            field_name=field_name, ok=True, severity="info",
            message="optional field absent",
            normalized_value=None, raw_value=value,
        )

    type_name = str(spec.get("type", "text")).lower()
    validator = BUILT_IN_VALIDATORS.get(type_name)
    if validator is None:
        return FieldResult(
            field_name=field_name, ok=False, severity="error",
            message=f"unknown type: {type_name!r}",
            normalized_value=None, raw_value=value,
        )
    ok, severity, message, normalized = validator(value, spec)
    return FieldResult(
        field_name=field_name, ok=ok, severity=severity,
        message=message, normalized_value=normalized, raw_value=value,
    )


def validate_fields(
    spec: dict[str, dict[str, Any]],
    record: dict[str, Any],
) -> ValidationReport:
    """Validate every field declared in ``spec``."""
    report = ValidationReport()
    for name, field_spec in spec.items():
        result = validate_field(name, record.get(name), field_spec)
        report.fields.append(result)
    return report


def register_validator(type_name: str, fn: Validator) -> None:
    """Register (or override) a validator for a given type name."""
    BUILT_IN_VALIDATORS[type_name.lower()] = fn


# ─── Built-in validators ────────────────────────────────────────────

def _v_text(value: Any, spec: dict[str, Any]) -> tuple[bool, str, str, Any]:
    if not isinstance(value, str):
        return False, "error", "expected a string", value
    s = value.strip()
    if "min_len" in spec and len(s) < int(spec["min_len"]):
        return False, "error", f"shorter than min_len={spec['min_len']}", s
    if "max_len" in spec and len(s) > int(spec["max_len"]):
        return False, "error", f"longer than max_len={spec['max_len']}", s
    pattern = spec.get("pattern")
    if pattern:
        try:
            if not re.fullmatch(pattern, s):
                return False, "error", f"does not match pattern {pattern!r}", s
        except re.error as e:
            return False, "error", f"invalid pattern: {e}", s
    return True, "info", "ok", s


def _coerce_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None  # explicitly reject bools — they are ints in Python
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip().replace(",", "").replace("$", "")
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _v_integer(value: Any, spec: dict[str, Any]) -> tuple[bool, str, str, Any]:
    n = _coerce_number(value)
    if n is None or n != int(n):
        return False, "error", "expected an integer", value
    i = int(n)
    if "min" in spec and i < spec["min"]:
        return False, "error", f"less than min={spec['min']}", i
    if "max" in spec and i > spec["max"]:
        return False, "error", f"greater than max={spec['max']}", i
    return True, "info", "ok", i


def _v_number(value: Any, spec: dict[str, Any]) -> tuple[bool, str, str, Any]:
    n = _coerce_number(value)
    if n is None:
        return False, "error", "expected a number", value
    if "min" in spec and n < spec["min"]:
        return False, "error", f"less than min={spec['min']}", n
    if "max" in spec and n > spec["max"]:
        return False, "error", f"greater than max={spec['max']}", n
    return True, "info", "ok", n


_SSN_RE = re.compile(r"^\d{3}-?\d{2}-?\d{4}$")


def _v_ssn(value: Any, _spec: dict[str, Any]) -> tuple[bool, str, str, Any]:
    if not isinstance(value, str):
        return False, "error", "expected a string SSN", value
    s = value.strip()
    digits = s.replace("-", "").replace(" ", "")
    if not digits.isdigit() or len(digits) != 9:
        return False, "error", "SSN must be 9 digits (optionally hyphenated)", s
    normalized = f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"
    # Known-invalid area numbers (first three digits).
    if digits[:3] == "000" or digits[:3] == "666" or digits[:3].startswith("9"):
        return False, "error", "invalid SSN area number", normalized
    if digits[3:5] == "00":
        return False, "error", "invalid SSN group number (00)", normalized
    if digits[5:] == "0000":
        return False, "error", "invalid SSN serial number (0000)", normalized
    return True, "info", "ok", normalized


_DATE_FORMATS = {
    "MM/DD/YYYY": re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$"),
    "MM/DD/YY":   re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{2})$"),
    "YYYY-MM-DD": re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$"),
    "DD/MM/YYYY": re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$"),
}


def _v_date(value: Any, spec: dict[str, Any]) -> tuple[bool, str, str, Any]:
    if not isinstance(value, str):
        return False, "error", "expected a date string", value
    fmt = str(spec.get("format", "MM/DD/YYYY")).upper()
    regex = _DATE_FORMATS.get(fmt)
    if regex is None:
        return False, "error", f"unsupported date format {fmt!r}", value
    s = value.strip()
    m = regex.fullmatch(s)
    if not m:
        return False, "error", f"expected format {fmt}", s
    parts = m.groups()
    if fmt == "YYYY-MM-DD":
        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
    elif fmt == "DD/MM/YYYY":
        day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
    elif fmt == "MM/DD/YY":
        month, day, yy = int(parts[0]), int(parts[1]), int(parts[2])
        year = 2000 + yy if yy < 50 else 1900 + yy
    else:
        month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
    if not (1 <= month <= 12):
        return False, "error", f"month {month} out of range", s
    if not (1 <= day <= 31):
        return False, "error", f"day {day} out of range", s
    # Cheap day-of-month bound per month. Good enough for review flagging;
    # calendar-perfect checking is left to callers who care.
    max_day = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
    if day > max_day:
        return False, "warning", f"day {day} too high for month {month}", s
    return True, "info", "ok", f"{year:04d}-{month:02d}-{day:02d}"


_ZIP_RE = re.compile(r"^\d{5}(?:-\d{4})?$")


def _v_zip_us(value: Any, _spec: dict[str, Any]) -> tuple[bool, str, str, Any]:
    if not isinstance(value, str):
        return False, "error", "expected a ZIP string", value
    s = value.strip()
    if not _ZIP_RE.fullmatch(s):
        return False, "error", "expected #####[-####]", s
    return True, "info", "ok", s


def _v_phone_us(value: Any, _spec: dict[str, Any]) -> tuple[bool, str, str, Any]:
    if not isinstance(value, str):
        return False, "error", "expected a phone string", value
    digits = re.sub(r"\D", "", value)
    if digits.startswith("1") and len(digits) == 11:
        digits = digits[1:]
    if len(digits) != 10:
        return False, "error", "phone must normalize to 10 digits", value.strip()
    normalized = f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    return True, "info", "ok", normalized


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _v_email(value: Any, _spec: dict[str, Any]) -> tuple[bool, str, str, Any]:
    if not isinstance(value, str):
        return False, "error", "expected an email string", value
    s = value.strip()
    if not _EMAIL_RE.fullmatch(s):
        return False, "error", "not a valid email", s
    return True, "info", "ok", s.lower()


def _v_currency(value: Any, spec: dict[str, Any]) -> tuple[bool, str, str, Any]:
    n = _coerce_number(value)
    if n is None:
        return False, "error", "expected a currency amount", value
    if "min" in spec and n < spec["min"]:
        return False, "error", f"less than min={spec['min']}", n
    if "max" in spec and n > spec["max"]:
        return False, "error", f"greater than max={spec['max']}", n
    return True, "info", "ok", round(n, 2)


def _v_state_us(value: Any, _spec: dict[str, Any]) -> tuple[bool, str, str, Any]:
    if not isinstance(value, str):
        return False, "error", "expected a US state code string", value
    s = value.strip().upper()
    if s not in US_STATES:
        return False, "error", f"{s!r} is not a US state postal code", s
    return True, "info", "ok", s


def _v_choice(value: Any, spec: dict[str, Any]) -> tuple[bool, str, str, Any]:
    choices = spec.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return False, "error", "choice type requires non-empty 'choices' list", value
    # Case-insensitive match for strings.
    if isinstance(value, str):
        lowered = {str(c).lower(): c for c in choices}
        match = lowered.get(value.strip().lower())
        if match is not None:
            return True, "info", "ok", match
    elif value in choices:
        return True, "info", "ok", value
    return False, "error", f"value {value!r} not in {choices}", value


BUILT_IN_VALIDATORS: dict[str, Validator] = {
    "text":     _v_text,
    "integer":  _v_integer,
    "number":   _v_number,
    "ssn":      _v_ssn,
    "date":     _v_date,
    "zip_us":   _v_zip_us,
    "phone_us": _v_phone_us,
    "email":    _v_email,
    "currency": _v_currency,
    "state_us": _v_state_us,
    "choice":   _v_choice,
}
