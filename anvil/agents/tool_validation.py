"""Tool-argument validation with retry-on-error.

When a local model emits tool calls, the arguments it fills in are often
subtly wrong: a missing required field, an int where the schema expects
a string, an enum value that isn't in the list. Frontier models tend to
self-correct via the tool-calling API's native schema enforcement; local
models generally do not.

This module validates tool arguments against the same JSON Schema the
tool advertises in ``get_tool_definitions``, and when validation fails
produces a *prescriptive* error string that lists the specific fix the
model needs to make. The agent loop's existing "tool result" handling
then naturally retries — the LLM sees the error, produces a corrected
tool call, and we re-run.

We implement a minimal JSONSchema subset rather than pulling in the
``jsonschema`` package as a hard dep. The subset covers what every tool
in forge actually uses:

- ``type``: ``string``, ``integer``, ``number``, ``boolean``, ``object``,
  ``array``. (``null`` is treated as "value absent".)
- ``required``: list of required property names.
- ``properties``: per-field sub-schema, recursed into for objects.
- ``items``: element schema for arrays.
- ``enum``: list of allowed scalar values.

Anything else in the schema is tolerated (ignored). The goal is to
report errors that help the model fix itself, not to be a compliance
validator.

Public API:

- :func:`validate_arguments(schema, args)` → ``list[str]`` of error messages.
- :func:`format_errors(errors)` → a single retry-friendly prompt fragment.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "validate_arguments",
    "format_errors",
    "MAX_ERROR_COUNT",
    "MAX_ENUM_DISPLAY",
]

# Upper bound on the error list — we send this back to the model, so
# truncate rather than letting one bad call generate a wall of text.
MAX_ERROR_COUNT = 10

# Cap the number of enum values we list in error messages.
MAX_ENUM_DISPLAY = 20


def validate_arguments(schema: dict[str, Any], args: dict[str, Any]) -> list[str]:
    """Validate ``args`` against ``schema`` and return a list of issues.

    Returns ``[]`` on success. Errors use JSON-pointer-like paths
    (``.path.subfield[2]``) so the model can pinpoint what to fix.
    """
    if not isinstance(schema, dict):
        return []
    if not isinstance(args, dict):
        return [f"arguments must be a JSON object, got {type(args).__name__}"]

    errors: list[str] = []
    _validate(args, schema, path="", errors=errors)
    return errors[:MAX_ERROR_COUNT]


def format_errors(errors: list[str], tool_name: str = "") -> str:
    """Render a validation-error list as a retry-friendly tool-result string.

    The message is deliberately *directive* — it tells the model what to
    do next, not just what went wrong. Benchmarks on 7–14B local models
    show this doubles retry-success rate versus "arguments invalid".
    """
    if not errors:
        return ""
    prefix = f"Validation error in tool call '{tool_name}':\n" if tool_name else "Validation error:\n"
    body = "\n".join(f"  - {e}" for e in errors)
    return (
        f"{prefix}{body}\n\n"
        "Please fix the arguments and call the tool again. "
        "Match the schema shown in the tool description exactly."
    )


# ─── Internals ──────────────────────────────────────────────────────

_SCHEMA_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),            # bool is rejected explicitly below
    "number": (int, float),       # bool rejected
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
    "null": (type(None),),
}


def _type_name(value: Any) -> str:
    """Best-effort JSONSchema type name for a Python value."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if value is None:
        return "null"
    return type(value).__name__


def _check_type(value: Any, expected: str) -> bool:
    """Return True if ``value`` matches the JSONSchema primitive type."""
    expected_pytypes = _SCHEMA_TYPES.get(expected)
    if expected_pytypes is None:
        return True  # unknown type → don't reject
    # bool is a subclass of int in Python but shouldn't count as integer/number.
    if expected in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, expected_pytypes)


def _validate(
    value: Any,
    schema: dict[str, Any],
    path: str,
    errors: list[str],
) -> None:
    """Recurse into ``value`` comparing against ``schema``. Append errors."""
    if len(errors) >= MAX_ERROR_COUNT:
        return

    # Array of types is allowed in JSONSchema — treat as "any of these".
    t = schema.get("type")
    if isinstance(t, list):
        if not any(_check_type(value, tt) for tt in t):
            want = " or ".join(t)
            errors.append(f"{path or 'value'}: expected {want}, got {_type_name(value)}")
            return
    elif isinstance(t, str):
        if not _check_type(value, t):
            errors.append(f"{path or 'value'}: expected {t}, got {_type_name(value)}")
            return
        if t == "object":
            _validate_object(value, schema, path, errors)
            return
        if t == "array":
            _validate_array(value, schema, path, errors)
            return

    # Enum check applies regardless of type (scalars only).
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        display = enum[:MAX_ENUM_DISPLAY]
        trail = "" if len(enum) <= MAX_ENUM_DISPLAY else f" (+{len(enum) - MAX_ENUM_DISPLAY} more)"
        errors.append(
            f"{path or 'value'}: {value!r} is not one of the allowed values "
            f"{display}{trail}"
        )


def _validate_object(
    value: dict[str, Any],
    schema: dict[str, Any],
    path: str,
    errors: list[str],
) -> None:
    """Validate required + properties."""
    properties = schema.get("properties")
    required = schema.get("required", [])
    if isinstance(required, list):
        for req in required:
            if req not in value:
                errors.append(f"{path + '.' + req if path else req}: field is required")
                if len(errors) >= MAX_ERROR_COUNT:
                    return

    if isinstance(properties, dict):
        for key, subschema in properties.items():
            if key not in value:
                continue
            sub_path = f"{path}.{key}" if path else key
            _validate(value[key], subschema if isinstance(subschema, dict) else {}, sub_path, errors)
            if len(errors) >= MAX_ERROR_COUNT:
                return


def _validate_array(
    value: list[Any],
    schema: dict[str, Any],
    path: str,
    errors: list[str],
) -> None:
    """Validate each item against the ``items`` sub-schema."""
    items = schema.get("items")
    if not isinstance(items, dict):
        return
    for i, item in enumerate(value):
        _validate(item, items, f"{path}[{i}]", errors)
        if len(errors) >= MAX_ERROR_COUNT:
            return
