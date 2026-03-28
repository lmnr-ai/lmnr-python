"""Shared output formatting for the Laminar CLI."""

import sys
import uuid as uuid_mod
from datetime import datetime, timedelta, timezone
from typing import Any

import orjson


def format_json(data: Any) -> str:
    """Format data as pretty-printed JSON string."""
    return orjson.dumps(
        data,
        option=orjson.OPT_INDENT_2
        | orjson.OPT_NON_STR_KEYS
        | orjson.OPT_SERIALIZE_UUID
        | orjson.OPT_UTC_Z,
    ).decode()


def print_json(data: Any) -> None:
    """Print data as JSON to stdout."""
    print(format_json(data))


def format_table(
    rows: list[dict[str, Any]],
    columns: list[tuple[str, str, int]],
) -> str:
    """Format rows as an aligned text table.

    Args:
        rows: List of dictionaries containing row data.
        columns: List of (key, header, width) tuples defining the table layout.

    Returns:
        Formatted table string.
    """
    if not rows:
        return ""

    lines = []

    # Header
    header_parts = []
    separator_parts = []
    for key, header, width in columns:
        header_parts.append(f"{header:<{width}}")
        separator_parts.append("-" * width)
    lines.append("  ".join(header_parts))
    lines.append("  ".join(separator_parts))

    # Rows
    for row in rows:
        row_parts = []
        for key, header, width in columns:
            value = row.get(key, "")
            if value is None:
                value = "-"
            value_str = str(value)
            if len(value_str) > width:
                value_str = value_str[: width - 3] + "..."
            row_parts.append(f"{value_str:<{width}}")
        lines.append("  ".join(row_parts))

    return "\n".join(lines)


def print_table(
    rows: list[dict[str, Any]],
    columns: list[tuple[str, str, int]],
) -> None:
    """Print a formatted table to stdout."""
    table = format_table(rows, columns)
    if table:
        print(table)


def format_cost(value: float | None) -> str:
    """Format a cost value as a dollar string."""
    if value is None or value == 0:
        return "$0.0000"
    return f"${value:.4f}"


def format_tokens(value: int | None) -> str:
    """Format token count with thousands separators."""
    if value is None:
        return "0"
    return f"{value:,}"


def format_duration_seconds(start_time: str | None, end_time: str | None) -> str:
    """Calculate and format duration between two ISO timestamps."""
    if not start_time or not end_time:
        return "-"
    try:
        start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
        duration = (end - start).total_seconds()
        if duration < 0:
            return "-"
        if duration < 1:
            return f"{duration * 1000:.0f}ms"
        elif duration < 60:
            return f"{duration:.3f}s"
        else:
            minutes = int(duration // 60)
            seconds = duration % 60
            return f"{minutes}m{seconds:.1f}s"
    except (ValueError, TypeError):
        return "-"


def truncate_id(uuid_str: str, length: int = 12) -> str:
    """Truncate a UUID string for display, keeping the first `length` chars."""
    if not uuid_str:
        return "-"
    if len(uuid_str) <= length:
        return uuid_str
    return uuid_str[:length] + "..."


def print_error(message: str, hint: str | None = None) -> None:
    """Print an error message to stderr with optional hint."""
    print(f"Error: {message}", file=sys.stderr)
    if hint:
        print(f"Hint: {hint}", file=sys.stderr)


def validate_uuid(value: str, name: str = "ID") -> str:
    """Validate that a string is a valid UUID. Returns the normalized UUID string."""
    try:
        return str(uuid_mod.UUID(value))
    except ValueError:
        print_error(
            f"Invalid {name} format: '{value}'.",
            f"Expected a UUID like 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'.",
        )
        sys.exit(1)


def resolve_time_filters(args: Any) -> tuple[str | None, str | None]:
    """Resolve time filters from args, returning (start_time, end_time) strings.

    Handles --past-hours as an alternative to --start-time/--end-time.
    """
    if hasattr(args, "past_hours") and args.past_hours is not None:
        start = datetime.now(timezone.utc) - timedelta(hours=args.past_hours)
        return (start.strftime("%Y-%m-%dT%H:%M:%SZ"), None)

    start_time = getattr(args, "start_time", None)
    end_time = getattr(args, "end_time", None)
    return (start_time, end_time)
