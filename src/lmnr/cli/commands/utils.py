"""Shared utilities for CLI command modules."""

from __future__ import annotations


def escape_sql_string(value: str) -> str:
    """Escape a string for safe use in a SQL single-quoted literal."""
    return value.replace("\\", "\\\\").replace("'", "\\'")
