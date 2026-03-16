"""Smart output formatter with auto-detection for terminal vs pipe.

Detects the output consumer (human terminal, piped process, AI agent) and
adapts formatting accordingly. Supports TABLE, JSON, COMPACT, and JSONL modes.

Detection priority:
    1. Explicit --json / --compact / --jsonl flag (highest)
    2. LMNR_OUTPUT_FORMAT env var
    3. Auto-detect: if stdout.isatty() -> TABLE, else -> JSON
"""

import os
import sys
from enum import Enum
from typing import Any, Optional

import orjson


class OutputMode(Enum):
    TABLE = "table"
    JSON = "json"
    COMPACT = "compact"
    JSONL = "jsonl"


# Default compact key abbreviations for token optimization
DEFAULT_COMPACT_KEYS: dict[str, str] = {
    "trace_id": "tid",
    "span_id": "sid",
    "start_time": "t0",
    "end_time": "t1",
    "duration": "dur",
    "total_cost": "cost",
    "total_tokens": "tok",
    "input_tokens": "itok",
    "output_tokens": "otok",
    "top_span_name": "name",
    "status": "st",
    "session_id": "sess",
    "evaluation_id": "eid",
    "created_at": "ts",
    "last_updated": "updated",
    "datapoints": "n",
    "user_id": "uid",
    "trace_type": "type",
    "span_type": "stype",
    "parent_span_id": "parent",
}

# ANSI color codes
_ANSI = {
    "bold": "\033[1m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "cyan": "\033[36m",
    "dim": "\033[2m",
    "reset": "\033[0m",
}


def _detect_mode() -> OutputMode:
    """Auto-detect output mode based on environment."""
    env_mode = os.environ.get("LMNR_OUTPUT_FORMAT", "").lower()
    if env_mode in ("json", "compact", "jsonl", "table"):
        return OutputMode(env_mode)
    if sys.stdout.isatty():
        return OutputMode.TABLE
    return OutputMode.JSON


def _should_disable_color() -> bool:
    """Check if color should be disabled."""
    if os.environ.get("NO_COLOR"):
        return True
    if not sys.stdout.isatty():
        return True
    return False


class OutputFormatter:
    """Smart output formatter with auto-detection.

    Formats data as table, JSON, compact JSON, or JSONL depending on
    the detected or explicitly requested output mode.
    """

    def __init__(
        self,
        mode: Optional[OutputMode] = None,
        no_color: bool = False,
        jq_filter: Optional[str] = None,
        truncate_at: int = 120,
        compact_keys: Optional[dict[str, str]] = None,
    ):
        self.mode = mode or _detect_mode()
        self.no_color = no_color or _should_disable_color()
        self.jq_filter = jq_filter
        self.truncate_at = truncate_at
        self.compact_keys = compact_keys or DEFAULT_COMPACT_KEYS

    def format(self, data: Any, columns: Optional[list[str]] = None) -> str:
        """Format data according to current mode. Returns string to print."""
        if self.jq_filter:
            data = self._apply_jq(data)

        if self.mode == OutputMode.TABLE:
            return self._format_table(data, columns)
        elif self.mode == OutputMode.JSON:
            return self._format_json(data)
        elif self.mode == OutputMode.COMPACT:
            return self._format_compact(data)
        elif self.mode == OutputMode.JSONL:
            return self._format_jsonl(data)
        return str(data)

    def output(self, data: Any, columns: Optional[list[str]] = None) -> None:
        """Format and print data to stdout."""
        print(self.format(data, columns))

    # --- Table formatting ---

    def _format_table(self, data: Any, columns: Optional[list[str]]) -> str:
        """Render data as an aligned, optionally colored table."""
        if not data:
            return "(no results)"
        if isinstance(data, dict):
            return self._format_kv_table(data)
        if isinstance(data, list) and len(data) > 0:
            return self._format_list_table(data, columns)
        return str(data)

    def _format_list_table(
        self, rows: list[dict], columns: Optional[list[str]] = None
    ) -> str:
        """Format a list of dicts as a table with column headers."""
        if not rows:
            return "(no results)"
        if columns is None:
            columns = list(rows[0].keys())

        # Calculate column widths
        widths: dict[str, int] = {}
        for col in columns:
            values = [self._cell_value(row.get(col, "")) for row in rows]
            widths[col] = max(
                len(col), max((len(v) for v in values), default=0)
            )
            widths[col] = min(widths[col], self.truncate_at)

        # Header
        header = "  ".join(
            self._colorize(col.upper().ljust(widths[col]), "bold")
            for col in columns
        )
        separator = "  ".join("-" * widths[col] for col in columns)

        # Rows
        lines = [header, separator]
        for row in rows:
            line = "  ".join(
                self._cell_value(row.get(col, "")).ljust(widths[col])[
                    : widths[col]
                ]
                for col in columns
            )
            if row.get("status") == "error":
                line = self._colorize(line, "red")
            lines.append(line)

        return "\n".join(lines)

    def _format_kv_table(self, obj: dict) -> str:
        """Format a single dict as key-value pairs."""
        if not obj:
            return "(no results)"
        max_key_len = max(len(str(k)) for k in obj.keys())
        lines = []
        for k, v in obj.items():
            val_str = self._cell_value(v)
            lines.append(
                f"  {self._colorize(str(k).ljust(max_key_len), 'bold')}  {val_str}"
            )
        return "\n".join(lines)

    # --- JSON formatting ---

    def _format_json(self, data: Any) -> str:
        """Pretty-printed JSON using orjson."""
        return orjson.dumps(
            data,
            option=orjson.OPT_INDENT_2
            | orjson.OPT_NON_STR_KEYS
            | orjson.OPT_SERIALIZE_UUID,
            default=str,
        ).decode("utf-8")

    # --- Compact formatting ---

    def _format_compact(self, data: Any) -> str:
        """Token-optimized compact output for AI agent consumption."""
        compacted = self._compact_data(data)
        return orjson.dumps(
            compacted,
            option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_UUID,
            default=str,
        ).decode("utf-8")

    # --- JSONL formatting ---

    def _format_jsonl(self, data: Any) -> str:
        """One JSON object per line."""
        if isinstance(data, list):
            return "\n".join(
                orjson.dumps(
                    item,
                    option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_UUID,
                    default=str,
                ).decode("utf-8")
                for item in data
            )
        return orjson.dumps(
            data,
            option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_UUID,
            default=str,
        ).decode("utf-8")

    # --- Helpers ---

    def _compact_data(self, data: Any) -> Any:
        """Recursively abbreviate keys and truncate values."""
        if isinstance(data, list):
            return [self._compact_data(item) for item in data]
        if isinstance(data, dict):
            result = {}
            for k, v in data.items():
                new_key = self.compact_keys.get(k, k)
                new_val = self._compact_data(v)
                if isinstance(new_val, str) and len(new_val) > 200:
                    new_val = new_val[:197] + "..."
                result[new_key] = new_val
            return result
        return data

    def _cell_value(self, val: Any) -> str:
        """Convert a value to a display string for table mode."""
        if val is None:
            return "-"
        if isinstance(val, float):
            if val == 0.0:
                return "0"
            if abs(val) < 0.01:
                return f"{val:.6f}"
            return f"{val:.4f}"
        if isinstance(val, list):
            return ", ".join(str(v) for v in val[:5])
        if isinstance(val, dict):
            compact = orjson.dumps(val, default=str).decode("utf-8")
            return compact[: self.truncate_at]
        return str(val)

    def _apply_jq(self, data: Any) -> Any:
        """Apply jmespath filter to data."""
        try:
            import jmespath
        except ImportError:
            print(
                "Error: jmespath is required for --jq filtering.\n"
                "Install it with: pip install 'lmnr[cli]'",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            result = jmespath.search(self.jq_filter, data)
            return result
        except jmespath.exceptions.ParseError as e:
            print(f"Error: Invalid JMESPath expression: {e}", file=sys.stderr)
            sys.exit(1)

    def _colorize(self, text: str, style: str) -> str:
        """Apply ANSI color codes if color is enabled."""
        if self.no_color:
            return text
        code = _ANSI.get(style, "")
        if not code:
            return text
        return f"{code}{text}{_ANSI['reset']}"
