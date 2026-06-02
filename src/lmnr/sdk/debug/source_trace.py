"""Fetch a source trace's spans via the existing SQL/query endpoint.

Two-phase by default (§9, §E): first pass pulls lightweight spine-detection
fields (path/type/times), then a second pass pulls response payloads only for
the chosen spine path. This keeps memory bounded since v1 only caches the spine.
"""

import datetime
import json
import re
from typing import Any

from lmnr.sdk.client.synchronous.sync_client import LaminarClient
from lmnr.sdk.debug.spine import SpanRecord
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

_PAGE_SIZE = 1000


def _to_epoch(value: Any, missing_default: float = 0.0) -> float:
    """Best-effort conversion of a ClickHouse timestamp to a float epoch.

    The SQL endpoint returns ISO-8601 strings; fall back to a chronological
    lexical encoding so detection still orders correctly even if parsing fails.
    `missing_default` is returned for a null / omitted value — callers pass `inf`
    for `end_time` so the §F overlap guard treats an unknown end as overlapping
    (fail loud, run live) rather than silently passing as 0.0.
    """
    if value is None:
        return missing_default
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value)
    try:
        # ClickHouse returns DateTime64(9) (nanosecond) strings, but
        # datetime.fromisoformat rejects >6 fractional digits before Python 3.11,
        # so truncate the fractional part to microseconds first.
        normalized = re.sub(
            r"(\.\d{6})\d+", r"\1", s.replace("Z", "+00:00")
        )
        return datetime.datetime.fromisoformat(normalized).timestamp()
    except Exception:
        # Lexical fallback: ISO-8601 strings sort chronologically, so map the
        # string to a fraction in [0, 1) whose digits are the (byte-clamped)
        # character codes, earliest character weighted most. This is monotonic
        # in lexical order, so a later timestamp never scores lower. (A flat sum
        # of ordinals — the previous approach — ignores position and silently
        # reorders the spine.) Float64 only resolves the first ~6-7 characters;
        # closer strings tie rather than reverse, and the SQL `ORDER BY
        # start_time` + stable sort preserve their original order on a tie.
        score = 0.0
        weight = 1.0
        for c in s[:32]:
            weight /= 256.0
            score += min(ord(c), 255) * weight
        return score


def fetch_spine_metadata(
    client: LaminarClient, trace_id: str
) -> list[SpanRecord]:
    """Phase 1: lightweight fetch of path / type / times for all spans."""
    records: list[SpanRecord] = []
    offset = 0
    while True:
        rows = client.sql.query(
            "SELECT path, span_type, start_time, end_time FROM spans "
            "WHERE trace_id = {trace_id:UUID} "
            "ORDER BY start_time LIMIT {limit:UInt32} OFFSET {offset:UInt32}",
            parameters={
                "trace_id": trace_id,
                "limit": _PAGE_SIZE,
                "offset": offset,
            },
        )
        if not rows:
            break
        for row in rows:
            records.append(
                SpanRecord(
                    span_path=row.get("path") or "",
                    span_type=str(row.get("span_type") or ""),
                    start_time=_to_epoch(row.get("start_time")),
                    end_time=_to_epoch(
                        row.get("end_time"), missing_default=float("inf")
                    ),
                )
            )
        if len(rows) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    return records


def fetch_spine_payloads(
    client: LaminarClient, trace_id: str, spine_path: str
) -> list[dict[str, Any]]:
    """Phase 2: pull response payloads for the spine path, ordered by start_time.

    Returns a list of dicts shaped like the old CachedSpan
    ({name, input, output, attributes}) so the per-provider
    cached_response_to_* functions consume them unchanged.
    """
    out: list[dict[str, Any]] = []
    offset = 0
    while True:
        rows = client.sql.query(
            "SELECT name, input, output, attributes, start_time FROM spans "
            "WHERE trace_id = {trace_id:UUID} AND path = {path:String} "
            "AND span_type = 'LLM' "
            "ORDER BY start_time LIMIT {limit:UInt32} OFFSET {offset:UInt32}",
            parameters={
                "trace_id": trace_id,
                "path": spine_path,
                "limit": _PAGE_SIZE,
                "offset": offset,
            },
        )
        if not rows:
            break
        for row in rows:
            out.append(_row_to_cached_span(row))
        if len(rows) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    return out


def _row_to_cached_span(row: dict[str, Any]) -> dict[str, Any]:
    attributes = row.get("attributes")
    if isinstance(attributes, str):
        try:
            attributes = json.loads(attributes)
        except Exception:
            attributes = {}
    return {
        "name": row.get("name") or "",
        "input": row.get("input") or "",
        "output": row.get("output") or "",
        "attributes": attributes or {},
        # Epoch nanoseconds: the provider wrappers divide by 1e9 to populate the
        # replayed response's `created` field (e.g. cached_response_to_openai).
        # `_to_epoch` yields epoch seconds, so scale to ns here.
        "start_time": int(_to_epoch(row.get("start_time")) * 1_000_000_000),
    }
