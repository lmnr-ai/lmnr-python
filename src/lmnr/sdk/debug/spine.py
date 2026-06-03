"""Spine detection over a source trace's spans.

Implements the authoritative heuristic plus the overlap guard.
Part of the cross-language parity surface — keep line-comparable with the TS
`spine.ts` and cover with the shared test vectors.
"""

from dataclasses import dataclass

from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

# Span-path segment separator. Matches the backend `flat_path` join (".").
PATH_SEPARATOR = "."


@dataclass
class SpanRecord:
    """The minimal span fields spine detection needs."""

    span_path: str
    span_type: str
    start_time: float
    end_time: float
    # The span's UUID (as stored in ClickHouse). Used to resolve a span-id
    # `LMNR_DEBUG_CACHE_UNTIL` to an occurrence count; "" when unavailable.
    span_id: str = ""


@dataclass
class SpineResult:
    spine_path: str | None
    # Spine calls in execution order (by start_time). Empty when no spine.
    spine_calls: list[SpanRecord]


def _depth(span_path: str) -> int:
    """Number of path segments. Identical computation in both SDKs."""
    if not span_path:
        return 0
    return len(span_path.split(PATH_SEPARATOR))


def detect_spine(spans: list[SpanRecord]) -> SpineResult:
    """Compute the spine over a fetched source trace.

    Returns the chosen spine_path and its calls sorted by start_time. spine_path
    is None only when the trace has zero LLM spans.
    """
    llm_spans = [s for s in spans if s.span_type in ["LLM", "CACHED"]]
    if not llm_spans:
        logger.warning("Source trace has no LLM spans; nothing to cache.")
        return SpineResult(spine_path=None, spine_calls=[])

    groups: dict[str, list[SpanRecord]] = {}
    for s in llm_spans:
        groups.setdefault(s.span_path, []).append(s)

    candidates = {path: calls for path, calls in groups.items() if len(calls) >= 2}

    if candidates:
        # Primary rule: shallowest looping path; tie-break by earliest start.
        def primary_key(path: str) -> tuple[int, float]:
            calls = groups[path]
            return (_depth(path), min(c.start_time for c in calls))

        spine_path = min(candidates.keys(), key=primary_key)
    else:
        # Fallback: shallowest single LLM call site; tie-break by earliest start.
        def fallback_key(s: SpanRecord) -> tuple[int, float]:
            return (_depth(s.span_path), s.start_time)

        spine_path = min(llm_spans, key=fallback_key).span_path

    spine_calls = sorted(groups[spine_path], key=lambda s: s.start_time)
    return SpineResult(spine_path=spine_path, spine_calls=spine_calls)


def _match_span_id(needle: str, span_id: str) -> bool:
    """True if `needle` (hyphen-stripped lowercase hex) suffix-matches `span_id`.

    The source-trace span ids are full UUIDs, but the user may pass a full UUID,
    the last two groups, the raw 16-hex OTel id, or a short hex suffix — all of
    which are a suffix of the hyphen-stripped UUID. Identical logic in both SDKs.
    """
    if not needle or not span_id:
        return False
    haystack = span_id.lower().replace("-", "")
    return haystack.endswith(needle)


def resolve_cache_until_span_id(
    spine_calls: list[SpanRecord], needle: str
) -> int | None:
    """Resolve a span-id `cache_until` to an occurrence count over the spine.

    The spine calls are in execution order, so the matched span's 1-based index
    is the number of calls to cache (inclusive of the target). Returns None when
    the needle matches none of the spine calls (invalid / not on the spine path)
    so the caller can warn and degrade to live.
    """
    for index, call in enumerate(spine_calls):
        if _match_span_id(needle, call.span_id):
            return index + 1
    return None


def has_overlap(spine_calls: list[SpanRecord], n: int) -> bool:
    """True if any of the first N spine calls overlap in [start, end].

    v1 assumes the spine is sequential; overlapping calls mean we cannot safely
    map occurrence index to a cached response, so the caller must fail loud.
    """
    window = sorted(spine_calls[:n], key=lambda s: s.start_time)
    for prev, cur in zip(window, window[1:]):
        if cur.start_time < prev.end_time:
            return True
    return False
