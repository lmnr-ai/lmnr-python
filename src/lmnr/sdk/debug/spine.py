"""Spine detection over a source trace's spans.

Implements the authoritative §7 heuristic plus the overlap guard from §F.
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


@dataclass
class SpineResult:
    spine_path: str | None
    # Spine calls in execution order (by start_time). Empty when no spine.
    spine_calls: list[SpanRecord]
    overlap: bool = False


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
    llm_spans = [s for s in spans if s.span_type == "LLM"]
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
