"""Temporal header <-> Laminar span context codec.

Temporal headers are ``Mapping[str, temporalio.api.common.v1.Payload]``. We
carry the full serialized ``LaminarSpanContext`` (which nests the optional
``debug`` block) under ``x-lmnr-span-context`` and a plain W3C ``traceparent``
for interop. Both values are encoded as ``json/plain`` Temporal Payloads via the
SDK's default ``PayloadConverter`` so we never depend on the wire format
ourselves.

This is the Python port of the TS ``helpers.ts``. The crucial difference from
TS: there is NO ``pushLaminarContext`` here. On the Python span-creation path,
``Laminar.start_span(parent_span_context=...)`` already (a) parses the nested
``debug`` block and arms the downstream debug runtime, and (b) stamps the
``lmnr.span.path`` / ``lmnr.span.ids_path`` parent attributes so the processor
builds correct nested paths. So restoration just returns the
``LaminarSpanContext`` to hand to ``parent_span_context=``.
"""

from __future__ import annotations

import uuid

from temporalio.api.common.v1 import Payload
from temporalio.converter import PayloadConverter

from lmnr.sdk.log import get_default_logger
from lmnr.sdk.types import LaminarSpanContext

from .consts import LAMINAR_SPAN_CONTEXT_HEADER, TRACEPARENT_HEADER

logger = get_default_logger(__name__)

_payload_converter = PayloadConverter.default


def encode_payload(value: str) -> Payload:
    """Encode a string as a ``json/plain`` Temporal Payload."""
    return _payload_converter.to_payloads([value])[0]


def decode_payload(payload: Payload | None) -> str | None:
    """Decode a Temporal Payload back into a string, or ``None`` on failure."""
    if payload is None:
        return None
    try:
        value = _payload_converter.from_payloads([payload])[0]
        return value if isinstance(value, str) else None
    except Exception:
        return None


def build_headers(
    existing: dict[str, Payload] | None,
    span_context: LaminarSpanContext | None,
) -> dict[str, Payload]:
    """Inject the given Laminar span context into a Temporal headers map.

    Writes both ``x-lmnr-span-context`` (full Laminar JSON — preferred) and
    ``traceparent`` (W3C, for interop). Returns ``existing`` unchanged when
    there is no span context to propagate.
    """
    headers: dict[str, Payload] = {**(existing or {})}
    if span_context is None:
        return headers

    headers[LAMINAR_SPAN_CONTEXT_HEADER] = encode_payload(str(span_context))
    trace_hex = span_context.trace_id.hex
    span_hex = span_context.span_id.hex[16:]
    headers[TRACEPARENT_HEADER] = encode_payload(f"00-{trace_hex}-{span_hex}-01")
    return headers


def restore_context_from_headers(
    headers: dict[str, Payload] | None,
) -> LaminarSpanContext | None:
    """Read a Laminar span context out of Temporal headers.

    Prefers the full ``x-lmnr-span-context`` header; falls back to W3C
    ``traceparent``. Returns ``None`` when neither carries usable trace context.
    The returned context is meant to be passed straight to
    ``Laminar.start_span(parent_span_context=...)``.
    """
    if not headers:
        return None

    laminar_raw = decode_payload(headers.get(LAMINAR_SPAN_CONTEXT_HEADER))
    if laminar_raw:
        try:
            return LaminarSpanContext.deserialize(laminar_raw)
        except Exception as e:
            logger.warning(
                f"Could not restore {LAMINAR_SPAN_CONTEXT_HEADER}: {e}"
            )

    traceparent = decode_payload(headers.get(TRACEPARENT_HEADER))
    if traceparent:
        parts = traceparent.split("-")
        if len(parts) >= 3:
            try:
                return LaminarSpanContext(
                    trace_id=uuid.UUID(hex=parts[1]),
                    span_id=uuid.UUID(hex=parts[2].rjust(32, "0")),
                    is_remote=True,
                )
            except Exception as e:
                logger.warning(f"Could not restore traceparent: {e}")

    return None
