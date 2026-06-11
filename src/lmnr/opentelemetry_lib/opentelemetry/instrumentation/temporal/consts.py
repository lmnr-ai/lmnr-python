"""Header keys used to propagate Laminar trace context through Temporal.

Kept byte-identical to the TypeScript integration
(`packages/lmnr/src/opentelemetry-lib/instrumentation/temporal/consts.ts`) so a
TS client and a Python worker (or vice versa) interoperate on the wire.
"""

#: Header key carrying the full serialized `LaminarSpanContext` (preferred).
LAMINAR_SPAN_CONTEXT_HEADER = "x-lmnr-span-context"

#: W3C ``traceparent`` header key — written alongside the Laminar header for
#: interop with non-Laminar clients/workers that only understand W3C trace
#: context.
TRACEPARENT_HEADER = "traceparent"
