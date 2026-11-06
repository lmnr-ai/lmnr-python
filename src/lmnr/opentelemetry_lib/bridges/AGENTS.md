# Bridges

A bridge sends what a competitor SDK records (Langfuse now; LangSmith and Braintrust planned) to Laminar too, enabled with `Laminar.initialize(bridges=[...])`. The competitor's own backend must receive exactly what it did without Laminar.

- `base.py`: `Bridge` (str enum; `bridges=` accepts members or their values) and `LaminarBridge`, the class every bridge subclasses:
  - `conflicting_instruments`: Laminar instruments that record the same calls. `Laminar.initialize` adds them to the blocked set when `instruments` is not passed; an explicit `instruments` set is honored as is, with a warning naming the overlap.
  - `requires_private_tracer_provider`: keep Laminar's provider off the global slot (and don't set `OTEL_ATTRIBUTE_COUNT_LIMIT` process-wide).
  - `inherits_global_context`: Laminar spans nest under the competitor's span active in the global OTel context (`tracing/context.py::set_inherit_global_context`).
  - `is_available()`, `install(lmnr_tracer_provider, lmnr_span_processor)`, `uninstall()`; neither may raise.
- `__init__.py`: one singleton per bridge in `_BRIDGES`. `resolve_bridges` raises `ValueError` on an unknown name before `initialize` marks itself initialized. `Laminar.initialize` installs the resolved bridges after `init_tracing`; `Laminar.shutdown` uninstalls all of them, so a re-initialize re-attaches with the new span processor.
- Adding a bridge: a `Bridge` member, a `LaminarBridge` subclass in its own package, and an `_BRIDGES` entry. Stamp `lmnr.span.bridge.name` / `lmnr.span.bridge.version` (`tracing/attributes.py`) on every span it produces. The TS SDK (`packages/lmnr/src/opentelemetry-lib/bridges/`) has the same shape; keep names and attributes in step.
- Bridges replaced `Instruments.LANGFUSE` and `Laminar.connect_to_langfuse()` in 0.9.0.
