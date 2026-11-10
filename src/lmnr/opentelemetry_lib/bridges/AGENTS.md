# Bridges

A bridge sends what a third-party tracing SDK records (Langfuse now; LangSmith and Braintrust planned) to Laminar too, enabled with `Laminar.initialize(bridges=[...])`. The source SDK's own backend must receive exactly what it did without Laminar.

- **Terminology:** call the bridged SDK the "source SDK" ("third-party tracing SDK" in public docstrings), never a "competitor". Future bridges need not be competing products, and the word spreads from here into other code and docs.
- **The contract assumes nothing about how a bridge hooks its SDK.** Langfuse is built on OpenTelemetry, so its bridge adds a span processor to Langfuse's tracer provider. LangSmith and Braintrust aren't, so their bridges will likely hook the SDK at the network level. Both shapes reach Laminar through `BridgeContext`:
  - `span_processor`: a bridge that receives finished spans hands Laminar's copies to it (Langfuse).
  - `tracer_provider`: a bridge that builds spans itself creates them with `tracer_provider.get_tracer(...)`, which routes them through the same processor. Langfuse only uses it to never attach to Laminar's own provider.
- `base.py`: `Bridge` (str enum; `bridges=` accepts members or their values), `BridgeContext`, and `LaminarBridge`, the class every bridge subclasses.
  - Every bridge sets:
    - `conflicting_instruments`: Laminar instruments that record the same calls. `Laminar.initialize` adds them to the blocked set when `instruments` is not passed; an explicit `instruments` set is honored as is, with a warning naming the overlap.
    - `requirement`: the text of the warning when `is_available()` is False.
    - `is_available()`, `install(context) -> bool`, `uninstall()`; none may raise.
  - Only OpenTelemetry-based SDKs set these (both default False; the last section of the class):
    - `requires_private_tracer_provider`: keep Laminar's provider off the global slot (and don't set `OTEL_ATTRIBUTE_COUNT_LIMIT` process-wide), so the source SDK doesn't adopt it.
    - `inherits_global_context`: Laminar spans nest under the source SDK's span active in the global OTel context (`tracing/context.py::set_inherit_global_context`).
  - **A bridge that isn't available has no effect at all** (Bugbot, PR #422): `Laminar.initialize` drops it via `available_bridges` before computing blocked instruments and the private provider, so a missing or too-old source SDK leaves Laminar exactly as without `bridges`. Global-context inheritance follows the bridges `install_bridges` reports as installed. Provider and instrument choices are made before `install` (it needs the provider), so an `install` that fails after `is_available()` passed still keeps them; that path only logs.
- `__init__.py`: one singleton per bridge in `_BRIDGES`. `resolve_bridges` raises `ValueError` on an unknown name before `initialize` marks itself initialized. `Laminar.initialize` installs the resolved bridges after `init_tracing`; `Laminar.shutdown` uninstalls all of them, so a re-initialize re-attaches with the new span processor.
- Adding a bridge: a `Bridge` member, a `LaminarBridge` subclass in its own package, and an `_BRIDGES` entry. Stamp `lmnr.span.bridge.name` / `lmnr.span.bridge.version` (`tracing/attributes.py`) on every span it produces.
- The TS SDK (`packages/lmnr/src/opentelemetry-lib/bridges/`) has the same contract: its `Bridge` has `inheritsGlobalContext` but no private-provider flag (the TS Langfuse SDK keeps its own provider slot), and its `BridgeContext` has only the span processor, since TS code gets Laminar's tracer from `getTracer()`. Keep names and attributes in step.
- Bridges replaced `Instruments.LANGFUSE` and `Laminar.connect_to_langfuse()` in 0.9.0.
