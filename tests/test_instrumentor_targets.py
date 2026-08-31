"""Characterization tests: which targets each instrumentor actually wraps.

These packages have no other test coverage, and they are about to be migrated
from hand-rolled `_instrument` loops onto `BaseLaminarInstrumentor`. The migration
is only safe if the set of wrapped targets — and the restore-on-uninstrument
behavior — is unchanged, so that is exactly what these assert.

Deliberately asserting on the *wrapped attribute*, not on emitted spans: that
needs no network, no provider credentials and no cassettes, and it is the part
the migration actually moves.
"""

import importlib
import sys

import pytest

# (instrumentor import path, class name, {(module, "Object.method" | "func"), ...})
#
# Recorded from the pre-migration `_instrument` bodies. `opentelemetry`,
# `langgraph` and `cua_agent` have no declarative table at all today — their
# targets are hardcoded inline — which is precisely why pinning them here first
# is worthwhile.
INSTRUMENTOR_TARGETS: dict[str, tuple[str, str, set[tuple[str, str]]]] = {
    "opentelemetry": (
        "lmnr.opentelemetry_lib.opentelemetry.instrumentation.opentelemetry",
        "OpentelemetryInstrumentor",
        {("opentelemetry.trace.span", "NonRecordingSpan.get_span_context")},
    ),
    "langgraph": (
        "lmnr.opentelemetry_lib.opentelemetry.instrumentation.langgraph",
        "LanggraphInstrumentor",
        {
            ("langgraph.pregel", "Pregel.stream"),
            ("langgraph.pregel", "Pregel.astream"),
        },
    ),
    "claude_agent": (
        "lmnr.opentelemetry_lib.opentelemetry.instrumentation.claude_agent",
        "ClaudeAgentInstrumentor",
        {
            (
                "claude_agent_sdk._internal.transport.subprocess_cli",
                "SubprocessCLITransport.connect",
            ),
            (
                "claude_agent_sdk._internal.transport.subprocess_cli",
                "SubprocessCLITransport.close",
            ),
            ("claude_agent_sdk.client", "ClaudeSDKClient.__init__"),
            ("claude_agent_sdk.client", "ClaudeSDKClient.connect"),
            ("claude_agent_sdk.client", "ClaudeSDKClient.query"),
            ("claude_agent_sdk.client", "ClaudeSDKClient.receive_messages"),
            ("claude_agent_sdk.client", "ClaudeSDKClient.receive_response"),
            ("claude_agent_sdk.client", "ClaudeSDKClient.interrupt"),
            ("claude_agent_sdk.client", "ClaudeSDKClient.disconnect"),
            # the two module-level functions, wrapped via alias replacement
            ("claude_agent_sdk", "query"),
            ("claude_agent_sdk", "create_sdk_mcp_server"),
        },
    ),
    # NOTE: skyvern is deliberately absent. It is also migrated, but its
    # instrumentor module hard-imports `skyvern` at module level, and skyvern
    # requires Python 3.11+ while this SDK supports 3.10 — so it cannot be a dev
    # dependency and none of its targets are importable here.
}


def _resolve(module_name: str, target: str):
    """Resolve a `("module", "Object.method")` pair to the attribute holder."""
    module = importlib.import_module(module_name)
    parts = target.split(".")
    holder = module
    for part in parts[:-1]:
        holder = getattr(holder, part)
    return holder, parts[-1]


def _is_wrapped(module_name: str, target: str) -> bool:
    holder, attr = _resolve(module_name, target)
    return hasattr(getattr(holder, attr), "__wrapped__")


def _reachable(targets: set[tuple[str, str]]) -> set[tuple[str, str]]:
    """Drop targets whose module cannot be imported in this environment.

    Every `_instrument` here swallows ModuleNotFoundError per wrap, so an
    unreachable target is legitimately left unwrapped rather than being a
    failure. Asserting on them anyway would make the suite depend on the target
    library's own optional extras.
    """
    ok = set()
    for module_name, target in targets:
        try:
            _resolve(module_name, target)
        except (ImportError, AttributeError):
            continue
        ok.add((module_name, target))
    return ok


def _instrumentor(key: str):
    module_path, class_name, targets = INSTRUMENTOR_TARGETS[key]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)(), targets


@pytest.fixture
def reinstrumented(request):
    """Yield a (instrumentor, targets) pair, restoring instrumented state after.

    `BaseInstrumentor` is a singleton and the session-scoped conftest fixture has
    already instrumented everything, so these tests drive a full
    uninstrument -> instrument cycle rather than starting from a clean slate —
    and must hand the process back instrumented, or every later test in the
    session loses its spans.
    """
    instrumentor, targets = _instrumentor(request.param)
    was_instrumented = instrumentor.is_instrumented_by_opentelemetry
    try:
        yield instrumentor, targets
    finally:
        # Restore whatever we found, not a fixed state: some of these are
        # auto-enabled by the conftest session fixture and some are not, and
        # leaving either one flipped would leak into the rest of the session.
        if was_instrumented and not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()
        elif not was_instrumented and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


@pytest.mark.parametrize(
    "reinstrumented", sorted(INSTRUMENTOR_TARGETS), indirect=True
)
def test_instrument_wraps_exactly_the_declared_targets(reinstrumented):
    instrumentor, targets = reinstrumented
    targets = _reachable(targets)
    if not targets:
        pytest.skip("no declared target is importable in this environment")

    instrumentor.uninstrument()
    for module_name, target in targets:
        assert not _is_wrapped(module_name, target), (
            f"{module_name}.{target} still wrapped after uninstrument()"
        )

    instrumentor.instrument()
    for module_name, target in targets:
        assert _is_wrapped(module_name, target), (
            f"failed to wrap {module_name}.{target}"
        )


@pytest.mark.parametrize(
    "reinstrumented", sorted(INSTRUMENTOR_TARGETS), indirect=True
)
def test_uninstrument_restores_the_original_attribute(reinstrumented):
    """A migration that wraps correctly but leaks on teardown would still break
    any host that re-initializes Laminar (test suites, notebooks)."""
    instrumentor, targets = reinstrumented
    targets = _reachable(targets)
    if not targets:
        pytest.skip("no declared target is importable in this environment")

    instrumentor.uninstrument()
    originals = {}
    for module_name, target in targets:
        holder, attr = _resolve(module_name, target)
        originals[(module_name, target)] = getattr(holder, attr)

    instrumentor.instrument()
    instrumentor.uninstrument()

    for (module_name, target), original in originals.items():
        holder, attr = _resolve(module_name, target)
        assert getattr(holder, attr) is original, (
            f"did not restore {module_name}.{target} to the original function"
        )



def test_claude_agent_alias_replacement_reaches_prior_from_imports():
    """claude_agent carried its own private copy of the alias-replacement
    machinery; it now uses `BaseLaminarInstrumentor`'s. The behaviour that copy
    existed for: a caller who did `from claude_agent_sdk import query` BEFORE
    instrumentation still ends up with the wrapped function, and gets the
    original back on uninstrument.
    """
    import claude_agent_sdk

    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.claude_agent import (
        ClaudeAgentInstrumentor,
    )

    instrumentor = ClaudeAgentInstrumentor()
    was_instrumented = instrumentor.is_instrumented_by_opentelemetry
    try:
        if was_instrumented:
            instrumentor.uninstrument()

        # a module that grabbed its own reference before we instrumented
        import types

        consumer = types.ModuleType("_lmnr_claude_consumer")
        consumer.query = claude_agent_sdk.query
        sys.modules["_lmnr_claude_consumer"] = consumer
        original = consumer.query

        instrumentor.instrument()
        assert consumer.query is not original, (
            "alias replacement did not reach a pre-existing `from ... import query`"
        )
        assert hasattr(consumer.query, "__wrapped__")

        instrumentor.uninstrument()
        assert consumer.query is original, (
            "alias replacement did not restore the pre-existing reference"
        )
    finally:
        sys.modules.pop("_lmnr_claude_consumer", None)
        if instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()
        if was_instrumented:
            instrumentor.instrument()


# ---------------------------------------------------------------------------
# Browser group
# ---------------------------------------------------------------------------


def test_playwright_wraps_and_restores_every_target():
    """playwright is installed here, so this is a real round trip. It also needs
    a client, which reaches the wrappers through `wrapper_kwargs()` rather than
    through the per-method spec."""
    from lmnr.sdk.browser.playwright_otel import (
        PlaywrightInstrumentor,
        WRAPPED_FUNCTIONS,
    )

    targets = _reachable(
        {
            (r["package_name"], f"{r['object_name']}.{r['method_name']}")
            for r in WRAPPED_FUNCTIONS
        }
    )
    assert targets, "expected at least one playwright target to be importable"

    instrumentor = PlaywrightInstrumentor(object())
    was_instrumented = instrumentor.is_instrumented_by_opentelemetry
    try:
        if was_instrumented:
            instrumentor.uninstrument()
        originals = {}
        for module_name, target in targets:
            holder, attr = _resolve(module_name, target)
            originals[(module_name, target)] = getattr(holder, attr)

        instrumentor.instrument()
        for module_name, target in targets:
            assert _is_wrapped(module_name, target), f"missed {module_name}.{target}"

        instrumentor.uninstrument()
        for (module_name, target), original in originals.items():
            holder, attr = _resolve(module_name, target)
            assert getattr(holder, attr) is original, (
                f"did not restore {module_name}.{target}"
            )
    finally:
        if instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()
        if was_instrumented:
            instrumentor.instrument()


def test_wrapper_kwargs_delivers_the_client_to_a_browser_wrapper():
    """The client is instrumentor-level state, not per-method config, so it rides
    the base's handler-kwargs channel. If that link broke, every browser wrapper
    would raise TypeError on its keyword-only `client` argument."""
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.wrapper_helpers import (
        add_spec_wrapper,
    )
    from lmnr.sdk.browser.playwright_otel import PlaywrightInstrumentor

    sentinel = object()
    instrumentor = PlaywrightInstrumentor(sentinel)
    assert instrumentor.wrapper_kwargs() == {"client": sentinel}

    seen = {}

    def handler(to_wrap, wrapped, instance, args, kwargs, *, client):
        seen["client"] = client
        return wrapped(*args, **kwargs)

    wrapper = add_spec_wrapper(handler, {}, **instrumentor.wrapper_kwargs())
    assert wrapper(lambda: "ok", None, (), {}) == "ok"
    assert seen["client"] is sentinel


def test_patchright_table_is_derived_from_playwright_and_covers_new_page():
    """patchright's table used to be a hand-maintained copy of playwright's and
    had drifted: both `Browser.new_page` rows were missing, so pages opened that
    way silently lost session recording. It is now derived, so it cannot drift.
    """
    from lmnr.sdk.browser.patchright_otel import WRAPPED_FUNCTIONS as PATCHRIGHT
    from lmnr.sdk.browser.playwright_otel import WRAPPED_FUNCTIONS as PLAYWRIGHT

    assert len(PATCHRIGHT) == len(PLAYWRIGHT) == 14

    # identical except for the package they target
    assert {
        (r["object_name"], r["method_name"], r["is_async"], r["wrapper_function"])
        for r in PATCHRIGHT
    } == {
        (r["object_name"], r["method_name"], r["is_async"], r["wrapper_function"])
        for r in PLAYWRIGHT
    }
    assert {r["package_name"] for r in PATCHRIGHT} == {
        "patchright.sync_api",
        "patchright.async_api",
    }

    # the rows that were missing before
    new_page = [r for r in PATCHRIGHT if r["method_name"] == "new_page"]
    assert {r["is_async"] for r in new_page} == {False, True}


# ---------------------------------------------------------------------------
# The unwrap() calling convention
# ---------------------------------------------------------------------------


def test_unwrap_silently_no_ops_on_the_wrapt_argument_split():
    """Documents the trap every broken `_uninstrument` fell into.

    `wrap_function_wrapper` takes (module, "Object.method") and resolves the
    dotted attribute path itself. `unwrap` takes (holder, "attr") and does a
    single getattr — so handing it wrapt's split makes it look for an attribute
    literally named "Object.method", find nothing, and return WITHOUT raising.
    The instrumentor looks like it uninstrumented; the method stays wrapped.
    """
    import sys
    import types

    from opentelemetry.instrumentation.utils import unwrap
    from wrapt import wrap_function_wrapper

    # A throwaway package so this is hermetic: the real instrumentors have
    # already wrapped their own targets session-wide via conftest, and layering
    # a second wrap on those would make the assertions ambiguous. It needs two
    # levels because `unwrap` rsplits the string it is given.
    parent = types.ModuleType("_lmnr_probe")
    child = types.ModuleType("_lmnr_probe.mod")
    parent.mod = child

    class Target:
        def method(self):
            return "original"

    child.Target = Target
    sys.modules["_lmnr_probe"] = parent
    sys.modules["_lmnr_probe.mod"] = child
    try:
        wrap_function_wrapper(
            "_lmnr_probe.mod",
            "Target.method",
            lambda wrapped, instance, args, kwargs: wrapped(*args, **kwargs),
        )
        assert hasattr(Target.method, "__wrapped__")

        # wrapt's split: no exception, no effect
        unwrap("_lmnr_probe.mod", "Target.method")
        assert hasattr(Target.method, "__wrapped__"), (
            "if unwrap ever starts handling the wrapt split, the explicit "
            "holder-path calls in every _uninstrument can be simplified"
        )

        # the correct split
        unwrap("_lmnr_probe.mod.Target", "method")
        assert not hasattr(Target.method, "__wrapped__")
    finally:
        del sys.modules["_lmnr_probe"]
        del sys.modules["_lmnr_probe.mod"]


# ---------------------------------------------------------------------------
# Declarative tables — assertable without the target library installed
# ---------------------------------------------------------------------------


def test_kernel_table_covers_the_sync_async_and_app_action_wraps():
    """kernel used to iterate its 20-row table twice — synthesizing
    `Async{object}` on the second pass — and wrap `KernelApp.action` outside the
    table entirely. All three are now explicit rows; the totals must still match.
    """
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.kernel import (
        WRAPPED_FUNCTIONS,
    )

    assert len(WRAPPED_FUNCTIONS) == 41

    sync = [
        r
        for r in WRAPPED_FUNCTIONS
        if not r["is_async"] and r["object_name"] != "KernelApp"
    ]
    async_ = [r for r in WRAPPED_FUNCTIONS if r["is_async"]]
    assert len(sync) == 20
    assert len(async_) == 20

    # each async row is the `Async`-prefixed twin of a sync row, on the same
    # package and method — that is exactly what the second pass used to build.
    assert {
        (r["package_name"], f"Async{r['object_name']}", r["method_name"])
        for r in sync
    } == {(r["package_name"], r["object_name"], r["method_name"]) for r in async_}

    app_action = [r for r in WRAPPED_FUNCTIONS if r["object_name"] == "KernelApp"]
    assert [r["method_name"] for r in app_action] == ["action"]


def test_kernel_wraps_and_restores_a_real_target():
    """kernel is actually installed here, so unlike the other tables this one can
    be verified end to end rather than only declaratively."""
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.kernel import (
        KernelInstrumentor,
    )

    target = ("kernel.resources.browsers", "BrowsersResource.create")
    instrumentor = KernelInstrumentor()
    was_instrumented = instrumentor.is_instrumented_by_opentelemetry
    try:
        if was_instrumented:
            instrumentor.uninstrument()
        original = getattr(*_resolve(*target)[:1], _resolve(*target)[1])

        instrumentor.instrument()
        assert _is_wrapped(*target)

        instrumentor.uninstrument()
        holder, attr = _resolve(*target)
        assert getattr(holder, attr) is original
    finally:
        if instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()
        if was_instrumented:
            instrumentor.instrument()


def test_cua_computer_merged_table_preserves_every_row():
    """cua_computer's two tables (sync + async) were collapsed into one list with
    an `is_async` column. Pin the resulting counts and the per-method extras, so
    a row cannot be dropped or silently flipped to the wrong wrapper.
    """
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.cua_computer import (
        WRAPPED_FUNCTIONS,
    )

    assert len(WRAPPED_FUNCTIONS) == 41
    assert sum(1 for r in WRAPPED_FUNCTIONS if not r["is_async"]) == 2
    assert sum(1 for r in WRAPPED_FUNCTIONS if r["is_async"]) == 39

    # every row targets a distinct (package, object, method)
    targets = {
        (r["package_name"], r["object_name"], r["method_name"])
        for r in WRAPPED_FUNCTIONS
    }
    assert len(targets) == len(WRAPPED_FUNCTIONS)

    # the two lifecycle rows that open/close the parent `computer.run` span
    assert {r.get("action") for r in WRAPPED_FUNCTIONS if r.get("action")} == {
        "start_parent_span",
        "end_parent_span",
    }
    # screenshot is the one row whose payload is swapped for a placeholder
    formatters = [r for r in WRAPPED_FUNCTIONS if r.get("output_formatter")]
    assert [r["method_name"] for r in formatters] == ["screenshot"]
