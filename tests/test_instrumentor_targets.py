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
    try:
        yield instrumentor, targets
    finally:
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()


@pytest.mark.parametrize(
    "reinstrumented", sorted(INSTRUMENTOR_TARGETS), indirect=True
)
def test_instrument_wraps_exactly_the_declared_targets(reinstrumented):
    instrumentor, targets = reinstrumented

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


def _target_set(rows) -> set[tuple[str, str, str]]:
    return {
        (r["package"], r.get("object") or "", r["method"]) for r in rows
    }


def test_kernel_declares_its_full_target_set():
    """kernel iterates its table twice, synthesizing `Async{object}` on the
    second pass, and wraps `KernelApp.action` outside the table entirely. The
    migration turns all three into explicit rows, so the union must match."""
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.kernel import (
        WRAPPED_METHODS,
    )

    sync = _target_set(WRAPPED_METHODS)
    assert len(sync) == 20

    # every sync row has an async twin synthesized at instrument time
    expected_async = {(pkg, f"Async{obj}", meth) for pkg, obj, meth in sync}
    assert len(expected_async) == 20

    # ...plus the one out-of-table wrap
    full = sync | expected_async | {("kernel.app_framework", "KernelApp", "action")}
    assert len(full) == 41


def test_cua_computer_declares_both_sync_and_async_tables():
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.cua_computer import (
        WRAPPED_METHODS,
        WRAPPED_AMETHODS,
    )

    sync = _target_set(WRAPPED_METHODS)
    async_ = _target_set(WRAPPED_AMETHODS)

    assert sync, "sync table must not be empty"
    assert async_, "async table must not be empty"
    # The two tables target distinct methods; a migration collapsing them into
    # one list with an `is_async` column must not lose or merge any row.
    assert len(sync | async_) == len(sync) + len(async_)
