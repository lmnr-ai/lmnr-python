"""`Laminar.initialize(bridges=["langfuse"])` end to end.

Global OTel state (the global tracer provider, instrumented libraries) can be
set once per process, so each case runs `scenarios/initialize_with_bridge.py`
in a fresh interpreter.
"""

from __future__ import annotations

import json
import subprocess
import sys
from functools import cache
from pathlib import Path
from typing import Any

import pytest

_SCENARIO = Path(__file__).parent / "scenarios" / "initialize_with_bridge.py"
_ORDERS = ["laminar_first", "langfuse_first"]


@cache
def _run(mode: str, order: str) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, str(_SCENARIO), mode, order],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def _paths(run: dict[str, Any]) -> list[list[str]]:
    return sorted(s["attributes"].get("lmnr.span.path", []) for s in run["laminar"])


@pytest.mark.parametrize("order", _ORDERS)
def test_langfuse_receives_exactly_what_it_did_without_laminar(order: str):
    assert _run("bridge", order)["langfuse"] == _run("none", order)["langfuse"]


@pytest.mark.parametrize("order", _ORDERS)
def test_bridged_trace_nests_with_laminar_spans_both_ways(order: str):
    assert _paths(_run("bridge", order)) == [
        ["laminar_outer"],
        ["laminar_outer", "agent"],
        ["laminar_outer", "agent", "generate"],
        ["laminar_outer", "agent", "laminar_inner"],
        ["laminar_outer", "agent", "laminar_inner", "lookup"],
        ["laminar_outer", "agent", "lookup"],
    ]
    trace_ids = {s["attributes"]["lmnr.span.ids_path"][0] for s in _run("bridge", order)["laminar"]}
    assert len(trace_ids) == 1


def test_laminar_stays_off_the_global_provider():
    assert _run("bridge", "laminar_first")["laminar_is_global"] is False


def test_conflicting_instrumentations_are_disabled():
    run = _run("bridge", "laminar_first")
    assert run["openai_instrumented"] is False
    assert "OPENAI" in run["block_instruments"]


def test_explicit_instruments_are_honored_with_a_warning():
    run = _run("bridge_explicit_openai", "laminar_first")
    assert run["openai_instrumented"] is True
    assert any("OPENAI" in w for w in run["warnings"])


@pytest.mark.parametrize("order", _ORDERS)
def test_bridge_survives_shutdown_and_reinitialize(order: str):
    assert _paths(_run("bridge_reinit", order)) == _paths(_run("bridge", order))
    assert _run("bridge_reinit", order)["langfuse"] == _run("none", order)["langfuse"]


def test_inherits_global_context_with_the_bridge():
    assert _run("bridge", "laminar_first")["inherits_global_context"] is True


def test_unavailable_bridge_changes_nothing_in_laminar():
    run = _run("bridge_unavailable", "laminar_first")
    assert run["laminar_is_global"] is True
    assert run["block_instruments"] == []
    assert run["inherits_global_context"] is False
    assert any("langfuse bridge needs langfuse >= 3.0" in w for w in run["warnings"])


def test_unknown_bridge_is_rejected():
    from lmnr.opentelemetry_lib.bridges import resolve_bridges

    with pytest.raises(ValueError, match="Unknown Laminar bridge 'langfsue'"):
        _bridges = resolve_bridges(["langfsue"])
