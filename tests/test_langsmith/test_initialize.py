"""What `Laminar.initialize(bridges=["langsmith"])` changes in Laminar's own setup.

`init_tracing` is mocked, as in `tests/test_langfuse/test_initialize.py`.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langsmith.client import Client

from lmnr.opentelemetry_lib.bridges import uninstall_bridges
from lmnr.opentelemetry_lib.bridges.langsmith import LangSmithBridge
from lmnr.opentelemetry_lib.tracing import context as laminar_context
from lmnr.opentelemetry_lib.tracing.instruments import Instruments
from lmnr.sdk.laminar import Laminar

_LAMINAR_STATE = (
    "_Laminar__initialized",
    "_Laminar__project_api_key",
    "_Laminar__base_http_url",
    "_Laminar__global_metadata",
    "_Laminar__base_url_for_debug",
    "_Laminar__http_port_for_debug",
)


@pytest.fixture
def init_tracing(monkeypatch: pytest.MonkeyPatch) -> Generator[MagicMock, None, None]:
    for name in _LAMINAR_STATE:
        monkeypatch.setattr(Laminar, name, getattr(Laminar, name, None))
    monkeypatch.setattr(Laminar, "_Laminar__initialized", False)
    with patch("lmnr.sdk.laminar.init_tracing") as mock:
        yield mock
    uninstall_bridges()


def _kwargs(init_tracing: MagicMock) -> Mapping[str, Any]:
    init_tracing.assert_called_once()
    return init_tracing.call_args.kwargs


def test_bridge_blocks_conflicts_and_keeps_laminar_off_the_global_provider(
    init_tracing: MagicMock,
) -> None:
    original = Client._create_run

    Laminar.initialize(project_api_key="test", bridges=["langsmith"])

    kwargs = _kwargs(init_tracing)
    blocked: set[Instruments] = kwargs["block_instruments"]
    assert LangSmithBridge.conflicting_instruments <= blocked
    assert {Instruments.LANGCHAIN, Instruments.OPENAI, Instruments.ANTHROPIC} <= blocked
    # It records no spans; the bridge reads the graph topology it publishes.
    assert Instruments.LANGGRAPH not in blocked
    assert kwargs["set_global_tracer_provider"] is False
    # LangSmith isn't OpenTelemetry-based: no global-context inheritance.
    assert laminar_context._inherit_global_context is False
    assert Client._create_run is not original

    uninstall_bridges()
    assert Client._create_run is original


def test_old_langsmith_changes_nothing_in_laminar(
    init_tracing: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "lmnr.opentelemetry_lib.bridges.langsmith._langsmith_version", lambda: "0.4.14"
    )
    original = Client._create_run

    Laminar.initialize(project_api_key="test", bridges=["langsmith"])

    kwargs = _kwargs(init_tracing)
    assert kwargs["block_instruments"] is None
    assert kwargs["set_global_tracer_provider"] is True
    assert Client._create_run is original
