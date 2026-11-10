"""What `Laminar.initialize(bridges=[...])` changes in Laminar's own setup.

`init_tracing` is mocked (as in `tests/test_initialize.py`), so these check
the arguments `initialize` derives from the bridges, not real tracing.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator, Mapping
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langfuse._client.resource_manager import LangfuseResourceManager
from typing_extensions import override

from lmnr.opentelemetry_lib.bridges import uninstall_bridges
from lmnr.opentelemetry_lib.bridges.langfuse import LangfuseBridge
from lmnr.opentelemetry_lib.tracing import context as laminar_context
from lmnr.opentelemetry_lib.tracing.context import set_inherit_global_context
from lmnr.opentelemetry_lib.tracing.instruments import Instruments
from lmnr.sdk.laminar import Laminar

# Class state `initialize` sets; restored after each test.
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
    LangfuseResourceManager._instances.clear()
    with patch("lmnr.sdk.laminar.init_tracing") as mock:
        yield mock
    uninstall_bridges()
    set_inherit_global_context(False)


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    @override
    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def logs() -> Generator[Callable[[], list[str]], None, None]:
    """Warnings from Laminar and the bridges. Their loggers don't propagate
    (and `initialize` resets that), so a handler is attached to each."""
    capture = _Capture()
    loggers = [logging.getLogger(n) for n in ("lmnr.sdk.laminar", "lmnr.opentelemetry_lib.bridges")]
    for lg in loggers:
        lg.addHandler(capture)
    yield lambda: capture.messages
    for lg in loggers:
        lg.removeHandler(capture)


def _initialize(**kwargs: Any) -> None:
    Laminar.initialize(project_api_key="test", **kwargs)


def _kwargs(init_tracing: MagicMock) -> Mapping[str, Any]:
    init_tracing.assert_called_once()
    return init_tracing.call_args.kwargs


def test_bridge_blocks_conflicts_and_keeps_laminar_off_the_global_provider(
    init_tracing: MagicMock,
):
    original_initialize_instance = LangfuseResourceManager._initialize_instance

    _initialize(bridges=["langfuse"])

    kwargs = _kwargs(init_tracing)
    assert LangfuseBridge.conflicting_instruments <= kwargs["block_instruments"]
    assert Instruments.CLAUDE_AGENT not in kwargs["block_instruments"]
    assert kwargs["set_global_tracer_provider"] is False
    assert laminar_context._inherit_global_context is True
    # Installed: Langfuse clients created from now on are bridged.
    assert LangfuseResourceManager._initialize_instance is not original_initialize_instance


def test_explicit_instruments_are_honored_with_a_warning(
    init_tracing: MagicMock, logs: Callable[[], list[str]]
):
    _initialize(bridges=["langfuse"], instruments={Instruments.OPENAI})

    assert _kwargs(init_tracing)["block_instruments"] is None
    assert any("OPENAI" in m for m in logs())


def test_explicit_global_provider_is_overridden_with_a_warning(
    init_tracing: MagicMock, logs: Callable[[], list[str]]
):
    _initialize(bridges=["langfuse"], set_global_tracer_provider=True)

    assert _kwargs(init_tracing)["set_global_tracer_provider"] is False
    assert any("set_global_tracer_provider" in m for m in logs())


def test_unavailable_bridge_changes_nothing_in_laminar(
    init_tracing: MagicMock, logs: Callable[[], list[str]], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(LangfuseBridge, "is_available", lambda _self: False)

    _initialize(bridges=["langfuse"])

    kwargs = _kwargs(init_tracing)
    assert kwargs["block_instruments"] is None
    assert kwargs["set_global_tracer_provider"] is True
    assert laminar_context._inherit_global_context is False
    assert any("langfuse bridge needs langfuse >= 3.0" in m for m in logs())


def test_unknown_bridge_is_rejected_before_initializing(init_tracing: MagicMock):
    with pytest.raises(ValueError, match="Unknown Laminar bridge 'langfsue'"):
        _initialize(bridges=["langfsue"])
    assert not Laminar.is_initialized()
    init_tracing.assert_not_called()
