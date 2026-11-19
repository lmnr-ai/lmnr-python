from __future__ import annotations

from collections.abc import Callable, Iterator
from importlib.metadata import version

import pytest

from lmnr import McpOptions
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp import context_field
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp.turns import TURNS
from lmnr.opentelemetry_lib.tracing.mcp_options import reset_mcp_options, set_mcp_options

# The dev dependency pins mcp 2.x; CI runs the 1.x modules under a
# `uv run --with mcp==1.x` overlay. Neither imports under the other major.
collect_ignore = (
    ["test_mcp_v1.py", "test_mcp_mode_v1.py"]
    if int(version("mcp").split(".")[0]) >= 2
    else ["test_mcp_v2.py", "test_mcp_mode_v2.py"]
)

SetMcpMode = Callable[[bool | McpOptions | None], None]


@pytest.fixture
def mcp_mode(monkeypatch: pytest.MonkeyPatch) -> Iterator[SetMcpMode]:
    """MCP observability mode on, as `Laminar.initialize(mcp=True)`; call it to change it.

    Also starts from fresh process-wide turn and own-`context` state.
    """
    # tests set `TURNS._clock`; restored on teardown
    monkeypatch.setattr(TURNS, "_clock", TURNS._clock)
    TURNS.clear()
    monkeypatch.setattr(context_field, "_own_context_tools", set[str]())
    monkeypatch.setattr(context_field, "_warned_tools", set[str]())
    set_mcp_options(True)
    yield set_mcp_options
    reset_mcp_options()
