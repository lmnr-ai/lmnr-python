import asyncio
import atexit
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, cast
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry import trace
from opentelemetry.context import get_value, set_value
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.types import AttributeValue

from lmnr import LaminarSpanProcessor
from lmnr.opentelemetry_lib.tracing.context import (
    CONTEXT_METADATA_KEY,
    attach_context,
    detach_context,
    get_current_context,
)
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
from lmnr.sdk.client.synchronous.sync_client import LaminarClient
from lmnr.sdk.debug import (
    DebugRuntime,
    get_runtime,
    init_debug_runtime,
    init_debug_runtime_from_context,
    reset_debug_runtime,
)
from lmnr.sdk.debug.config import DebugConfig, build_debug_config_from_context
from lmnr.sdk.debug.outcome import CacheOutcome
from lmnr.sdk.laminar import Laminar
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.types import DebugContext, LaminarSpanContext


@pytest.fixture
def no_browser(monkeypatch: MagicMock):
    """Prevent _init_debug_runtime from opening a browser tab during tests."""
    monkeypatch.setenv("LMNR_DEBUG_SESSION_ID", "test-session")


def _reset_runtime():
    reset_debug_runtime()


class _NoopExporter:
    def export(self, _spans: list[ReadableSpan]) -> Literal[SpanExportResult.SUCCESS]:
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

    def force_flush(self, _timeout_millis: int=30000):
        return True


class _FakeSpan:
    """Minimal span double for `LaminarSpanProcessor.on_start`."""

    def __init__(self, name: str = "openai.chat", trace_id: int = 1):
        self.parent: Any = None
        self.name: str = name
        self.attributes: dict[str, AttributeValue] = {}
        self._ctx: trace.SpanContext = trace.SpanContext(
            trace_id=trace_id,
            span_id=0x0123456789ABCDEF,
            is_remote=False,
        )

    def get_span_context(self) -> trace.SpanContext:
        return self._ctx

    def set_attribute(self, key: str, value: AttributeValue):
        self.attributes[key] = value


def _config(**kwargs: Any) -> DebugConfig:
    base = {
        "session_id": "s",
        "replay_trace_id": "r",
        "cache_until_span_id": "abcdef",
    }
    base.update(kwargs)
    return DebugConfig(**cast(Any, base))


@pytest.fixture
def make_runtime(sync_client: LaminarClient, async_client: AsyncLaminarClient) -> Callable[..., DebugRuntime]:
    """Factory for `DebugRuntime`s sharing the per-test clients above."""

    def _make(*, debugger_url: str | None = None, **cfg: Any) -> DebugRuntime:
        return DebugRuntime(_config(**cfg), sync_client, async_client, debugger_url)

    return _make


def test_replay_configured_reflects_config(make_runtime: Callable[..., DebugRuntime]):
    # v2 has no synchronous cache build, so replay_configured collapses to the
    # config's replay_enabled (source trace + cache_until span-id needle).
    assert make_runtime().replay_configured is True
    assert make_runtime(replay_trace_id=None).replay_configured is False
    assert make_runtime(cache_until_span_id=None).replay_configured is False


def test_runtime_retains_both_clients(
    make_runtime: Callable[..., DebugRuntime],
    sync_client: LaminarClient,
    async_client: AsyncLaminarClient,
):
    runtime = make_runtime()
    assert runtime.client is sync_client
    assert runtime.async_client is async_client


def test_update_context_config_moves_coordinates_keeps_identity(make_runtime: Callable[..., DebugRuntime]):
    # The dynamic coordinates (session / replay / cache-until) follow the live
    # request; local_origin / session_minted are run identity and must NOT change.
    runtime = make_runtime(
        session_id="sess-a",
        replay_trace_id="trace-a",
        cache_until_span_id="aaaa",
        local_origin=False,
        session_minted=False,
    )
    changed = runtime.update_context_config(
        _config(
            session_id="sess-b",
            replay_trace_id="trace-b",
            cache_until_span_id="bbbb",
            # Deliberately different identity flags — they must be ignored.
            local_origin=True,
            session_minted=True,
        )
    )
    assert changed is True
    assert runtime.session_id == "sess-b"
    assert runtime.replay_trace_id == "trace-b"
    assert runtime.cache_until_span_id == "bbbb"
    # Identity is preserved: a downstream run stays downstream.
    assert runtime.local_origin is False
    assert runtime.should_open_browser is False


def test_update_context_config_reports_change_for_moved_replay_coords(make_runtime: Callable[..., DebugRuntime]):
    # The flag tracks ANY moved dynamic coordinate (per-run), not only the
    # session id: a replay-trace change on the same session still reports True.
    runtime = make_runtime(session_id="sess", local_origin=False)
    changed = runtime.update_context_config(
        _config(session_id="sess", replay_trace_id="other", local_origin=False)
    )
    assert changed is True
    # Replay coords still move even when the session id is unchanged.
    assert runtime.replay_trace_id == "other"


def test_record_trace_id_first_wins(make_runtime: Callable[..., DebugRuntime]):
    runtime = make_runtime()
    runtime.record_trace_id("trace-a")
    runtime.record_trace_id("trace-b")
    assert runtime._trace_id == "trace-a"


def test_record_project_id_first_wins(make_runtime: Callable[..., DebugRuntime]):
    runtime = make_runtime(debugger_url="https://x")
    runtime.record_project_id("proj-a")
    runtime.record_project_id("proj-b")
    assert runtime._project_id == "proj-a"


def test_debugger_session_url_falls_back_to_base_without_project_id(make_runtime: Callable[..., DebugRuntime]):
    # Before register resolves a project id, the URL is just the base.
    runtime = make_runtime(debugger_url="https://app.x")
    assert runtime.debugger_session_url() == "https://app.x"


def test_debugger_session_url_is_none_without_base(make_runtime: Callable[..., DebugRuntime]):
    runtime = make_runtime(debugger_url=None)
    assert runtime.debugger_session_url() is None


def test_debugger_session_url_full_with_project_id(make_runtime: Callable[..., DebugRuntime]):
    runtime = make_runtime(session_id="sess-1", debugger_url="https://app.x")
    runtime.record_project_id("proj-1")
    assert runtime.debugger_session_url() == (
        "https://app.x/project/proj-1/debugger-sessions/sess-1"
    )


def test_record_debug_trace_id_from_env_populates_pointer(\
    monkeypatch: MagicMock,
    make_runtime: Callable[..., DebugRuntime]
):
    # A run attached via LMNR_SPAN_CONTEXT never opens a root span, so the
    # pointer would emit an empty trace_id unless the inherited trace id is
    # recorded at env-attach time.


    _reset_runtime()
    runtime = make_runtime()
    monkeypatch.setattr("lmnr.sdk.debug._runtime", runtime)

    trace_id = uuid.UUID("01234567-89ab-cdef-0123-456789abcdef")
    span_context = trace.SpanContext(
        trace_id=trace_id.int,
        span_id=0x0123456789ABCDEF,
        is_remote=True,
    )
    Laminar._record_debug_trace_id_from_env(span_context)

    assert runtime._trace_id == str(trace_id)
    _reset_runtime()


def test_env_context_arms_debug_runtime_from_block(monkeypatch: MagicMock):
    # A debug block carried by LMNR_SPAN_CONTEXT must arm the debug runtime: an
    # LMNR_SPAN_CONTEXT-attached run parents off the pushed context with
    # parent_span_context=None, so the span-creation funnels never see the block
    # and only _initialize_context_from_env can activate replay downstream.
    _reset_runtime()

    session_id = str(uuid.uuid4())
    ctx = LaminarSpanContext(
        trace_id=uuid.UUID("01234567-89ab-cdef-0123-456789abcdef"),
        span_id=uuid.UUID("00000000-0000-0000-0123-456789abcdef"),
        debug=DebugContext(enabled=True, session_id=session_id),
    )

    armed_with = {}

    def _fake_arm(debug: Any):
        armed_with["debug"] = debug

    monkeypatch.setattr(Laminar, "_arm_debug_runtime_from_context", _fake_arm)
    monkeypatch.setenv("LMNR_SPAN_CONTEXT", str(ctx))

    Laminar._initialize_context_from_env()

    assert "debug" in armed_with
    assert armed_with["debug"] is not None
    assert armed_with["debug"]["enabled"] is True
    assert armed_with["debug"]["session_id"] == session_id
    _reset_runtime()


def test_processor_records_trace_id_when_tracing_disabled(
    monkeypatch: MagicMock,
    make_runtime: Callable[..., DebugRuntime]
):
    # Even with LMNR_DISABLE_TRACING=true the processor must record the root
    # trace id, otherwise the shutdown pointer emits an empty trace_id while
    # replay (gated only on get_runtime() is not None) may still be active.
    _reset_runtime()
    runtime = make_runtime()
    monkeypatch.setattr("lmnr.sdk.debug._runtime", runtime)
    monkeypatch.setenv("LMNR_DISABLE_TRACING", "true")

    trace_id = uuid.UUID("01234567-89ab-cdef-0123-456789abcdef")

    processor = LaminarSpanProcessor(exporter=cast(Any, _NoopExporter()), disable_batch=True)
    processor.on_start(cast(Any, _FakeSpan("root", trace_id.int)))

    assert runtime._trace_id == str(trace_id)
    _reset_runtime()


def test_processor_keeps_real_span_path_for_replay_when_disabled(
    monkeypatch: MagicMock,
    make_runtime: Callable[..., DebugRuntime]
):
    # With replay active, LMNR_DISABLE_TRACING=true must NOT mask span names to
    # "_" in lmnr.span.path: the replay wrapper reads that in-process path to
    # match the cache (keyed on the source trace's real dotted paths). Masking
    # would never match, so replay would silently run live.
    _reset_runtime()
    # replay_trace_id + cache_until span-id => replay_configured is True.
    runtime = make_runtime()
    monkeypatch.setattr("lmnr.sdk.debug._runtime", runtime)
    monkeypatch.setenv("LMNR_DISABLE_TRACING", "true")

    span = _FakeSpan(trace_id=uuid.UUID(int=1).int)
    processor = LaminarSpanProcessor(exporter=cast(Any, _NoopExporter()), disable_batch=True)
    processor.on_start(cast(Any, span))

    assert span.attributes["lmnr.span.path"] == ["openai.chat"]
    _reset_runtime()


def test_processor_masks_span_path_when_disabled_without_replay(monkeypatch: MagicMock):
    # No debug runtime: disabled tracing still masks span names to "_" (privacy).
    _reset_runtime()
    monkeypatch.setenv("LMNR_DISABLE_TRACING", "true")

    span = _FakeSpan(trace_id=uuid.UUID(int=2).int)
    processor = LaminarSpanProcessor(exporter=cast(Any, _NoopExporter()), disable_batch=True)
    processor.on_start(cast(Any, span))

    assert span.attributes["lmnr.span.path"] == ["_"]
    _reset_runtime()


def test_record_debug_trace_id_from_env_noop_without_runtime():
    from lmnr.sdk.laminar import Laminar

    _reset_runtime()
    span_context = trace.SpanContext(
        trace_id=0x0123456789ABCDEF0123456789ABCDEF,
        span_id=0x0123456789ABCDEF,
        is_remote=True,
    )
    # No runtime registered -> silent no-op, never raises.
    Laminar._record_debug_trace_id_from_env(span_context)


def test_emit_pointer_uses_construction_time_started_at(
    tmp_path: Path,
    monkeypatch: MagicMock,
    capsys: MagicMock,
    make_runtime: Callable[..., DebugRuntime],
):
    # started_at must reflect when the run began (runtime construction at SDK
    # init), not when the pointer is emitted (shutdown). Sleep between the two so
    # an emit-time timestamp would differ from the captured one.
    monkeypatch.chdir(tmp_path)
    runtime = make_runtime()
    captured = runtime._started_at
    time.sleep(0.01)
    runtime.record_trace_id("trace-a")
    runtime.emit_pointer()

    line = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("LMNR_DEBUG_RUN ")
    )
    payload = json.loads(line[len("LMNR_DEBUG_RUN ") :])
    assert payload["started_at"] == captured


def test_emit_pointer_persists_cache_until_span_id(
    tmp_path: Path,
    monkeypatch: MagicMock,
    capsys: MagicMock,
    make_runtime: Callable[..., DebugRuntime],
):
    # v2 persists the raw span-id needle in the record's cache_until (no
    # resolution step), so a later replay re-sends the needle.
    import json

    monkeypatch.chdir(tmp_path)
    runtime = make_runtime(cache_until_span_id="0123456789abcdef")
    runtime.record_trace_id("trace-a")
    runtime.emit_pointer()

    line = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("LMNR_DEBUG_RUN ")
    )
    payload = json.loads(line[len("LMNR_DEBUG_RUN ") :])
    assert payload["cache_until"] == "0123456789abcdef"


def test_emit_pointer_only_once(
    tmp_path: Path,
    monkeypatch: MagicMock,
    capsys: MagicMock,
    make_runtime: Callable[..., DebugRuntime]
):
    monkeypatch.chdir(tmp_path)
    runtime = make_runtime()
    runtime.record_trace_id("trace-a")
    runtime.emit_pointer()
    runtime.emit_pointer()
    lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("LMNR_DEBUG_RUN ")
    ]
    assert len(lines) == 1


def test_emit_pointer_noop_for_downstream_run(
    tmp_path: Path,
    monkeypatch: MagicMock,
    capsys: MagicMock,
    make_runtime: Callable[..., DebugRuntime],
):
    # A runtime armed from a propagated DebugContext (local_origin=False) joins
    # the upstream replay session and must NOT write a run pointer — the origin
    # owns it. Gated inside emit_pointer so shutdown()/atexit stay safe.
    monkeypatch.chdir(tmp_path)
    runtime = make_runtime(local_origin=False)
    runtime.record_trace_id("trace-downstream")
    runtime.emit_pointer()

    lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("LMNR_DEBUG_RUN ")
    ]
    assert lines == []
    assert not (tmp_path / ".lmnr" / "debug-session.json").exists()


def test_emit_pointer_uses_full_debugger_url(
    tmp_path: Path,
    monkeypatch: MagicMock,
    capsys: MagicMock,
    make_runtime: Callable[..., DebugRuntime],
):
    # The pointer's debugger_url must carry the SAME full per-session URL the
    # console prints, not just the base — built via the shared
    # debugger_session_url code path once the project id is recorded.
    import json

    monkeypatch.chdir(tmp_path)
    runtime = make_runtime(session_id="sess-1", debugger_url="https://app.x")
    runtime.record_project_id("proj-1")
    runtime.record_trace_id("trace-a")
    runtime.emit_pointer()

    line = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("LMNR_DEBUG_RUN ")
    )
    payload = json.loads(line[len("LMNR_DEBUG_RUN ") :])
    assert payload["debugger_url"] == (
        "https://app.x/project/proj-1/debugger-sessions/sess-1"
    )


def test_init_disabled_returns_none(monkeypatch: MagicMock, sync_client: LaminarClient, async_client: AsyncLaminarClient):
    _reset_runtime()
    monkeypatch.delenv("LMNR_DEBUG", raising=False)
    assert init_debug_runtime(client=sync_client, async_client=async_client) is None
    assert get_runtime() is None


def test_init_debug_runtime_skips_client_when_debug_off(monkeypatch: MagicMock):
    # When LMNR_DEBUG is off, Laminar._init_debug_runtime must NOT construct a
    # LaminarClient (and its httpx.Client) — that would leak unclosed on every
    # normal initialize().
    from lmnr.sdk.laminar import Laminar

    _reset_runtime()
    monkeypatch.delenv("LMNR_DEBUG", raising=False)

    constructed: list[tuple[Sequence[Any], dict[str, Any]]] = []

    class _SpyClient:
        def __init__(self, *args: Any, **kwargs: Any):
            constructed.append((args, kwargs))

    monkeypatch.setattr(
        "lmnr.sdk.client.synchronous.sync_client.LaminarClient", _SpyClient
    )
    monkeypatch.setattr(Laminar, "_Laminar__project_api_key", "k", raising=False)

    Laminar._init_debug_runtime(base_url="http://localhost", http_port=8000)

    assert constructed == []
    assert get_runtime() is None
    _reset_runtime()


def test_init_off_does_not_latch_flag(
    monkeypatch: MagicMock,
    sync_client: LaminarClient,
    async_client: AsyncLaminarClient
):
    # When debug is off, init must NOT spend the one-shot flag: a later init
    # after the env flips LMNR_DEBUG on must still build a runtime without an
    # intervening reset_debug_runtime().
    _reset_runtime()
    monkeypatch.delenv("LMNR_DEBUG", raising=False)
    assert init_debug_runtime(client=sync_client, async_client=async_client) is None

    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.delenv("LMNR_DEBUG_REPLAY_TRACE_ID", raising=False)
    monkeypatch.delenv("LMNR_DEBUG_CACHE_UNTIL", raising=False)
    runtime = init_debug_runtime(client=sync_client, async_client=async_client)
    assert runtime is not None
    assert get_runtime() is runtime
    _reset_runtime()


def test_init_is_idempotent(
    monkeypatch: MagicMock,
    sync_client: LaminarClient,
    async_client: AsyncLaminarClient
):
    _reset_runtime()
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.delenv("LMNR_DEBUG_REPLAY_TRACE_ID", raising=False)
    monkeypatch.delenv("LMNR_DEBUG_CACHE_UNTIL", raising=False)
    first = init_debug_runtime(client=sync_client, async_client=async_client)
    second = init_debug_runtime(client=sync_client, async_client=async_client)
    assert first is second is get_runtime()
    _reset_runtime()


def test_reset_allows_reinit_to_reread_env(
    monkeypatch: MagicMock,
    sync_client: LaminarClient,
    async_client: AsyncLaminarClient
):
    # A shutdown/initialize cycle must re-read LMNR_DEBUG*: reset clears the
    # one-shot flag so a previously-off run can turn debug on (and vice versa).
    _reset_runtime()
    monkeypatch.delenv("LMNR_DEBUG", raising=False)
    assert init_debug_runtime(client=sync_client, async_client=async_client) is None

    reset_debug_runtime()
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.delenv("LMNR_DEBUG_REPLAY_TRACE_ID", raising=False)
    monkeypatch.delenv("LMNR_DEBUG_CACHE_UNTIL", raising=False)
    runtime = init_debug_runtime(client=sync_client, async_client=async_client)
    assert runtime is not None
    assert get_runtime() is runtime
    _reset_runtime()


def test_init_builds_replay_runtime_from_env(
    monkeypatch: MagicMock,
    sync_client: LaminarClient,
    async_client: AsyncLaminarClient
):
    # A replay-configured run (source trace + cache_until span id) builds a
    # runtime that reports replay configured. v2 builds no cache synchronously.
    _reset_runtime()
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.setenv("LMNR_DEBUG_REPLAY_TRACE_ID", "trace-1")
    monkeypatch.setenv("LMNR_DEBUG_CACHE_UNTIL", "0123-456789abcdef")

    runtime = init_debug_runtime(
        client=sync_client, async_client=async_client, debugger_url="https://www.lmnr.ai"
    )
    assert runtime is not None
    assert runtime.replay_configured is True
    assert runtime.replay_trace_id == "trace-1"
    assert runtime.cache_until_span_id == "0123456789abcdef"
    _reset_runtime()


class _SpyRolloutSessions:
    def __init__(self, raises: bool = False, project_id: str | None = None):
        self.registered: list[tuple[str, str | None]] = []
        self._raises: bool= raises
        self._project_id: str | None = project_id

    def register(self, session_id: str, name: str | None=None) -> str | None:
        self.registered.append((session_id, name))
        if self._raises:
            raise RuntimeError("backend down")
        return self._project_id

    def cache(self, **kwargs: Any):
        return CacheOutcome(kind="live")


class _SpyDebugClient:
    def __init__(
        self,
        *args: Any,
        raises: bool = False,
        project_id: str | None = None,
        **kwargs: Any
    ):
        self.rollout_sessions: _SpyRolloutSessions = _SpyRolloutSessions(
            raises=raises, project_id=project_id
        )
        self.closed: bool = False

    def close(self):
        self.closed = True


class _SpyAsyncDebugClient:
    def __init__(self, *args: Any, **kwargs: Any):
        self.rollout_sessions: _SpyRolloutSessions = _SpyRolloutSessions()
        self.closed: bool = False

    async def close(self):
        self.closed = True


def _patch_clients(
    monkeypatch: MagicMock,
    sync_client: _SpyDebugClient,
    async_client: _SpyAsyncDebugClient | None = None
):
    """Patch both retained client classes used by _init_debug_runtime."""
    monkeypatch.setattr(
        "lmnr.sdk.client.synchronous.sync_client.LaminarClient",
        lambda *a, **k: sync_client,  # pyright: ignore[reportUnknownLambdaType]
    )
    monkeypatch.setattr(
        "lmnr.sdk.client.asynchronous.async_client.AsyncLaminarClient",
        lambda *a, **k: async_client or _SpyAsyncDebugClient(),  # pyright: ignore[reportUnknownLambdaType]
    )


def test_init_registers_session_with_backend(monkeypatch: MagicMock):
    # A bare LMNR_DEBUG=true run must POST its SDK-minted session id to the
    # backend so the session shows up in the UI.
    _reset_runtime()
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.delenv("LMNR_DEBUG_REPLAY_TRACE_ID", raising=False)
    monkeypatch.delenv("LMNR_DEBUG_CACHE_UNTIL", raising=False)

    spy = _SpyDebugClient()
    _patch_clients(monkeypatch, spy)
    monkeypatch.setattr(Laminar, "_Laminar__project_api_key", "k", raising=False)

    Laminar._init_debug_runtime(base_url="http://localhost", http_port=8000)

    runtime = get_runtime()
    assert runtime is not None
    assert spy.rollout_sessions.registered == [(runtime.session_id, None)]
    # v2 RETAINS the cache clients for the run's lifetime (the provider wrappers
    # hit the cache endpoint on every live call); they are closed at shutdown,
    # NOT at init. So the sync client must still be open here.
    assert spy.closed is False
    assert runtime.client is spy
    _reset_runtime()


def test_init_logs_debugger_url_when_project_id_returned(
    no_browser: Any,
    monkeypatch: MagicMock,
    caplog: MagicMock,
):
    # When the backend returns a project id, init must log the human-facing
    # debugger session URL at INFO, respecting LMNR_FRONTEND_URL.
    _reset_runtime()
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.setenv("LMNR_FRONTEND_URL", "https://app.example.com")
    monkeypatch.delenv("LMNR_DEBUG_REPLAY_TRACE_ID", raising=False)
    monkeypatch.delenv("LMNR_DEBUG_CACHE_UNTIL", raising=False)

    spy = _SpyDebugClient(project_id="proj-123")
    _patch_clients(monkeypatch, spy)
    monkeypatch.setattr(Laminar, "_Laminar__project_api_key", "k", raising=False)

    # Laminar's loggers set propagate=False (they attach their own handler, so
    # propagating would double-emit through the app's root handler), and caplog
    # only sees records that reach the root. Re-enable it just for this assertion.
    laminar_logger = get_default_logger("lmnr.sdk.laminar")
    monkeypatch.setattr(laminar_logger, "propagate", True)

    with caplog.at_level(logging.INFO, logger="lmnr.sdk.laminar"):
        Laminar._init_debug_runtime(base_url="http://localhost", http_port=8000)

    runtime = get_runtime()
    assert runtime is not None
    expected = (
        f"https://app.example.com/project/proj-123"
        f"/debugger-sessions/{runtime.session_id}"
    )
    assert any(expected in record.getMessage() for record in caplog.records)
    _reset_runtime()


def test_init_survives_registration_failure(monkeypatch: MagicMock):
    # Registration is best-effort: a backend error must never crash init.
    from lmnr.sdk.laminar import Laminar

    _reset_runtime()
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.delenv("LMNR_DEBUG_REPLAY_TRACE_ID", raising=False)
    monkeypatch.delenv("LMNR_DEBUG_CACHE_UNTIL", raising=False)

    spy = _SpyDebugClient(raises=True)
    _patch_clients(monkeypatch, spy)
    monkeypatch.setattr(Laminar, "_Laminar__project_api_key", "k", raising=False)

    Laminar._init_debug_runtime(base_url="http://localhost", http_port=8000)

    runtime = get_runtime()
    assert runtime is not None  # init still completed
    assert len(spy.rollout_sessions.registered) == 1
    _reset_runtime()


def test_init_does_not_build_debug_runtime_when_tracing_fails(monkeypatch: MagicMock):
    # If init_tracing() raises, initialize() must abort BEFORE any debug
    # side effects: no backend session registration and no debug runtime left
    # live on a process whose tracing never came up.

    _reset_runtime()
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.delenv("LMNR_DEBUG_REPLAY_TRACE_ID", raising=False)
    monkeypatch.delenv("LMNR_DEBUG_CACHE_UNTIL", raising=False)

    spy = _SpyDebugClient()
    _patch_clients(monkeypatch, spy)

    def _boom(*args: Any, **kwargs: Any):
        raise RuntimeError("tracer down")

    monkeypatch.setattr("lmnr.sdk.laminar.init_tracing", _boom)
    monkeypatch.setattr(Laminar, "_Laminar__initialized", False, raising=False)

    with patch.dict(os.environ, {"LMNR_PROJECT_API_KEY": "k"}):
        try:
            Laminar.initialize(project_api_key="k")
        except RuntimeError:
            pass

    # Tracer init failed before _init_debug_runtime ran: no runtime, no session.
    assert get_runtime() is None
    assert spy.rollout_sessions.registered == []
    _reset_runtime()
    monkeypatch.setattr(Laminar, "_Laminar__initialized", False, raising=False)


def test_initialize_captures_debug_connection_args_before_marking_initialized(
    monkeypatch: MagicMock,
):
    # The from-context arm path (_arm_debug_runtime_from_context) builds its own
    # cache clients from __base_url_for_debug / __http_port_for_debug. Those are
    # also set inside _init_debug_runtime, but that runs AFTER initialize() flips
    # __initialized (and after init_tracing) — so a span arriving with a
    # propagated debug block in that window would read None and target the
    # default base URL, then first-wins would pin the mis-targeted runtime.
    # initialize() must therefore capture the args itself, BEFORE __initialized.
    # We stub _init_debug_runtime to a no-op so ONLY the initialize()-level
    # capture can populate the fields, and assert they hold the parsed values.
    _reset_runtime()
    monkeypatch.setattr(Laminar, "_Laminar__initialized", False, raising=False)
    monkeypatch.setattr(Laminar, "_Laminar__base_url_for_debug", None, raising=False)
    monkeypatch.setattr(Laminar, "_Laminar__http_port_for_debug", None, raising=False)
    monkeypatch.setattr(
        "lmnr.sdk.laminar.init_tracing", lambda *a, **k: None  # pyright: ignore[reportUnknownLambdaType]
    )
    # No-op the debug-runtime build so the only thing that can set the static
    # connection fields is the capture in initialize() itself.
    monkeypatch.setattr(
        Laminar, "_init_debug_runtime", classmethod(lambda cls, **k: None)  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    )

    with patch.dict(os.environ, {"LMNR_PROJECT_API_KEY": "k"}, clear=True):
        Laminar.initialize(
            project_api_key="k",
            base_url="https://custom.example.com",
            http_port=1234,
        )

    assert Laminar._Laminar__base_url_for_debug == "https://custom.example.com"  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    assert Laminar._Laminar__http_port_for_debug == 1234  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    monkeypatch.setattr(Laminar, "_Laminar__initialized", False, raising=False)
    _reset_runtime()


def test_exit_hook_does_not_accumulate_across_cycles(tmp_path: Path, monkeypatch: MagicMock):
    # atexit holds a strong ref to whatever it registers, so each debug-mode
    # init must unregister the previous pointer hook on shutdown — otherwise an
    # init/shutdown loop pins every retired DebugRuntime alive and leaks one
    # atexit handler per cycle.
    _reset_runtime()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.delenv("LMNR_DEBUG_REPLAY_TRACE_ID", raising=False)
    monkeypatch.delenv("LMNR_DEBUG_CACHE_UNTIL", raising=False)

    registered: list[Callable[..., Any]] = []
    monkeypatch.setattr(atexit, "register", lambda fn, *a, **k: registered.append(fn))  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    monkeypatch.setattr(
        atexit,
        "unregister",
        lambda fn: registered.remove(fn) if fn in registered else None,  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    )

    _patch_clients(monkeypatch, _SpyDebugClient())
    monkeypatch.setattr(Laminar, "_Laminar__project_api_key", "k", raising=False)
    monkeypatch.setattr("lmnr.sdk.laminar.shutdown_tracing", lambda: None)

    for _ in range(12):
        Laminar._init_debug_runtime(base_url="http://localhost", http_port=8000)
        monkeypatch.setattr(Laminar, "_Laminar__initialized", True, raising=False)
        Laminar.shutdown()

    assert registered == []
    _reset_runtime()
    monkeypatch.setattr(Laminar, "_Laminar__initialized", False, raising=False)


def test_shutdown_closes_retained_clients(tmp_path: Path, monkeypatch: MagicMock):
    # v2 keeps both cache clients open for the run; shutdown must close both so
    # their httpx connection pools aren't leaked across init/shutdown cycles.
    _reset_runtime()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.delenv("LMNR_DEBUG_REPLAY_TRACE_ID", raising=False)
    monkeypatch.delenv("LMNR_DEBUG_CACHE_UNTIL", raising=False)

    sync_spy = _SpyDebugClient()
    async_spy = _SpyAsyncDebugClient()
    _patch_clients(monkeypatch, sync_spy, async_spy)
    monkeypatch.setattr(Laminar, "_Laminar__project_api_key", "k", raising=False)
    monkeypatch.setattr("lmnr.sdk.laminar.shutdown_tracing", lambda: None)

    Laminar._init_debug_runtime(base_url="http://localhost", http_port=8000)
    monkeypatch.setattr(Laminar, "_Laminar__initialized", True, raising=False)

    Laminar.shutdown()

    assert sync_spy.closed is True
    assert async_spy.closed is True
    _reset_runtime()
    monkeypatch.setattr(Laminar, "_Laminar__initialized", False, raising=False)


def test_shutdown_resets_run_live_latch(tmp_path: Path, monkeypatch: MagicMock):
    # A MISS latches the process-wide run-live flag; shutdown must clear it so a
    # fresh debug run in the same process starts from a clean cache state.
    _reset_runtime()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.delenv("LMNR_DEBUG_REPLAY_TRACE_ID", raising=False)
    monkeypatch.delenv("LMNR_DEBUG_CACHE_UNTIL", raising=False)

    _patch_clients(monkeypatch, _SpyDebugClient())
    monkeypatch.setattr(Laminar, "_Laminar__project_api_key", "k", raising=False)
    monkeypatch.setattr("lmnr.sdk.laminar.shutdown_tracing", lambda: None)

    Laminar._init_debug_runtime(base_url="http://localhost", http_port=8000)
    monkeypatch.setattr(Laminar, "_Laminar__initialized", True, raising=False)
    Laminar.set_debug_run_live(True)
    assert Laminar.is_debug_run_live() is True

    Laminar.shutdown()

    assert Laminar.is_debug_run_live() is False
    _reset_runtime()
    monkeypatch.setattr(Laminar, "_Laminar__initialized", False, raising=False)


def test_shutdown_completes_cleanup_when_emit_pointer_raises(tmp_path: Path, monkeypatch: MagicMock):
    # emit_pointer prints to stdout, which can raise OSError/BrokenPipeError
    # (closed stdout in daemons/containers, notebook kernel restarts). That must
    # never abort shutdown's cleanup: shutdown_tracing(), the reset, and
    # the __initialized flip must still run.
    from lmnr.sdk.laminar import Laminar

    _reset_runtime()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.delenv("LMNR_DEBUG_REPLAY_TRACE_ID", raising=False)
    monkeypatch.delenv("LMNR_DEBUG_CACHE_UNTIL", raising=False)

    _patch_clients(monkeypatch, _SpyDebugClient())
    monkeypatch.setattr(Laminar, "_Laminar__project_api_key", "k", raising=False)

    shutdown_calls: list[bool] = []
    monkeypatch.setattr(
        "lmnr.sdk.laminar.shutdown_tracing",
        lambda: shutdown_calls.append(True),
    )

    Laminar._init_debug_runtime(base_url="http://localhost", http_port=8000)
    monkeypatch.setattr(Laminar, "_Laminar__initialized", True, raising=False)

    runtime = get_runtime()
    assert runtime is not None
    # Simulate a broken stdout: emit_pointer raises.
    monkeypatch.setattr(runtime, "emit_pointer", _raise_broken_pipe)

    # Must not propagate the BrokenPipeError out of shutdown().
    Laminar.shutdown()

    # The cleanup after emit_pointer still ran.
    assert shutdown_calls == [True]
    assert Laminar.is_initialized() is False
    assert get_runtime() is None
    _reset_runtime()
    monkeypatch.setattr(Laminar, "_Laminar__initialized", False, raising=False)


def _raise_broken_pipe():
    raise BrokenPipeError("stdout closed")


def test_arm_from_context_closes_clients_when_losing_race(monkeypatch: MagicMock):
    # _arm_debug_runtime_from_context allocates fresh sync/async clients BEFORE
    # consulting init_debug_runtime_from_context, which is first-wins. Under a
    # concurrent arm, both callers pass the get_runtime() fast path, both
    # allocate, and one loses inside init_*_from_context — getting back a runtime
    # that retains the WINNER's clients. The loser's freshly-allocated clients
    # are orphaned and must be closed, or their httpx pools leak. We simulate the
    # lost race by patching init_*_from_context to return a winner runtime built
    # from different clients (get_runtime() stays None at the fast-path check).
    _reset_runtime()
    SESSION = "00000000-0000-0000-0000-0000000000aa"
    block = DebugContext(enabled=True, session_id=SESSION)

    # The winner runtime (built by the thread that won the race) retains its own
    # clients; _arm_debug_runtime_from_context must NOT close these.
    winner_sync = _SpyDebugClient()
    winner_async = _SpyAsyncDebugClient()
    winner = DebugRuntime(
        DebugConfig(session_id=SESSION, replay_trace_id=None, local_origin=False),
        cast(Any, winner_sync),
        cast(Any, winner_async),
        None,
    )

    def _fake_init_from_context(
        dbg: Any,
        client: LaminarClient,
        async_client: AsyncLaminarClient,
        debugger_url: str | None = None
    ) -> tuple[DebugRuntime, bool]:
        # Mimic the lost-race outcome: return the winner runtime, ignoring the
        # clients this caller passed (init is first-arm and already armed). The
        # session changed (a runtime was just published), so report True.
        return winner, True

    monkeypatch.setattr(
        "lmnr.sdk.debug.init_debug_runtime_from_context", _fake_init_from_context
    )

    # _arm_debug_runtime_from_context allocates these (the loser's clients).
    loser_sync = _SpyDebugClient()
    loser_async = _SpyAsyncDebugClient()
    _patch_clients(monkeypatch, loser_sync, loser_async)
    monkeypatch.setattr(Laminar, "_Laminar__project_api_key", "k", raising=False)
    monkeypatch.setattr(
        Laminar, "_Laminar__base_url_for_debug", "http://localhost", raising=False
    )
    monkeypatch.setattr(Laminar, "_Laminar__http_port_for_debug", 8000, raising=False)

    Laminar._arm_debug_runtime_from_context(block)

    # The loser's freshly-allocated clients are closed (not leaked); the winner's
    # clients stay open (the winner owns the run).
    assert loser_sync.closed is True
    assert loser_async.closed is True
    assert winner_sync.closed is False
    assert winner_async.closed is False
    _reset_runtime()


def test_init_closes_clients_when_runtime_already_armed_from_context(monkeypatch: MagicMock):
    # _init_debug_runtime (the env path) allocates fresh sync/async clients
    # BEFORE consulting init_debug_runtime, which is first-wins. If a propagated
    # DebugContext armed the runtime first (deep in span creation, before
    # initialize() ran to completion), init_debug_runtime returns the EXISTING
    # runtime retaining ITS clients — so the env path's freshly-allocated clients
    # are orphaned and must be closed, or their httpx pools leak. We simulate this
    # by patching init_debug_runtime to return a runtime built from different
    # clients.
    from lmnr.sdk.debug.config import DebugConfig
    from lmnr.sdk.laminar import Laminar

    _reset_runtime()
    SESSION = "00000000-0000-0000-0000-0000000000aa"
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.delenv("LMNR_DEBUG_REPLAY_TRACE_ID", raising=False)
    monkeypatch.delenv("LMNR_DEBUG_CACHE_UNTIL", raising=False)

    # The runtime armed earlier from a propagated context retains its own
    # clients; _init_debug_runtime must NOT close these.
    existing_sync = _SpyDebugClient()
    existing_async = _SpyAsyncDebugClient()
    existing = DebugRuntime(
        DebugConfig(session_id=SESSION, replay_trace_id=None, local_origin=False),
        cast(Any, existing_sync),
        cast(Any, existing_async),
        None,
    )

    def _fake_init(client: LaminarClient, async_client: AsyncLaminarClient, debugger_url: str | None = None):
        # Mimic first-wins: a context already armed the runtime, so return the
        # existing instance, ignoring the clients this caller passed.
        return existing

    monkeypatch.setattr("lmnr.sdk.debug.init_debug_runtime", _fake_init)

    # _init_debug_runtime allocates these (the orphaned env-path clients).
    env_sync = _SpyDebugClient()
    env_async = _SpyAsyncDebugClient()
    _patch_clients(monkeypatch, env_sync, env_async)
    monkeypatch.setattr(Laminar, "_Laminar__project_api_key", "k", raising=False)

    Laminar._init_debug_runtime(base_url="http://localhost", http_port=8000)

    # The env path's freshly-allocated clients are closed (not leaked); the
    # context-armed runtime's clients stay open (it owns the run).
    assert env_sync.closed is True
    assert env_async.closed is True
    assert existing_sync.closed is False
    assert existing_async.closed is False
    _reset_runtime()


def test_init_preempts_context_runtime_when_env_debug_set(monkeypatch: MagicMock):
    # initialize() flips __initialized BEFORE _init_debug_runtime runs, and the
    # span funnels gate only on is_initialized() — so a span carrying a
    # propagated debug block can arm a context runtime (local_origin=False) in
    # that window. With LMNR_DEBUG set, env config owns the process, so
    # _init_debug_runtime must DISCARD that context runtime (closing its
    # orphaned clients) and build a fresh local-origin runtime that registers
    # the session and is wired for the browser / pointer hook. Without the
    # preempt, init_debug_runtime would idempotently return the context runtime
    # and the `runtime.client is not client` guard would bail — leaving the
    # local debug run with no session registration at all.
    _reset_runtime()
    CONTEXT_SESSION = "00000000-0000-0000-0000-0000000000c1"
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.delenv("LMNR_DEBUG_SESSION_ID", raising=False)
    monkeypatch.delenv("LMNR_DEBUG_REPLAY_TRACE_ID", raising=False)
    monkeypatch.delenv("LMNR_DEBUG_CACHE_UNTIL", raising=False)

    # A context-armed runtime that raced ahead of _init_debug_runtime, retaining
    # its own clients.
    ctx_sync = _SpyDebugClient()
    ctx_async = _SpyAsyncDebugClient()
    import lmnr.sdk.debug as debug_mod

    debug_mod._runtime = DebugRuntime(
        DebugConfig(
            session_id=CONTEXT_SESSION,
            replay_trace_id=None,
            local_origin=False,
        ),
        cast(Any, ctx_sync),
        cast(Any, ctx_async),
        None,
    )
    debug_mod._initialized = True

    # The env path allocates these fresh local-origin clients.
    env_sync = _SpyDebugClient()
    env_async = _SpyAsyncDebugClient()
    _patch_clients(monkeypatch, env_sync, env_async)
    monkeypatch.setattr(Laminar, "_Laminar__project_api_key", "k", raising=False)
    monkeypatch.setattr(Laminar, "_Laminar__global_metadata", {}, raising=False)

    Laminar._init_debug_runtime(base_url="http://localhost", http_port=8000)

    runtime = get_runtime()
    assert runtime is not None
    # The local-origin env runtime took over the process.
    assert runtime.local_origin is True
    assert runtime.client is env_sync
    assert runtime.session_id != CONTEXT_SESSION
    # The new local run registered its own session id with the backend.
    assert env_sync.rollout_sessions.registered == [(runtime.session_id, None)]
    # The raced context runtime's orphaned clients were closed; the env run's
    # clients stay open (it now owns the run).
    assert ctx_sync.closed is True
    assert ctx_async.closed is True
    assert env_sync.closed is False
    _reset_runtime()


def test_arm_from_context_refreshes_isolated_context_metadata(monkeypatch: MagicMock):
    # `LaminarSpanProcessor.on_start` reads `rollout.session_id` from
    # CONTEXT_METADATA_KEY on the parent context (NOT from __global_metadata), so
    # arming the debug runtime from a propagated block must also re-stamp the
    # ambient isolated context — otherwise auto-instrumented spans on a
    # downstream joined run omit `rollout.session_id`.
    _reset_runtime()
    SESSION = "00000000-0000-0000-0000-0000000000aa"
    block = DebugContext(enabled=True, session_id=SESSION)

    _patch_clients(monkeypatch, _SpyDebugClient())
    monkeypatch.setattr(Laminar, "_Laminar__project_api_key", "k", raising=False)
    monkeypatch.setattr(
        Laminar, "_Laminar__base_url_for_debug", "http://localhost", raising=False
    )
    monkeypatch.setattr(Laminar, "_Laminar__http_port_for_debug", 8000, raising=False)
    monkeypatch.setattr(Laminar, "_Laminar__global_metadata", {}, raising=False)

    # Start from a clean ambient context that carries no rollout.session_id.
    token = attach_context(get_current_context())
    try:
        assert cast(dict[str, str], get_value(CONTEXT_METADATA_KEY, get_current_context()) or {}).get(
            "rollout.session_id"
        ) is None

        Laminar._arm_debug_runtime_from_context(block)

        runtime = get_runtime()
        assert runtime is not None
        ctx_metadata = cast(dict[str, str], get_value(CONTEXT_METADATA_KEY, get_current_context()) or {})
        assert ctx_metadata.get("rollout.session_id") == runtime.session_id
    finally:
        detach_context(token)
        _reset_runtime()


def test_init_from_context_refreshes_context_runtime_reusing_clients(
    monkeypatch: MagicMock,
    sync_client: LaminarClient,
    async_client: AsyncLaminarClient
):
    # A context-armed runtime must REFRESH its dynamic coordinates on a new
    # context (the transport is reused; only the coordinates move) instead of
    # bailing first-wins. A different client pair on the refresh must be IGNORED.
    _reset_runtime()
    first, first_changed = init_debug_runtime_from_context(
        DebugContext(enabled=True, session_id="sess-a", replay_trace_id="trace-a"),
        sync_client,
        async_client,
    )
    assert first is not None
    assert first_changed is True

    # A different client pair on the refresh must be IGNORED — the transport
    # built on first arm is reused; only the coordinates move.
    other_sync, other_async = LaminarClient(project_api_key="test-123"), AsyncLaminarClient(project_api_key="test-123")
    try:
        second, second_changed = init_debug_runtime_from_context(
            DebugContext(enabled=True, session_id="sess-b", replay_trace_id="trace-b"),
            other_sync,
            other_async,
        )
    finally:
        other_sync.close()
        asyncio.run(other_async.close())
    assert second is first
    assert second_changed is True
    assert second is not None
    assert second.session_id == "sess-b"
    assert second.client is sync_client
    assert second.async_client is async_client
    _reset_runtime()


def test_init_from_context_never_overrides_env_origin_runtime(
    monkeypatch: MagicMock,
    sync_client: LaminarClient,
    async_client: AsyncLaminarClient
):
    # An env-origin runtime owns the process: a propagated context must not
    # hijack it — neither its coordinates nor its clients change.
    _reset_runtime()
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.setenv("LMNR_DEBUG_SESSION_ID", "env-sess")
    monkeypatch.delenv("LMNR_DEBUG_REPLAY_TRACE_ID", raising=False)
    monkeypatch.delenv("LMNR_DEBUG_CACHE_UNTIL", raising=False)

    env = init_debug_runtime(client=sync_client, async_client=async_client)
    assert env is not None
    assert env.local_origin is True

    runtime, changed = init_debug_runtime_from_context(
        DebugContext(enabled=True, session_id="ctx-sess", replay_trace_id="trace"),
        sync_client,
        async_client,
    )
    # Env config owns the process: the context must not hijack it.
    assert runtime is env
    assert changed is False
    rt = get_runtime()
    assert rt is not None
    assert rt.session_id == "env-sess"
    _reset_runtime()


def test_arm_from_context_refreshes_session_on_new_context(monkeypatch: MagicMock):
    # The coordinates in a propagated debug block are DYNAMIC: a long-lived
    # downstream service must follow each request's session id, not freeze on the
    # first context it ever saw. A second block with a different session must
    # update the runtime in place (clients reused) and re-stamp the metadata.
    _reset_runtime()
    SESSION_A = "00000000-0000-0000-0000-0000000000a1"
    SESSION_B = "00000000-0000-0000-0000-0000000000b2"

    spy = _SpyDebugClient()
    _patch_clients(monkeypatch, spy)
    monkeypatch.setattr(Laminar, "_Laminar__project_api_key", "k", raising=False)
    monkeypatch.setattr(
        Laminar, "_Laminar__base_url_for_debug", "http://localhost", raising=False
    )
    monkeypatch.setattr(Laminar, "_Laminar__http_port_for_debug", 8000, raising=False)
    monkeypatch.setattr(Laminar, "_Laminar__global_metadata", {}, raising=False)

    token = attach_context(get_current_context())
    try:
        Laminar._arm_debug_runtime_from_context(
            DebugContext(enabled=True, session_id=SESSION_A)
        )
        runtime_a = get_runtime()
        assert runtime_a is not None
        assert runtime_a.session_id == SESSION_A

        # Simulate the first session having latched run-live on a cache MISS —
        # the new session below must clear it so it starts from a clean state.
        Laminar.set_debug_run_live(True)

        Laminar._arm_debug_runtime_from_context(
            DebugContext(enabled=True, session_id=SESSION_B)
        )
        # Same runtime instance (clients reused), refreshed coordinates.
        rt = get_runtime()
        assert rt is not None
        assert rt is runtime_a
        assert rt.session_id == SESSION_B
        # Both sessions were registered (one per change), the new one re-stamped.
        assert spy.rollout_sessions.registered == [
            (SESSION_A, None),
            (SESSION_B, None),
        ]
        ctx_metadata = cast(dict[str, str], get_value(CONTEXT_METADATA_KEY, get_current_context()) or {})
        assert ctx_metadata.get("rollout.session_id") == SESSION_B
        # The new session reset the process-wide run-live latch.
        assert Laminar.is_debug_run_live() is False
    finally:
        detach_context(token)
        Laminar.set_debug_run_live(False)
        _reset_runtime()


def test_span_uses_freshly_armed_session_over_stale_context(
    monkeypatch: MagicMock,
    span_exporter: InMemorySpanExporter,
):
    # Regression: in start_span / start_as_current_span the parent `ctx` is
    # snapshot BEFORE arming the debug runtime. Arming the session attaches a
    # refreshed isolated context carrying that session's `rollout.session_id`, but
    # the metadata merge reads from the pre-arm snapshot — where context wins over
    # global — so a PRIOR request's `rollout.session_id` lingering on the ambient
    # context would override the just-armed session on the emitted span. The fix
    # re-reads the isolated context after arming; the span must carry the armed
    # session, not the stale one.
    _reset_runtime()
    span_exporter.clear()
    STALE_SESSION = "00000000-0000-0000-0000-0000000000a1"
    ARMED_SESSION = "00000000-0000-0000-0000-0000000000b2"

    _patch_clients(monkeypatch, _SpyDebugClient())
    monkeypatch.setattr(Laminar, "_Laminar__project_api_key", "k", raising=False)
    monkeypatch.setattr(
        Laminar, "_Laminar__base_url_for_debug", "http://localhost", raising=False
    )
    monkeypatch.setattr(Laminar, "_Laminar__http_port_for_debug", 8000, raising=False)
    monkeypatch.setattr(Laminar, "_Laminar__global_metadata", {}, raising=False)

    # Simulate a PRIOR request having left its session id on the ambient context.
    stale_ctx = set_value(
        CONTEXT_METADATA_KEY,
        {"rollout.session_id": STALE_SESSION},
        get_current_context(),
    )
    token = attach_context(stale_ctx)
    try:
        parent = LaminarSpanContext(
            trace_id=uuid.uuid4(),
            span_id=uuid.uuid4(),
            debug=DebugContext(enabled=True, session_id=ARMED_SESSION),
        )
        with Laminar.start_as_current_span("test", parent_span_context=parent):
            pass
    finally:
        detach_context(token)
        _reset_runtime()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert (
        (spans[0].attributes or {})["lmnr.association.properties.metadata.rollout.session_id"]
        == ARMED_SESSION
    )


def test_arm_from_context_reuses_clients_on_same_session(monkeypatch: MagicMock):
    # A steady stream of requests on the SAME session must not allocate new
    # clients per span, nor re-register or re-stamp metadata on every span.
    _reset_runtime()
    SESSION = "00000000-0000-0000-0000-0000000000aa"
    block = DebugContext(enabled=True, session_id=SESSION)

    built: list[_SpyDebugClient] = []

    def _spy_factory(*a: Any, **k: Any):
        c = _SpyDebugClient()
        built.append(c)
        return c

    monkeypatch.setattr(
        "lmnr.sdk.client.synchronous.sync_client.LaminarClient", _spy_factory
    )
    monkeypatch.setattr(
        "lmnr.sdk.client.asynchronous.async_client.AsyncLaminarClient",
        lambda *a, **k: _SpyAsyncDebugClient(),
    )
    monkeypatch.setattr(Laminar, "_Laminar__project_api_key", "k", raising=False)
    monkeypatch.setattr(
        Laminar, "_Laminar__base_url_for_debug", "http://localhost", raising=False
    )
    monkeypatch.setattr(Laminar, "_Laminar__http_port_for_debug", 8000, raising=False)
    monkeypatch.setattr(Laminar, "_Laminar__global_metadata", {}, raising=False)

    Laminar._arm_debug_runtime_from_context(block)
    Laminar._arm_debug_runtime_from_context(block)
    Laminar._arm_debug_runtime_from_context(block)

    runtime = get_runtime()
    assert runtime is not None
    # A sync client was built exactly once (first arm), reused thereafter.
    assert len(built) == 1
    # The session was registered exactly once (no re-register on unchanged id).
    assert cast(Any, runtime.client.rollout_sessions).registered == [(SESSION, None)]
    _reset_runtime()


def test_init_from_context_publishes_single_runtime_under_concurrency(
    monkeypatch: MagicMock,
    sync_client: LaminarClient,
    async_client: AsyncLaminarClient,
):
    # Span creation calls init_debug_runtime_from_context from arbitrary worker
    # threads. The check-and-set of the one-shot globals must be atomic, or two
    # threads both pass the _initialized check, both build a DebugRuntime, and
    # publish different instances — leaving every loser's clients (each thread
    # passes its own pair) returned to a caller whose `runtime.client is client`
    # cleanup guard then fails to recognize the win and leaks them. With the lock,
    # exactly one runtime is built and published and every caller gets it back.
    import lmnr.sdk.debug as debug_mod
    _reset_runtime()
    SESSION = "00000000-0000-0000-0000-0000000000aa"
    block = DebugContext(enabled=True, session_id=SESSION)

    n = 16
    start = threading.Barrier(n)

    # Count how many runtimes actually get constructed. Widen the race window by
    # sleeping in the config build (runs before the flag is set in the buggy
    # path) so every thread clears the unlocked _initialized check before any
    # thread publishes — the unsynchronized version then builds n runtimes.
    constructed = 0
    construct_lock = threading.Lock()
    real_runtime_cls = debug_mod.DebugRuntime

    class _CountingRuntime(real_runtime_cls):
        def __init__(self, *args: Any, **kwargs: Any):
            nonlocal constructed
            with construct_lock:
                constructed += 1
            super().__init__(*args, **kwargs)

    real_build = build_debug_config_from_context

    def _slow_build(dbg: Any) -> DebugConfig | None:
        time.sleep(0.02)
        return real_build(dbg)

    monkeypatch.setattr(debug_mod, "DebugRuntime", _CountingRuntime)
    monkeypatch.setattr(debug_mod, "build_debug_config_from_context", _slow_build)

    returned: list[DebugRuntime] = []
    returned_lock = threading.Lock()

    def _arm():
        # Each thread brings its own client pair, like _arm_debug_runtime_from_context.
        _success = start.wait()
        runtime, _ = init_debug_runtime_from_context(block, sync_client, async_client)
        with returned_lock:
            if runtime:
                returned.append(runtime)

    threads = [threading.Thread(target=_arm) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    published = get_runtime()
    assert published is not None
    # Exactly one runtime was built (no losers whose clients would leak), and
    # every caller got back that single published instance.
    assert constructed == 1
    assert len(returned) == n
    assert all(r is published for r in returned)
    _reset_runtime()
