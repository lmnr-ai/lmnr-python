"""Fixtures for the LangSmith bridge tests."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Generator
from typing import Any, cast

import pytest
import requests
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanContext
from requests.adapters import BaseAdapter
from typing_extensions import override

from lmnr.opentelemetry_lib.bridges import BridgeContext
from lmnr.opentelemetry_lib.bridges.langsmith import LangSmithBridge
from lmnr.opentelemetry_lib.tracing.processor import LaminarSpanProcessor

_new_val = os.environ.setdefault("LANGSMITH_API_KEY", "ls-test")
# Never reach a real LangSmith: every test client gets a recording session.
_new_val = os.environ.setdefault("LANGSMITH_ENDPOINT", "http://127.0.0.1:1")


class RecordingAdapter(BaseAdapter):
    """Answers every LangSmith request with 200 and keeps what was sent."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[tuple[str, str, bytes]] = []

    @override
    def send(self, request: requests.PreparedRequest, **kwargs: Any) -> requests.Response:  # pyright: ignore[reportIncompatibleMethodOverride]
        body = request.body or b""
        if isinstance(body, str):
            body = body.encode()
        path = re.sub(r"^https?://[^/]+", "", request.url or "")
        self.requests.append((request.method or "", path, body))
        response = requests.Response()
        response.status_code = 200
        response._content = b"{}"
        response.headers["content-type"] = "application/json"
        response.request = request
        return response

    @override
    def close(self) -> None:
        pass


class LangSmith:
    """A LangSmith client whose traffic is recorded instead of sent."""

    def __init__(self, **client_kwargs: Any) -> None:
        from langsmith import Client

        self.adapter: RecordingAdapter = RecordingAdapter()
        kwargs: dict[str, Any] = {
            "api_url": "http://langsmith.test",
            "api_key": "ls-test",
            "auto_batch_tracing": False,
            "info": {},
        }
        kwargs.update(client_kwargs)
        self.client: Any = Client(**kwargs)
        # The client mounts its own adapter on its session; the longer prefix wins.
        session = cast(requests.Session, self.client.session)
        session.mount("http://langsmith.test", self.adapter)

    def bodies(self) -> list[tuple[str, str, Any]]:
        """Run create / update requests as (method, path, parsed body)."""
        self.client.flush()
        out: list[tuple[str, str, Any]] = []
        for method, path, body in self.adapter.requests:
            if "/runs" not in path:
                continue
            try:
                parsed = json.loads(body)
            except ValueError:
                parsed = body
            out.append((method, path, parsed))
        return out


class Laminar:
    """Laminar's side of the bridge: its span processor and what it exported."""

    def __init__(self) -> None:
        self.exporter: InMemorySpanExporter = InMemorySpanExporter()
        self.processor: LaminarSpanProcessor = LaminarSpanProcessor(
            api_key="test", exporter=self.exporter, disable_batch=True
        )
        self.provider: TracerProvider = TracerProvider()
        self.provider.add_span_processor(self.processor)
        self.context: BridgeContext = BridgeContext(
            span_processor=self.processor, tracer_provider=self.provider
        )

    def spans(self) -> list[ReadableSpan]:
        return list(self.exporter.get_finished_spans())

    def span(self, name: str) -> ReadableSpan:
        matches = [s for s in self.spans() if s.name == name]
        assert len(matches) == 1, f"{name}: {[s.name for s in self.spans()]}"
        return matches[0]


def ids(span: ReadableSpan) -> SpanContext:
    assert span.context is not None
    return span.context


def parent_id(span: ReadableSpan) -> int | None:
    return span.parent.span_id if span.parent is not None else None


@pytest.fixture
def laminar() -> Generator[Laminar, None, None]:
    lmnr = Laminar()
    yield lmnr
    lmnr.processor.shutdown()


@pytest.fixture
def bridge(laminar: Laminar) -> Generator[LangSmithBridge, None, None]:
    bridge = LangSmithBridge()
    assert bridge.install(laminar.context)
    yield bridge
    bridge.uninstall()


@pytest.fixture
def langsmith() -> LangSmith:
    return LangSmith()
