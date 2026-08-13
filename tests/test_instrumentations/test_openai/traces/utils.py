import httpx
from opentelemetry.sdk.trace import Span
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.trace.propagation import get_current_span
from unittest.mock import MagicMock


# from: https://stackoverflow.com/a/41599695/2749989
def spy_decorator(method_to_decorate):
    mock = MagicMock()

    def wrapper(self, *args, **kwargs):
        mock(*args, **kwargs)
        return method_to_decorate(self, *args, **kwargs)

    wrapper.mock = mock
    return wrapper


def single_request_to_path(mock: MagicMock, path: str) -> httpx.Request:
    """Return the one request the spy captured for ``path``.

    These spies patch ``httpx.Client.send`` / ``httpx.AsyncClient.send`` GLOBALLY, so
    they see every HTTP call any thread makes during the ``with`` block — not just the
    OpenAI one under test. Asserting ``call_once`` (or reading ``call_args``, which is
    the LAST call) therefore fails whenever unrelated background traffic happens to
    land in the window, e.g. a leaked Claude-proxy health poll. Select by path instead
    so the assertion is about this test's request and nothing else.
    """
    matching = [
        request
        for call in mock.call_args_list
        for request in call.args[:1]
        if isinstance(request, httpx.Request) and request.url.path == path
    ]
    assert len(matching) == 1, (
        f"expected exactly 1 request to {path}, got {len(matching)}. "
        f"All captured: {[c.args[0] for c in mock.call_args_list if c.args]}"
    )
    return matching[0]


def assert_request_contains_tracecontext(request: httpx.Request, expected_span: Span):
    assert TraceContextTextMapPropagator._TRACEPARENT_HEADER_NAME in request.headers
    ctx = TraceContextTextMapPropagator().extract(request.headers)
    request_span_context = get_current_span(ctx).get_span_context()
    expected_span_context = expected_span.get_span_context()

    assert request_span_context.trace_id == expected_span_context.trace_id
    assert request_span_context.span_id == expected_span_context.span_id
