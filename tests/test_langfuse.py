"""Tests for the Laminar <-> Langfuse bridge.

These tests exercise the bridge end-to-end against the real `langfuse` SDK
(pinned to 3.14.6 in dev deps). Langfuse v3 is OTel-native, so the bridge
works by attaching Laminar's `SpanProcessor` (plus our attribute translator)
to Langfuse's `TracerProvider`.

Coverage:
- Opt-in behaviour in `init_instrumentations`: the bridge is NEVER in the
  default set, even when langfuse is installed — it activates only via an
  explicit `instruments` set or `Laminar.connect_to_langfuse()`.
- Attribute translation from `langfuse.*` to `gen_ai.*` / `lmnr.*`.
- Langfuse spans reach Laminar's exporter (dual-export).
- `Laminar.connect_to_langfuse()` manual helper path.
"""

from __future__ import annotations

import json
import logging
import os
from unittest.mock import MagicMock

import pytest

# Silence Langfuse's OTLP exporter errors during tests — it will try to POST
# to a real endpoint on flush and fail loudly otherwise.
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "sk-lf-test")
os.environ.setdefault("LANGFUSE_HOST", "http://127.0.0.1:1")
# Disable the media-upload consumer thread. Its `run()` loop blocks on
# `queue.get(block=True, timeout=1)` with a HARDCODED 1s timeout (unaffected by
# flush_interval), so `client.shutdown()` would join() it for ~1s per test —
# the dominant cost in this file's teardown.
os.environ.setdefault("LANGFUSE_MEDIA_UPLOAD_ENABLED", "False")
logging.getLogger("opentelemetry.sdk.trace.export").setLevel(logging.CRITICAL)
logging.getLogger("opentelemetry.exporter.otlp.proto.http").setLevel(logging.CRITICAL)

from lmnr import Laminar  # noqa: E402
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.langfuse import (  # noqa: E402
    LangfuseAttributeTranslator,
    LangfuseInstrumentor,
    _is_langfuse_span,
)
from lmnr.opentelemetry_lib.tracing import TracerWrapper, instruments as instruments_mod  # noqa: E402
from lmnr.opentelemetry_lib.tracing.attributes import (  # noqa: E402
    ASSOCIATION_PROPERTIES,
    SPAN_INPUT,
    SPAN_OUTPUT,
    SPAN_TYPE,
)
from lmnr.opentelemetry_lib.tracing.instruments import (  # noqa: E402
    INSTRUMENTATION_INITIALIZERS,
    Instruments,
    init_instrumentations,
)


# Raw-provider / framework instrumentors that overlap with what Langfuse emits
# through its own wrappers. The bridge is opt-in, so these must stay enabled in
# the default set even when langfuse is installed (the user only pays the
# double-coverage cost if they explicitly opt into LANGFUSE alongside them).
_LANGFUSE_OVERLAP_PROVIDERS = {
    Instruments.ANTHROPIC,
    Instruments.BEDROCK,
    Instruments.COHERE,
    Instruments.GOOGLE_GENAI,
    Instruments.GROQ,
    Instruments.LANGCHAIN,
    Instruments.MISTRAL,
    Instruments.OPENAI,
}


def _langfuse_sdk_importable() -> bool:
    """Probe whether the real `langfuse` SDK can be imported on this
    interpreter. Langfuse pins pydantic v1, which fails to build some of
    langfuse's own generated API models on Python 3.14
    (`pydantic.v1.errors.ConfigError: unable to infer type`). The bridge
    itself still loads (it only uses OTel), so the translator/attach-path
    unit tests stay green — the SDK-backed integration tests skip instead.
    """
    try:
        import langfuse  # noqa: F401
        import langfuse._client.resource_manager  # noqa: F401
    except Exception:
        return False
    return True


_LANGFUSE_IMPORTABLE = _langfuse_sdk_importable()
_langfuse_sdk_required = pytest.mark.skipif(
    not _LANGFUSE_IMPORTABLE,
    reason="langfuse SDK cannot be imported on this interpreter "
    "(known pydantic v1 incompatibility on Python 3.14)",
)


# ---------------------------------------------------------------------------
# init_instrumentations auto-enable tests (no real langfuse interaction)
# ---------------------------------------------------------------------------


@pytest.fixture
def track_initializers(monkeypatch):
    """Patch every initializer in the map to record which were invoked."""
    called: set[Instruments] = set()
    replacements: dict[Instruments, object] = {}

    for instrument, initializer in INSTRUMENTATION_INITIALIZERS.items():
        fake = MagicMock(spec=initializer)
        fake.init_instrumentor = MagicMock(
            side_effect=lambda *_a, _inst=instrument, **_kw: called.add(_inst) or None
        )
        replacements[instrument] = fake

    monkeypatch.setattr(
        instruments_mod,
        "INSTRUMENTATION_INITIALIZERS",
        replacements,
    )
    return called


@pytest.fixture
def langfuse_installed(monkeypatch):
    monkeypatch.setattr(instruments_mod, "_langfuse_installed", lambda: True)


@pytest.fixture
def langfuse_not_installed(monkeypatch):
    monkeypatch.setattr(instruments_mod, "_langfuse_installed", lambda: False)


@pytest.fixture
def deepagents_installed(monkeypatch):
    monkeypatch.setattr(instruments_mod, "_deepagents_installed", lambda: True)


@pytest.fixture
def deepagents_not_installed(monkeypatch):
    monkeypatch.setattr(instruments_mod, "_deepagents_installed", lambda: False)


@pytest.fixture
def pydantic_ai_not_installed(monkeypatch):
    monkeypatch.setattr(instruments_mod, "_pydantic_ai_installed", lambda: False)


@pytest.fixture
def pydantic_ai_installed(monkeypatch):
    monkeypatch.setattr(instruments_mod, "_pydantic_ai_installed", lambda: True)


def test_langfuse_not_installed_defaults_exclude_it(
    track_initializers, langfuse_not_installed, pydantic_ai_not_installed
):
    """When langfuse is absent, LANGFUSE is not in the default set and all
    overlapping raw-provider instrumentors remain enabled."""
    init_instrumentations(tracer_provider=MagicMock(), instruments=None)
    assert Instruments.LANGFUSE not in track_initializers
    for instrument in _LANGFUSE_OVERLAP_PROVIDERS:
        assert (
            instrument in track_initializers
        ), f"{instrument} should remain when langfuse isn't installed"


def test_langfuse_initializer_skips_on_unreadable_or_invalid_version(monkeypatch):
    """Regression: the initializer used to pass version guards when
    `get_package_version` returned `None` (the check
    `if version and parse(version) < parse("3.0.0")` short-circuits on None),
    silently installing the bridge. For an explicit
    `instruments={Instruments.LANGFUSE}` call this would flip `_installed=True`
    and block any later valid install. Same story for an unparseable version
    string. Both must return `None` from the initializer."""
    from lmnr.opentelemetry_lib.tracing import _instrument_initializers

    monkeypatch.setattr(
        _instrument_initializers,
        "is_package_installed",
        lambda name: True if name == "langfuse" else False,
    )

    # Unreadable version.
    monkeypatch.setattr(
        _instrument_initializers,
        "get_package_version",
        lambda name: None,
    )
    initializer = _instrument_initializers.LangfuseInstrumentorInitializer()
    assert initializer.init_instrumentor() is None

    # Unparseable version string.
    monkeypatch.setattr(
        _instrument_initializers,
        "get_package_version",
        lambda name: "not-a-version",
    )
    assert initializer.init_instrumentor() is None

    # Happy path — parseable, >= 3.0 → instrumentor returned.
    monkeypatch.setattr(
        _instrument_initializers,
        "get_package_version",
        lambda name: "3.14.6",
    )
    assert initializer.init_instrumentor() is not None


def test_langfuse_v2_reports_not_installed(monkeypatch):
    """langfuse 2.x is not OTel-native, so the bridge initializer returns None.
    `_langfuse_installed()` must report False in that case so
    `connect_to_langfuse()` refuses to install a useless translator."""
    monkeypatch.setattr(
        instruments_mod,
        "is_package_installed",
        lambda name: True if name == "langfuse" else False,
    )
    monkeypatch.setattr(
        instruments_mod,
        "get_package_version",
        lambda name: "2.60.0" if name == "langfuse" else None,
    )
    assert instruments_mod._langfuse_installed() is False

    monkeypatch.setattr(
        instruments_mod,
        "get_package_version",
        lambda name: "3.0.0" if name == "langfuse" else None,
    )
    assert instruments_mod._langfuse_installed() is True


def test_langfuse_installed_is_never_in_default_set(
    track_initializers,
    langfuse_installed,
    pydantic_ai_not_installed,
    deepagents_not_installed,
):
    """The Langfuse bridge is opt-in: having langfuse installed must NOT add
    LANGFUSE to the default set or strip any overlapping raw-provider
    instrumentor. This guards the development-environment footgun where a
    transitive langfuse install would otherwise silently disable Laminar's own
    tracing for a user who only instrumented with Laminar."""
    init_instrumentations(
        tracer_provider=MagicMock(),
        instruments=None,
        lmnr_span_processor=MagicMock(),
    )
    assert Instruments.LANGFUSE not in track_initializers
    for instrument in _LANGFUSE_OVERLAP_PROVIDERS:
        assert (
            instrument in track_initializers
        ), f"{instrument} should remain when the bridge isn't opted into"


def test_langfuse_opt_in_via_explicit_instruments(
    track_initializers,
    langfuse_installed,
    pydantic_ai_installed,
    deepagents_installed,
):
    """Passing an explicit `instruments` set is the opt-in path. The set is
    honored verbatim — LANGFUSE runs alongside whatever else the caller asked
    for, with no auto-removal of overlapping providers (the caller accepts the
    double-coverage cost). Auto-enable logic for pydantic_ai/deepagents does
    NOT run when `instruments` is explicit."""
    init_instrumentations(
        tracer_provider=MagicMock(),
        instruments={Instruments.LANGFUSE, Instruments.OPENAI},
        lmnr_span_processor=MagicMock(),
    )
    assert Instruments.LANGFUSE in track_initializers
    assert Instruments.OPENAI in track_initializers


def test_explicit_instruments_without_langfuse_skips_bridge(
    track_initializers, langfuse_installed, deepagents_installed, pydantic_ai_installed
):
    """An explicit `instruments` set that omits LANGFUSE leaves the bridge
    off even though langfuse is installed — only what was asked for runs."""
    init_instrumentations(
        tracer_provider=MagicMock(),
        instruments={Instruments.OPENAI, Instruments.ANTHROPIC},
        lmnr_span_processor=MagicMock(),
    )
    assert Instruments.OPENAI in track_initializers
    assert Instruments.ANTHROPIC in track_initializers
    assert Instruments.LANGFUSE not in track_initializers


# ---------------------------------------------------------------------------
# Attribute translator unit tests (direct — no langfuse import needed)
# ---------------------------------------------------------------------------


class _FakeSpan:
    """Minimal stand-in for `opentelemetry.sdk.trace.Span` + `ReadableSpan`."""

    def __init__(self, attributes: dict, scope_name: str = "langfuse-sdk"):
        self._attributes = dict(attributes)
        scope = MagicMock()
        scope.name = scope_name
        self.instrumentation_scope = scope

    @property
    def attributes(self):
        return self._attributes

    def set_attribute(self, key, value):
        self._attributes[key] = value


def test_translator_ignores_non_langfuse_spans():
    translator = LangfuseAttributeTranslator()
    span = _FakeSpan({"some.attr": "x"}, scope_name="openai")
    translator.on_end(span)
    assert span.attributes == {"some.attr": "x"}


def test_is_langfuse_span_detects_attrs_without_scope():
    """A span carrying `langfuse.*` attributes is a Langfuse span even when its
    instrumentation scope is missing — detection must fall through to the
    attribute check rather than short-circuiting on a None scope."""
    span = _FakeSpan({"langfuse.observation.type": "generation"})
    span.instrumentation_scope = None
    assert _is_langfuse_span(span) is True


def test_translator_maps_generation_to_llm_span():
    translator = LangfuseAttributeTranslator()
    span = _FakeSpan(
        {
            "langfuse.observation.type": "generation",
            "langfuse.observation.model.name": "gpt-4o",
            "langfuse.observation.usage_details": json.dumps(
                {"input": 10, "output": 20, "total": 30}
            ),
            "langfuse.observation.cost_details": json.dumps(
                {"input": 0.001, "output": 0.002, "total": 0.003}
            ),
            # A non-message input dict (no "messages" key) is not a recognized
            # chat-prompt shape, so it falls back to the raw lmnr.span.input blob.
            "langfuse.observation.input": '{"prompt": "hi"}',
            "langfuse.observation.output": '{"role": "assistant", "content": "hello"}',
        }
    )
    translator.on_end(span)

    assert span.attributes[SPAN_TYPE] == "LLM"
    assert span.attributes["gen_ai.request.model"] == "gpt-4o"
    assert span.attributes["gen_ai.response.model"] == "gpt-4o"
    assert span.attributes["gen_ai.usage.input_tokens"] == 10
    assert span.attributes["gen_ai.usage.output_tokens"] == 20
    assert span.attributes["llm.usage.total_tokens"] == 30
    assert span.attributes["gen_ai.usage.input_cost"] == 0.001
    assert span.attributes["gen_ai.usage.output_cost"] == 0.002
    assert span.attributes["gen_ai.usage.cost"] == 0.003
    assert span.attributes[SPAN_INPUT] == '{"prompt": "hi"}'
    # A dict output is normalized into a one-element gen_ai.output.messages.
    assert json.loads(span.attributes["gen_ai.output.messages"]) == [
        {"role": "assistant", "content": "hello"}
    ]


def test_translator_empty_tool_calls_output_stays_normal_message():
    """Regression: a plain text completion that carries an empty `tool_calls`
    list must NOT be wrapped into the `{"message": ...}` choices shape.

    `all([])` is True, so the OpenAI-output-choice branch would otherwise fire
    on an empty `tool_calls` and warp the message shape, breaking transcript
    rendering."""
    translator = LangfuseAttributeTranslator()
    msg = {"role": "assistant", "content": "just text", "tool_calls": []}
    span = _FakeSpan(
        {
            "langfuse.observation.type": "generation",
            "langfuse.observation.output": json.dumps(msg),
        }
    )
    translator.on_end(span)
    out = json.loads(span.attributes["gen_ai.output.messages"])
    assert len(out) == 1
    # Must stay a normal message dict, not a {"message": {...}} choice wrapper.
    assert "message" not in out[0]
    assert out[0]["role"] == "assistant"
    # Empty tool_calls folds into content (no actual calls to append).
    assert out[0]["content"] == [{"type": "text", "text": "just text"}]


def test_translator_splits_openai_input_messages_and_tools():
    """Langfuse's OpenAI integration ships input as {messages, tools, ...}.

    The translator must split that into gen_ai.input.messages +
    gen_ai.tool.definitions rather than dumping the whole dict into
    lmnr.span.input, or the Laminar frontend can't render the transcript.
    """
    translator = LangfuseAttributeTranslator()
    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "weather in SF?"},
    ]
    span = _FakeSpan(
        {
            "langfuse.observation.type": "generation",
            "langfuse.observation.input": json.dumps(
                {"messages": messages, "tools": tools}
            ),
            "langfuse.observation.output": json.dumps(
                {"role": "assistant", "content": "It's sunny"}
            ),
        }
    )
    translator.on_end(span)

    assert json.loads(span.attributes["gen_ai.input.messages"]) == messages
    assert json.loads(span.attributes["gen_ai.tool.definitions"]) == tools
    # The whole {messages, tools} dict must NOT leak into lmnr.span.input.
    assert SPAN_INPUT not in span.attributes
    assert json.loads(span.attributes["gen_ai.output.messages"]) == [
        {"role": "assistant", "content": "It's sunny"}
    ]


def test_translator_handles_bare_message_list_input():
    """The vanilla OpenAI case (no tools) ships a bare message array."""
    translator = LangfuseAttributeTranslator()
    messages = [{"role": "user", "content": "hi"}]
    span = _FakeSpan(
        {
            "langfuse.observation.type": "generation",
            "langfuse.observation.input": json.dumps(messages),
        }
    )
    translator.on_end(span)
    assert json.loads(span.attributes["gen_ai.input.messages"]) == messages
    assert "gen_ai.tool.definitions" not in span.attributes
    assert SPAN_INPUT not in span.attributes


def test_translator_maps_openai_functions_to_tool_definitions():
    """The legacy `functions` key maps to tool definitions too."""
    translator = LangfuseAttributeTranslator()
    functions = [{"name": "get_weather", "parameters": {}}]
    span = _FakeSpan(
        {
            "langfuse.observation.type": "generation",
            "langfuse.observation.input": json.dumps(
                {
                    "messages": [{"role": "user", "content": "hi"}],
                    "functions": functions,
                }
            ),
        }
    )
    translator.on_end(span)
    assert json.loads(span.attributes["gen_ai.tool.definitions"]) == functions


def test_translator_splits_langchain_openai_inlined_tool_defs():
    """langfuse.langchain inlines OpenAI tool definitions into the message array
    as role="tool" messages whose content is {type:"function", function:{...}}.
    Those must be pulled out into gen_ai.tool.definitions, leaving only the real
    messages in gen_ai.input.messages."""
    translator = LangfuseAttributeTranslator()
    fn = {"name": "get_weather", "parameters": {"type": "object"}}
    span = _FakeSpan(
        {
            "langfuse.observation.type": "generation",
            "langfuse.observation.input": json.dumps(
                [
                    {"role": "user", "content": "weather?"},
                    {"role": "tool", "content": {"type": "function", "function": fn}},
                ]
            ),
        }
    )
    translator.on_end(span)
    assert json.loads(span.attributes["gen_ai.input.messages"]) == [
        {"role": "user", "content": "weather?"}
    ]
    assert json.loads(span.attributes["gen_ai.tool.definitions"]) == [fn]


def test_translator_splits_langchain_anthropic_inlined_tool_defs():
    """Regression: langchain + Anthropic via langfuse.

    Anthropic-langchain inlines tool DEFINITIONS into the input array as
    role="tool" messages whose content is the Anthropic-native shape
    {name, input_schema, description} — there is NO {type:"function"} wrapper.
    The old splitter did `content["type"]` (KeyError), which the on_end
    try/except swallowed, so the whole span was left untranslated (all
    langfuse.* attrs stayed in their original shape). The splitter must
    recognize the Anthropic shape and route it to gen_ai.tool.definitions."""
    translator = LangfuseAttributeTranslator()
    tool_def = {
        "name": "transform_1",
        "input_schema": {
            "properties": {
                "number": {"type": "integer"},
                "number2": {"type": "integer"},
            },
            "required": ["number"],
            "type": "object",
        },
        "description": "Runs transformation 1 on the given number.",
    }
    output = {
        "role": "assistant",
        "content": [
            {"text": "I'll apply these transformations.", "type": "text"},
            {
                "id": "toolu_01EF",
                "input": {"number": 2},
                "name": "transform_1",
                "type": "tool_use",
            },
        ],
        "tool_calls": [
            {
                "name": "transform_1",
                "args": {"number": 2},
                "id": "toolu_01EF",
                "type": "tool_call",
            }
        ],
    }
    span = _FakeSpan(
        {
            "langfuse.observation.type": "generation",
            "langfuse.observation.model.name": "claude-haiku-4-5-20251001",
            "langfuse.observation.usage_details": json.dumps(
                {
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "input": 862,
                    "output": 75,
                }
            ),
            "langfuse.observation.input": json.dumps(
                [
                    {"role": "user", "content": "Apply transformation 1 to 2."},
                    {"role": "tool", "content": tool_def},
                ]
            ),
            "langfuse.observation.output": json.dumps(output),
        }
    )
    translator.on_end(span)

    # The span is recognized as an LLM call and model/tokens are translated.
    assert span.attributes[SPAN_TYPE] == "LLM"
    assert span.attributes["gen_ai.request.model"] == "claude-haiku-4-5-20251001"
    assert span.attributes["gen_ai.usage.input_tokens"] == 862
    assert span.attributes["gen_ai.usage.output_tokens"] == 75
    # The Anthropic tool definition is pulled out of the message array.
    assert json.loads(span.attributes["gen_ai.input.messages"]) == [
        {"role": "user", "content": "Apply transformation 1 to 2."}
    ]
    assert json.loads(span.attributes["gen_ai.tool.definitions"]) == [tool_def]
    # The output keeps the tool_use content blocks and drops the redundant
    # top-level tool_calls mirror.
    out_messages = json.loads(span.attributes["gen_ai.output.messages"])
    assert len(out_messages) == 1
    assert "tool_calls" not in out_messages[0]
    assert out_messages[0]["content"] == output["content"]


def test_translator_non_llm_input_falls_back_to_span_input():
    """Non-LLM observations keep the raw input/output blob behaviour."""
    translator = LangfuseAttributeTranslator()
    span = _FakeSpan(
        {
            "langfuse.observation.type": "span",
            "langfuse.observation.input": '{"messages": [{"role": "user"}]}',
            "langfuse.observation.output": '{"role": "assistant"}',
        }
    )
    translator.on_end(span)
    assert span.attributes[SPAN_INPUT] == '{"messages": [{"role": "user"}]}'
    assert span.attributes[SPAN_OUTPUT] == '{"role": "assistant"}'
    assert "gen_ai.input.messages" not in span.attributes
    assert "gen_ai.output.messages" not in span.attributes


def test_translator_converts_openinference_llm_span():
    """openinference instrumentations (groq / google_genai) emit a flat,
    indexed attribute layout with no langfuse.* keys. The translator must
    recognize and convert them into Laminar / GenAI conventions."""
    translator = LangfuseAttributeTranslator()
    span = _FakeSpan(
        {
            "openinference.span.kind": "LLM",
            "llm.model_name": "gemini-2.0-flash",
            "llm.token_count.prompt": 12,
            "llm.token_count.completion": 8,
            "llm.token_count.total": 20,
            "llm.input_messages.0.message.role": "system",
            "llm.input_messages.0.message.content": "You are helpful",
            "llm.input_messages.1.message.role": "user",
            "llm.input_messages.1.message.content": "weather in SF?",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.tool_calls.0.tool_call.id": "call_1",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "get_weather",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": '{"city": "SF"}',
            "llm.tools.0.tool.json_schema": json.dumps(
                {"type": "function", "function": {"name": "get_weather"}}
            ),
        },
        scope_name="openinference.instrumentation.google_genai",
    )
    translator.on_end(span)

    assert span.attributes[SPAN_TYPE] == "LLM"
    assert span.attributes["gen_ai.request.model"] == "gemini-2.0-flash"
    assert span.attributes["gen_ai.usage.input_tokens"] == 12
    assert span.attributes["gen_ai.usage.output_tokens"] == 8
    assert span.attributes["llm.usage.total_tokens"] == 20
    assert json.loads(span.attributes["gen_ai.input.messages"]) == [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "weather in SF?"},
    ]
    out = json.loads(span.attributes["gen_ai.output.messages"])
    assert out[0]["role"] == "assistant"
    assert out[0]["content"][0]["type"] == "tool_call"
    assert out[0]["content"][0]["name"] == "get_weather"
    assert out[0]["content"][0]["arguments"] == '{"city": "SF"}'
    assert json.loads(span.attributes["gen_ai.tool.definitions"]) == [
        {"type": "function", "function": {"name": "get_weather"}}
    ]


def test_translator_converts_openinference_without_span_kind():
    """openinference spans are still recognized via the llm.* indexed keys
    even when openinference.span.kind is absent."""
    translator = LangfuseAttributeTranslator()
    span = _FakeSpan(
        {
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.0.message.content": "hi",
            "llm.token_count.prompt": 3,
        },
        scope_name="openinference.instrumentation.groq",
    )
    translator.on_end(span)
    assert span.attributes[SPAN_TYPE] == "LLM"
    assert json.loads(span.attributes["gen_ai.input.messages"]) == [
        {"role": "user", "content": "hi"}
    ]
    assert span.attributes["gen_ai.usage.input_tokens"] == 3


def test_translator_ignores_plain_non_langfuse_non_oi_spans():
    """A span that is neither langfuse- nor openinference-shaped is untouched."""
    translator = LangfuseAttributeTranslator()
    span = _FakeSpan({"some.attr": "x"}, scope_name="my.app")
    translator.on_end(span)
    assert span.attributes == {"some.attr": "x"}


def test_translator_maps_tool_observation():
    translator = LangfuseAttributeTranslator()
    span = _FakeSpan({"langfuse.observation.type": "tool"})
    translator.on_end(span)
    assert span.attributes[SPAN_TYPE] == "TOOL"


def test_translator_promotes_trace_session_user_tags_metadata():
    translator = LangfuseAttributeTranslator()
    span = _FakeSpan(
        {
            "session.id": "s456",
            "user.id": "u123",
            "langfuse.trace.tags": ("t1", "t2"),
            "langfuse.trace.metadata.mk": "mv",
            "langfuse.observation.type": "span",
        }
    )
    translator.on_end(span)
    assert span.attributes[f"{ASSOCIATION_PROPERTIES}.session_id"] == "s456"
    assert span.attributes[f"{ASSOCIATION_PROPERTIES}.user_id"] == "u123"
    assert span.attributes[f"{ASSOCIATION_PROPERTIES}.tags"] == ["t1", "t2"]
    assert span.attributes[f"{ASSOCIATION_PROPERTIES}.metadata.mk"] == "mv"


def test_translator_infers_total_tokens_when_absent():
    translator = LangfuseAttributeTranslator()
    span = _FakeSpan(
        {
            "langfuse.observation.type": "generation",
            "langfuse.observation.usage_details": json.dumps({"input": 7, "output": 3}),
        }
    )
    translator.on_end(span)
    assert span.attributes["llm.usage.total_tokens"] == 10


def test_translator_handles_dict_usage_without_json_wrap():
    """Langfuse sometimes passes a dict rather than a JSON string."""
    translator = LangfuseAttributeTranslator()
    span = _FakeSpan(
        {
            "langfuse.observation.type": "generation",
            # Directly a dict (not JSON-encoded) — the translator should still
            # handle it, but OTel's attribute type-check will have flattened it by
            # the time on_end runs. We keep the helper resilient.
            "langfuse.observation.usage_details": {"input": 1, "output": 2},
        }
    )
    translator.on_end(span)
    assert span.attributes["gen_ai.usage.input_tokens"] == 1
    assert span.attributes["gen_ai.usage.output_tokens"] == 2


# ---------------------------------------------------------------------------
# End-to-end integration tests with real langfuse SDK
# ---------------------------------------------------------------------------


def _silence_langfuse_background_threads(instances):
    """Neutralize the background threads a Langfuse client spins up so they
    don't slow the test suite down.

    Two distinct costs are eliminated:

    1. The OTLP span exporter (would POST to a dead host on every flush) is
       swapped for a no-op. Laminar's own SimpleSpanProcessor already exports
       Langfuse spans synchronously into the InMemorySpanExporter, so the tests
       never need Langfuse's exporter to run.

    2. The prompt-cache refresh thread (`PromptCacheTaskManager`) blocks on a
       HARDCODED 1s `queue.get(timeout=1)` and registers its OWN
       `atexit.register(shutdown)` that `client.shutdown()` never reaches. Left
       alone, every test's manager accumulates an atexit handler, and at
       interpreter teardown each one `join()`s a thread mid-`get()` — ~1s
       apiece, which is the multi-second freeze after the suite finishes. We
       pause the consumers and unregister their atexit hook; the threads are
       daemons, so the interpreter reaps them without a blocking join.
    """
    import atexit

    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    class _NoopExporter(SpanExporter):
        def export(self, spans):
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass

    for instance in instances:
        if getattr(instance, "tracer_provider", None) is not None:
            active = instance.tracer_provider._active_span_processor
            for proc in getattr(active, "_span_processors", ()):
                if hasattr(proc, "_batch_processor"):
                    proc._batch_processor._exporter = _NoopExporter()

        task_manager = getattr(
            getattr(instance, "prompt_cache", None), "_task_manager", None
        )
        if task_manager is not None:
            atexit.unregister(task_manager.shutdown)
            for consumer in task_manager._consumers:
                consumer.pause()


@pytest.fixture
def langfuse_client():
    """Fresh Langfuse client with an isolated resource manager.

    Langfuse caches instances in a module-level singleton keyed by
    public_key. We reset the cache each test so every test gets its own
    TracerProvider (otherwise earlier tests' bridges stack up and confuse the
    assertions).
    """
    if not _LANGFUSE_IMPORTABLE:
        pytest.skip(
            "langfuse SDK not importable on this interpreter "
            "(known pydantic v1 incompatibility on Python 3.14)"
        )
    # Delay import so tests that don't need langfuse can still be collected.
    from langfuse._client.resource_manager import LangfuseResourceManager

    # Clear the singleton cache so we get a fresh TracerProvider.
    LangfuseResourceManager._instances.clear()

    from langfuse import Langfuse

    client = Langfuse()
    _silence_langfuse_background_threads(LangfuseResourceManager._instances.values())

    yield client
    try:
        client.shutdown()
    except Exception:
        pass
    LangfuseResourceManager._instances.clear()


def test_connect_to_langfuse_dual_exports_observation(span_exporter, langfuse_client):
    """After `connect_to_langfuse`, Langfuse `@observe` spans reach Laminar's
    in-memory exporter with translated attributes."""
    assert Laminar.connect_to_langfuse() is True

    from langfuse import observe

    @observe
    def compute(x: int) -> int:
        return x * 2

    compute(21)

    spans = span_exporter.get_finished_spans()
    names = [s.name for s in spans]
    assert "compute" in names, f"langfuse span did not reach laminar exporter: {names}"
    span = next(s for s in spans if s.name == "compute")
    # Langfuse emits `langfuse.observation.input/output` with JSON encoding;
    # the translator rewrites them to lmnr.span.input / lmnr.span.output.
    assert SPAN_INPUT in span.attributes
    assert SPAN_OUTPUT in span.attributes
    assert "21" in span.attributes[SPAN_INPUT]
    assert span.attributes[SPAN_OUTPUT] == "42"


def test_connect_to_langfuse_translates_generation_attributes(
    span_exporter, langfuse_client
):
    assert Laminar.connect_to_langfuse() is True

    with langfuse_client.start_as_current_observation(
        name="my-llm",
        as_type="generation",
        model="gpt-4o",
        input={"prompt": "hi"},
    ) as gen:
        gen.update(
            output={"text": "hello"},
            usage_details={"input": 10, "output": 20, "total": 30},
            cost_details={"input": 0.001, "output": 0.002, "total": 0.003},
        )

    span = next(s for s in span_exporter.get_finished_spans() if s.name == "my-llm")
    assert span.attributes[SPAN_TYPE] == "LLM"
    assert span.attributes["gen_ai.request.model"] == "gpt-4o"
    assert span.attributes["gen_ai.usage.input_tokens"] == 10
    assert span.attributes["gen_ai.usage.output_tokens"] == 20
    assert span.attributes["llm.usage.total_tokens"] == 30
    assert span.attributes["gen_ai.usage.input_cost"] == 0.001
    assert span.attributes["gen_ai.usage.output_cost"] == 0.002
    assert span.attributes["gen_ai.usage.cost"] == 0.003


def test_connect_to_langfuse_splits_openai_generation_input(
    span_exporter, langfuse_client
):
    """End-to-end: a generation whose input is the OpenAI {messages, tools}
    shape (what `langfuse.openai` ships) is split into gen_ai.input.messages +
    gen_ai.tool.definitions on the Laminar side."""
    assert Laminar.connect_to_langfuse() is True

    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    messages = [{"role": "user", "content": "weather in SF?"}]
    with langfuse_client.start_as_current_observation(
        name="openai-gen",
        as_type="generation",
        model="gpt-4o",
        input={"messages": messages, "tools": tools},
    ) as gen:
        gen.update(output={"role": "assistant", "content": "It's sunny"})

    span = next(s for s in span_exporter.get_finished_spans() if s.name == "openai-gen")
    assert json.loads(span.attributes["gen_ai.input.messages"]) == messages
    assert json.loads(span.attributes["gen_ai.tool.definitions"]) == tools
    assert SPAN_INPUT not in span.attributes
    assert json.loads(span.attributes["gen_ai.output.messages"]) == [
        {"role": "assistant", "content": "It's sunny"}
    ]


def test_connect_to_langfuse_promotes_trace_session_and_user(
    span_exporter, langfuse_client
):
    assert Laminar.connect_to_langfuse() is True

    from langfuse import propagate_attributes

    with langfuse_client.start_as_current_observation(name="root") as root:
        with propagate_attributes(
            user_id="user-42",
            session_id="session-7",
            tags=["env:test"],
            metadata={"feature": "beta"},
        ):
            pass

    root = next(s for s in span_exporter.get_finished_spans() if s.name == "root")
    assert root.attributes[f"{ASSOCIATION_PROPERTIES}.session_id"] == "session-7"
    assert root.attributes[f"{ASSOCIATION_PROPERTIES}.user_id"] == "user-42"
    tags = root.attributes[f"{ASSOCIATION_PROPERTIES}.tags"]
    assert "env:test" in tags
    assert root.attributes[f"{ASSOCIATION_PROPERTIES}.metadata.feature"] == "beta"


def test_connect_to_langfuse_is_idempotent(span_exporter, langfuse_client):
    """Calling the bridge twice must not duplicate spans on the Laminar side."""
    assert Laminar.connect_to_langfuse() is True
    assert Laminar.connect_to_langfuse() is True  # second call: no-op

    from langfuse import observe

    @observe
    def once() -> str:
        return "done"

    once()

    spans = [s for s in span_exporter.get_finished_spans() if s.name == "once"]
    assert len(spans) == 1, f"expected exactly one span, got {len(spans)}"


def test_translator_mutates_before_synchronous_exporter():
    """Regression: under `disable_batch=True` the Laminar exporter uses
    `SimpleSpanProcessor` and exports inside `on_end`. The translator MUST run
    first, otherwise the exporter ships the pre-translation `langfuse.*`
    shape. We verify this by installing a fake SimpleSpanProcessor-backed
    exporter on the same provider and confirming it sees translated attrs.
    """
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        SimpleSpanProcessor,
        SpanExporter,
        SpanExportResult,
    )

    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.langfuse import (
        _prepend_span_processor,
    )

    exported: list[dict] = []

    class CaptureExporter(SpanExporter):
        def export(self, spans):
            for s in spans:
                exported.append(dict(s.attributes or {}))
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(CaptureExporter()))

    # Install the translator AFTER the exporter (the real-world order) but
    # via `_prepend_span_processor`, which reorders it to the front.
    translator = LangfuseAttributeTranslator()
    assert _prepend_span_processor(provider, translator) is True

    tracer = provider.get_tracer("langfuse-sdk")
    span = tracer.start_span(
        "llm-call",
        attributes={"langfuse.observation.model.name": "gpt-4o"},
    )
    span.end()

    assert len(exported) == 1
    assert (
        exported[0].get("gen_ai.request.model") == "gpt-4o"
    ), f"translator must run before exporter, got {exported[0]}"


def test_connect_to_langfuse_swallows_install_exceptions(monkeypatch):
    """Regression: `LangfuseInstrumentor.instrument()` re-raises on
    attach-phase failures (e.g. `RuntimeError` from concurrent modification
    of `LangfuseResourceManager._instances`). `Laminar.connect_to_langfuse()`
    documents a `bool` return and callers should not have to wrap it in a
    try/except — so the helper must swallow the exception and return
    `False` on failure."""
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation import langfuse as lf

    LangfuseInstrumentor._installed = False
    LangfuseInstrumentor._handled_providers = set()
    LangfuseInstrumentor._attached_providers = {}
    LangfuseInstrumentor._translator = None
    LangfuseInstrumentor._lmnr_span_processor = None
    LangfuseInstrumentor._lmnr_tracer_provider = None
    LangfuseInstrumentor._original_initialize_instance = None

    def exploding_attach(self):  # noqa: ARG001
        raise RuntimeError("simulated attach failure")

    monkeypatch.setattr(
        lf.LangfuseInstrumentor,
        "_attach_to_existing_langfuse_providers",
        exploding_attach,
    )

    # Must not raise.
    assert Laminar.connect_to_langfuse() is False
    # `instrument()`'s rollback path must have cleaned up, leaving
    # `_installed=False`.
    assert LangfuseInstrumentor._installed is False


def test_connect_to_langfuse_before_initialize_does_not_crash(monkeypatch):
    """Regression: `connect_to_langfuse()` must return `False` (not raise)
    when called before `Laminar.initialize()`. The not-initialized branch
    previously accessed the private name-mangled `cls.__logger`, which is
    only set by `_initialize_logger()` during `initialize()`. Any caller
    that probed for Langfuse support before initializing Laminar would hit
    an `AttributeError` instead of the documented False return."""
    # Force the "not initialized" state: both the `__initialized` flag and
    # (crucially) the `__logger` attribute absent — the latter is what the
    # original bug stumbled on.
    monkeypatch.setattr(Laminar, "_Laminar__initialized", False)
    if hasattr(Laminar, "_Laminar__logger"):
        monkeypatch.delattr(Laminar, "_Laminar__logger")

    # Must not raise.
    assert Laminar.connect_to_langfuse() is False


def test_connect_to_langfuse_returns_false_on_install_failure(monkeypatch):
    """If `instrument()` bails out before setting `_installed=True` (e.g.
    translator install raises), the public helper must surface the failure
    as `False` — not claim success."""
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation import langfuse as lf

    LangfuseInstrumentor._installed = False
    LangfuseInstrumentor._translator = None
    LangfuseInstrumentor._original_initialize_instance = None

    def failing_prepend(provider, processor):  # noqa: ARG001
        raise RuntimeError("boom")

    monkeypatch.setattr(lf, "_prepend_span_processor", failing_prepend)

    assert Laminar.connect_to_langfuse() is False
    assert LangfuseInstrumentor._installed is False


def test_connect_to_langfuse_returns_false_without_langfuse(monkeypatch):
    """If `langfuse` isn't importable (or is too old), the helper must return
    False and must NOT install the bridge (i.e. no translator added, no
    monkey-patch). The version-aware `_langfuse_installed` check is what
    guards this — see the companion 2.x-specific test below."""
    monkeypatch.setattr(
        instruments_mod,
        "_langfuse_installed",
        lambda: False,
    )

    LangfuseInstrumentor._installed = False
    LangfuseInstrumentor._translator = None
    LangfuseInstrumentor._original_initialize_instance = None

    assert Laminar.connect_to_langfuse() is False
    assert LangfuseInstrumentor._installed is False
    assert LangfuseInstrumentor._translator is None


def test_connect_to_langfuse_returns_false_when_sdk_unimportable(monkeypatch):
    """If `langfuse` is present per metadata but cannot actually be imported
    (the pydantic-v1 failure on Python 3.14), the bridge's resource-manager
    attach/patch path would silently no-op while `instrument()` still flips
    `_installed=True`. `connect_to_langfuse()` must NOT report success there —
    SDK spans (`@observe`, `langfuse.openai`, ...) would never reach Laminar.
    Guard on a real import probe and return False without installing."""
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation import langfuse as lf

    # Metadata says installed and modern, but the SDK can't be imported.
    monkeypatch.setattr(instruments_mod, "_langfuse_installed", lambda: True)
    monkeypatch.setattr(lf, "langfuse_sdk_importable", lambda: False)

    LangfuseInstrumentor._installed = False
    LangfuseInstrumentor._translator = None
    LangfuseInstrumentor._original_initialize_instance = None

    assert Laminar.connect_to_langfuse() is False
    # The bridge must NOT have been installed (no orphaned translator that a
    # later valid install would have to clean up / could stack a second one).
    assert LangfuseInstrumentor._installed is False
    assert LangfuseInstrumentor._translator is None


def test_connect_to_langfuse_rejects_langfuse_v2(monkeypatch):
    """Regression: `connect_to_langfuse()` must also version-gate on
    langfuse >= 3.0. With 2.x installed, the bridge initializer returns
    None, so installing it would attach a useless translator and
    permanently flip `_installed=True` (blocking a later valid install)."""
    monkeypatch.setattr(
        instruments_mod,
        "is_package_installed",
        lambda name: True if name == "langfuse" else False,
    )
    monkeypatch.setattr(
        instruments_mod,
        "get_package_version",
        lambda name: "2.60.0" if name == "langfuse" else None,
    )

    LangfuseInstrumentor._installed = False
    LangfuseInstrumentor._translator = None
    LangfuseInstrumentor._original_initialize_instance = None

    assert Laminar.connect_to_langfuse() is False
    assert LangfuseInstrumentor._installed is False
    assert LangfuseInstrumentor._translator is None


@_langfuse_sdk_required
def test_late_attach_patches_future_langfuse_clients(span_exporter):
    """The resource-manager patch means Langfuse clients created AFTER the
    bridge is installed still get dual-attached."""
    from langfuse._client.resource_manager import LangfuseResourceManager

    # Install bridge with NO langfuse client yet.
    LangfuseResourceManager._instances.clear()
    assert Laminar.connect_to_langfuse() is True

    # Now create a Langfuse client — this triggers
    # `LangfuseResourceManager._initialize_instance`, which our bridge
    # has monkey-patched to attach Laminar's processor.
    from langfuse import Langfuse, observe

    client = Langfuse()
    _silence_langfuse_background_threads(LangfuseResourceManager._instances.values())

    try:

        @observe
        def late() -> int:
            return 7

        late()
    finally:
        client.shutdown()
        LangfuseResourceManager._instances.clear()

    names = [s.name for s in span_exporter.get_finished_spans()]
    assert "late" in names, f"late-created Langfuse client was not bridged: {names}"


def test_instrumentor_skips_laminar_own_provider(span_exporter):
    """Guardrail: if Langfuse somehow ends up sharing Laminar's
    TracerProvider, `_attach_to_provider` must skip it (otherwise Laminar's
    processor would be attached to Laminar's provider twice → duplicate
    spans on every Laminar export).
    """
    instrumentor = LangfuseInstrumentor()
    wrapper = TracerWrapper.instance
    initial_count = len(
        wrapper._tracer_provider._active_span_processor._span_processors
    )
    instrumentor._lmnr_span_processor = wrapper._span_processor
    instrumentor._attach_to_provider(wrapper._tracer_provider)
    after_count = len(wrapper._tracer_provider._active_span_processor._span_processors)
    assert (
        after_count == initial_count
    ), "attached to Laminar's own provider — would cause duplicate exports"


def test_instrument_rolls_back_translator_if_attach_phase_raises(span_exporter):
    """Regression: if `_attach_to_existing_langfuse_providers` or
    `_patch_resource_manager` raises, the translator that was already
    prepended to Laminar's provider in step 1 must be removed and all
    class-level state cleared. Otherwise, a subsequent `instrument()` (e.g.
    via `Laminar.connect_to_langfuse()`) would pass the `_installed` guard
    and prepend a SECOND translator, double-translating every Langfuse span.
    """
    from opentelemetry.sdk.trace import TracerProvider

    # Clean state.
    LangfuseInstrumentor._installed = False
    LangfuseInstrumentor._handled_providers = set()
    LangfuseInstrumentor._attached_providers = {}
    LangfuseInstrumentor._translator = None
    LangfuseInstrumentor._lmnr_span_processor = None
    LangfuseInstrumentor._lmnr_tracer_provider = None

    provider = TracerProvider()

    def count_translators() -> int:
        return sum(
            1
            for p in provider._active_span_processor._span_processors
            if isinstance(p, LangfuseAttributeTranslator)
        )

    baseline = count_translators()

    instrumentor = LangfuseInstrumentor()

    # Force `_attach_to_existing_langfuse_providers` to blow up on first
    # install.
    def exploding_attach(self):  # noqa: ARG001
        raise RuntimeError("simulated attach failure")

    original_attach = LangfuseInstrumentor._attach_to_existing_langfuse_providers
    LangfuseInstrumentor._attach_to_existing_langfuse_providers = exploding_attach
    try:
        with pytest.raises(RuntimeError, match="simulated attach failure"):
            instrumentor.instrument(
                lmnr_tracer_provider=provider,
                lmnr_span_processor=MagicMock(),
            )
    finally:
        LangfuseInstrumentor._attach_to_existing_langfuse_providers = original_attach

    # Translator must have been rolled back.
    assert (
        count_translators() == baseline
    ), "translator must be removed on partial install failure"
    assert LangfuseInstrumentor._installed is False
    assert LangfuseInstrumentor._translator is None

    # A subsequent successful install must attach exactly ONE translator —
    # not two, which is what would happen if the failed-install translator
    # were still around.
    instrumentor2 = LangfuseInstrumentor()
    instrumentor2.instrument(
        lmnr_tracer_provider=provider,
        lmnr_span_processor=MagicMock(),
    )
    assert count_translators() == baseline + 1
    instrumentor2.uninstrument()


def test_instrument_skips_shared_laminar_provider_without_tracerwrapper(span_exporter):
    """Regression: during auto-install via `init_instrumentations`,
    `TracerWrapper.instance` is assigned AFTER `init_instrumentations`
    returns. If a pre-existing Langfuse client happens to share Laminar's
    newly-created `TracerProvider`, `_attach_to_provider`'s
    `TracerWrapper.verify_initialized()` fallback guard returns False
    (the wrapper isn't set yet) and the translator + Laminar span processor
    would be double-attached.

    `instrument()` must pre-register `id(lmnr_tracer_provider)` in
    `_handled_providers` so the short-circuit works independently of the
    TracerWrapper lifecycle.
    """
    from opentelemetry.sdk.trace import TracerProvider

    # Start from a clean slate.
    LangfuseInstrumentor._installed = False
    LangfuseInstrumentor._handled_providers = set()
    LangfuseInstrumentor._attached_providers = {}
    LangfuseInstrumentor._translator = None
    LangfuseInstrumentor._lmnr_span_processor = None
    LangfuseInstrumentor._lmnr_tracer_provider = None

    # A fresh provider standing in for Laminar's. We deliberately do NOT
    # reassign `TracerWrapper.instance` to point at this provider — that's
    # exactly the case the guard has to cover during auto-install.
    shared_provider = TracerProvider()

    mock_processor = MagicMock()
    instrumentor = LangfuseInstrumentor()
    instrumentor.instrument(
        lmnr_tracer_provider=shared_provider,
        lmnr_span_processor=mock_processor,
    )

    # One translator from `_prepend_span_processor` in `instrument()` —
    # baseline.
    processors_after_install = list(
        shared_provider._active_span_processor._span_processors
    )
    translator_count = sum(
        1
        for p in processors_after_install
        if isinstance(p, LangfuseAttributeTranslator)
    )
    assert translator_count == 1
    assert mock_processor not in processors_after_install, (
        "Laminar span processor should NOT be attached to Laminar's own "
        "provider — the translator is enough there"
    )

    # Now simulate Langfuse's resource manager calling `_attach_to_provider`
    # with the same provider. The pre-registered id must prevent a
    # double-attach.
    instrumentor._attach_to_provider(shared_provider)

    processors_after_simulated_langfuse = list(
        shared_provider._active_span_processor._span_processors
    )
    translator_count_after = sum(
        1
        for p in processors_after_simulated_langfuse
        if isinstance(p, LangfuseAttributeTranslator)
    )
    assert (
        translator_count_after == 1
    ), "translator should not be attached twice to Laminar's provider"
    assert mock_processor not in processors_after_simulated_langfuse, (
        "Laminar span processor should not be appended to Laminar's own "
        "provider via the Langfuse attach path"
    )

    instrumentor.uninstrument()


@_langfuse_sdk_required
def test_uninstrument_removes_translator_and_clears_state(span_exporter):
    """Regression: `uninstrument` must detach the translator from Laminar's
    provider and clear class-level state (`_handled_providers`, `_translator`,
    `_lmnr_span_processor`, `_attached_providers`, `_lmnr_tracer_provider`).
    Otherwise a subsequent `instrument()` call would prepend a SECOND
    translator onto Laminar's provider (the first was never removed), and
    `_handled_providers` would still hold ids from the previous session,
    making `_attach_to_existing_langfuse_providers` skip providers it saw
    last time around.
    """
    from langfuse._client.resource_manager import LangfuseResourceManager

    LangfuseResourceManager._instances.clear()
    wrapper = TracerWrapper.instance
    lmnr_provider = wrapper._tracer_provider

    def count_translators() -> int:
        return sum(
            1
            for p in lmnr_provider._active_span_processor._span_processors
            if isinstance(p, LangfuseAttributeTranslator)
        )

    # Start from a clean slate in case an earlier test left state behind.
    LangfuseInstrumentor._installed = False
    LangfuseInstrumentor._handled_providers = set()
    LangfuseInstrumentor._attached_providers = {}
    LangfuseInstrumentor._translator = None
    LangfuseInstrumentor._lmnr_span_processor = None
    LangfuseInstrumentor._lmnr_tracer_provider = None

    baseline_translators = count_translators()

    instrumentor = LangfuseInstrumentor()
    instrumentor.instrument(
        lmnr_tracer_provider=lmnr_provider,
        lmnr_span_processor=wrapper._span_processor,
    )
    assert LangfuseInstrumentor._installed is True
    assert count_translators() == baseline_translators + 1

    instrumentor.uninstrument()

    # Translator must have been removed from Laminar's provider.
    assert count_translators() == baseline_translators
    # All class-level state must be cleared so a fresh install starts clean.
    assert LangfuseInstrumentor._installed is False
    assert LangfuseInstrumentor._translator is None
    assert LangfuseInstrumentor._lmnr_span_processor is None
    assert LangfuseInstrumentor._lmnr_tracer_provider is None
    assert LangfuseInstrumentor._handled_providers == set()
    assert LangfuseInstrumentor._attached_providers == {}

    # Re-install: the translator count should increase by exactly one again —
    # NOT two, which is what would happen if the previous translator was
    # still attached.
    instrumentor2 = LangfuseInstrumentor()
    instrumentor2.instrument(
        lmnr_tracer_provider=lmnr_provider,
        lmnr_span_processor=wrapper._span_processor,
    )
    assert count_translators() == baseline_translators + 1
    instrumentor2.uninstrument()
    assert count_translators() == baseline_translators


def test_uninstrument_detaches_processors_from_langfuse_providers(span_exporter):
    """Regression: after `uninstrument`, every provider we previously attached
    the translator / Laminar span processor to must lose those processors.
    Re-install must also succeed (stale `_handled_providers` must not
    short-circuit re-attachment).

    We simulate a Langfuse-owned TracerProvider with a plain
    `sdk.trace.TracerProvider`, directed at `_attach_to_provider`: the bridge
    treats any non-Laminar provider the same way, so this exercises the
    install/uninstall/reinstall flow deterministically without depending on
    whether the real Langfuse SDK reuses the global provider.
    """
    from opentelemetry.sdk.trace import TracerProvider

    wrapper = TracerWrapper.instance
    lmnr_provider = wrapper._tracer_provider
    lmnr_processor = wrapper._span_processor

    # Start from a known-clean state.
    LangfuseInstrumentor._installed = False
    LangfuseInstrumentor._handled_providers = set()
    LangfuseInstrumentor._attached_providers = {}
    LangfuseInstrumentor._translator = None
    LangfuseInstrumentor._lmnr_span_processor = None
    LangfuseInstrumentor._lmnr_tracer_provider = None

    fake_lf_provider = TracerProvider()
    baseline = list(fake_lf_provider._active_span_processor._span_processors)

    instrumentor = LangfuseInstrumentor()
    instrumentor.instrument(
        lmnr_tracer_provider=lmnr_provider,
        lmnr_span_processor=lmnr_processor,
    )
    # Manually attach to our fake Langfuse-owned provider the same way the
    # monkey-patched `_initialize_instance` would in production.
    instrumentor._attach_to_provider(fake_lf_provider)

    after_install = list(fake_lf_provider._active_span_processor._span_processors)
    assert any(isinstance(p, LangfuseAttributeTranslator) for p in after_install)
    assert lmnr_processor in after_install
    assert id(fake_lf_provider) in LangfuseInstrumentor._attached_providers

    instrumentor.uninstrument()

    after_uninstall = list(fake_lf_provider._active_span_processor._span_processors)
    assert not any(
        isinstance(p, LangfuseAttributeTranslator) for p in after_uninstall
    ), "translator should be removed from the Langfuse-owned provider"
    assert (
        lmnr_processor not in after_uninstall
    ), "Laminar span processor should be removed from the Langfuse-owned provider"
    # Ordering of unrelated processors should be preserved.
    assert after_uninstall == baseline

    # Re-install must work — stale `_handled_providers` was cleared so the
    # existing provider is seen again.
    instrumentor.instrument(
        lmnr_tracer_provider=lmnr_provider,
        lmnr_span_processor=lmnr_processor,
    )
    instrumentor._attach_to_provider(fake_lf_provider)
    reinstalled = list(fake_lf_provider._active_span_processor._span_processors)
    assert any(isinstance(p, LangfuseAttributeTranslator) for p in reinstalled)
    assert lmnr_processor in reinstalled
    instrumentor.uninstrument()


# ---------------------------------------------------------------------------
# LiteLLM `langfuse_otel` bridge
# ---------------------------------------------------------------------------


def _litellm_langfuse_otel_importable() -> bool:
    try:
        import litellm  # noqa: F401
        from litellm.integrations.langfuse.langfuse_otel import (  # noqa: F401
            LangfuseOtelLogger,
        )
    except Exception:
        return False
    return True


_LITELLM_IMPORTABLE = _litellm_langfuse_otel_importable()
_litellm_required = pytest.mark.skipif(
    not _LITELLM_IMPORTABLE,
    reason="litellm (with langfuse_otel) not importable on this interpreter",
)


def test_translator_routes_litellm_hybrid_span_through_openinference():
    """LiteLLM's `langfuse_otel` callback emits a HYBRID span: openinference
    `llm.*` attrs (model, tokens, indexed messages, tools) AND `langfuse.*`
    trace attrs (session.id, user.id) — but NO `langfuse.observation.type`.

    The openinference path must win (it carries the LLM data the langfuse path
    can't see without an observation type) AND it must still promote the
    `langfuse.*` trace-level session/user attributes.
    """
    translator = LangfuseAttributeTranslator()
    span = _FakeSpan(
        {
            # openinference half
            "openinference.span.kind": "LLM",
            "llm.model_name": "gpt-4o",
            "llm.token_count.prompt": 11,
            "llm.token_count.completion": 5,
            "llm.token_count.total": 16,
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.0.message.content": "hi",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": "hello",
            # langfuse half (no observation.type!)
            "session.id": "sess-9",
            "user.id": "user-3",
            "langfuse.observation.input": '{"messages": [{"role": "user"}]}',
        },
        scope_name="litellm",
    )
    translator.on_end(span)

    assert span.attributes[SPAN_TYPE] == "LLM"
    assert span.attributes["gen_ai.request.model"] == "gpt-4o"
    assert span.attributes["gen_ai.usage.input_tokens"] == 11
    assert span.attributes["gen_ai.usage.output_tokens"] == 5
    assert json.loads(span.attributes["gen_ai.input.messages"]) == [
        {"role": "user", "content": "hi"}
    ]
    assert json.loads(span.attributes["gen_ai.output.messages"]) == [
        {"role": "assistant", "content": "hello"}
    ]
    # Trace-level promotion from the langfuse half must still happen.
    assert span.attributes[f"{ASSOCIATION_PROPERTIES}.session_id"] == "sess-9"
    assert span.attributes[f"{ASSOCIATION_PROPERTIES}.user_id"] == "user-3"


def _reset_langfuse_instrumentor_state():
    LangfuseInstrumentor._installed = False
    LangfuseInstrumentor._handled_providers = set()
    LangfuseInstrumentor._attached_providers = {}
    LangfuseInstrumentor._translator = None
    LangfuseInstrumentor._lmnr_span_processor = None
    LangfuseInstrumentor._lmnr_tracer_provider = None
    LangfuseInstrumentor._original_initialize_instance = None
    LangfuseInstrumentor._original_litellm_init_logger = None


@_litellm_required
def test_litellm_bridge_attaches_to_existing_logger(span_exporter):
    """A `LangfuseOtelLogger` already constructed before the bridge installs
    must get the translator + Laminar span processor attached to its private
    `_tracer_provider`."""
    from opentelemetry.sdk.trace import TracerProvider

    from litellm.integrations.langfuse.langfuse_otel import LangfuseOtelLogger
    from litellm.litellm_core_utils import litellm_logging

    _reset_langfuse_instrumentor_state()

    logger_obj = LangfuseOtelLogger(callback_name="langfuse_otel")
    provider = logger_obj._tracer_provider
    assert isinstance(provider, TracerProvider)

    original_loggers = list(litellm_logging._in_memory_loggers)
    litellm_logging._in_memory_loggers.append(logger_obj)
    try:
        wrapper = TracerWrapper.instance
        instrumentor = LangfuseInstrumentor()
        instrumentor.instrument(
            lmnr_tracer_provider=wrapper._tracer_provider,
            lmnr_span_processor=wrapper._span_processor,
        )
        processors = list(provider._active_span_processor._span_processors)
        assert any(
            isinstance(p, LangfuseAttributeTranslator) for p in processors
        ), "translator must be attached to the LiteLLM logger's provider"
        assert (
            wrapper._span_processor in processors
        ), "Laminar span processor must be attached to the LiteLLM provider"

        instrumentor.uninstrument()
        after = list(provider._active_span_processor._span_processors)
        assert not any(
            isinstance(p, LangfuseAttributeTranslator) for p in after
        ), "uninstrument must detach the translator from the LiteLLM provider"
        assert wrapper._span_processor not in after
    finally:
        litellm_logging._in_memory_loggers[:] = original_loggers
        _reset_langfuse_instrumentor_state()


@_litellm_required
def test_litellm_bridge_patches_factory_for_late_loggers(span_exporter):
    """A `langfuse_otel` logger constructed AFTER the bridge installs (via
    LiteLLM's `_init_custom_logger_compatible_class` factory) must also get
    dual-attached. The factory patch must be reverted on uninstrument."""
    from litellm.litellm_core_utils import litellm_logging

    _reset_langfuse_instrumentor_state()

    original_loggers = list(litellm_logging._in_memory_loggers)
    original_factory = litellm_logging._init_custom_logger_compatible_class
    try:
        wrapper = TracerWrapper.instance
        instrumentor = LangfuseInstrumentor()
        instrumentor.instrument(
            lmnr_tracer_provider=wrapper._tracer_provider,
            lmnr_span_processor=wrapper._span_processor,
        )
        # The factory must have been wrapped.
        assert (
            litellm_logging._init_custom_logger_compatible_class is not original_factory
        )

        logger_obj = litellm_logging._init_custom_logger_compatible_class(
            "langfuse_otel",
            internal_usage_cache=None,
            llm_router=None,
        )
        provider = logger_obj._tracer_provider
        processors = list(provider._active_span_processor._span_processors)
        assert any(
            isinstance(p, LangfuseAttributeTranslator) for p in processors
        ), "late-constructed LiteLLM logger must be bridged via the factory"
        assert wrapper._span_processor in processors

        instrumentor.uninstrument()
        # Factory patch must be reverted.
        assert litellm_logging._init_custom_logger_compatible_class is original_factory
    finally:
        litellm_logging._init_custom_logger_compatible_class = original_factory
        litellm_logging._in_memory_loggers[:] = original_loggers
        _reset_langfuse_instrumentor_state()


@_litellm_required
def test_litellm_factory_does_not_bridge_non_langfuse_loggers(span_exporter):
    """The factory patch must bridge ONLY `langfuse_otel`. Other OTel-based
    LiteLLM callbacks built through the same factory also carry a private
    `_tracer_provider` (the base `otel` callback is exactly such a case);
    attaching Laminar's translator to them would ship unrelated spans into
    Laminar. The base `otel` provider is in fact the global (Laminar's own)
    provider, so we can't assert on its processors — instead we assert the
    factory patch never routes a non-langfuse logger through
    `_attach_to_provider`."""
    from litellm.litellm_core_utils import litellm_logging

    _reset_langfuse_instrumentor_state()

    original_loggers = list(litellm_logging._in_memory_loggers)
    original_factory = litellm_logging._init_custom_logger_compatible_class
    try:
        wrapper = TracerWrapper.instance
        instrumentor = LangfuseInstrumentor()
        instrumentor.instrument(
            lmnr_tracer_provider=wrapper._tracer_provider,
            lmnr_span_processor=wrapper._span_processor,
        )

        attached: list = []
        original_attach = instrumentor._attach_to_provider

        def _spy(provider):
            attached.append(provider)
            return original_attach(provider)

        instrumentor._attach_to_provider = _spy

        # The base `otel` callback is an `OpenTelemetry` instance (NOT a
        # `LangfuseOtelLogger`) that carries its own `_tracer_provider`. The
        # factory patch must NOT attach to it.
        litellm_logging._init_custom_logger_compatible_class(
            "otel",
            internal_usage_cache=None,
            llm_router=None,
        )
        assert attached == [], (
            "non-langfuse LiteLLM logger must NOT be routed to " "_attach_to_provider"
        )

        # A `langfuse_otel` logger built through the same factory MUST attach.
        litellm_logging._init_custom_logger_compatible_class(
            "langfuse_otel",
            internal_usage_cache=None,
            llm_router=None,
        )
        assert attached, "langfuse_otel logger must be bridged via the factory"

        instrumentor.uninstrument()
    finally:
        litellm_logging._init_custom_logger_compatible_class = original_factory
        litellm_logging._in_memory_loggers[:] = original_loggers
        _reset_langfuse_instrumentor_state()
