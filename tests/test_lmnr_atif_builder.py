"""Tests for the lmnr-atif writer: `lmnr.sdk.trajectory.normalize` and `.builder`."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from lmnr.sdk.trajectory import reader
from lmnr.sdk.trajectory.builder import (
    BuildOptions,
    SpanRecord,
    build_trajectory,
    format_time,
    parse_time_ns,
)
from lmnr.sdk.trajectory.normalize import content_text, normalize_call, normalize_tools
from lmnr.sdk.trajectory.reader import TrajectoryExtension

SPEC_DIR = Path(__file__).resolve().parents[1] / "spec" / "lmnr-atif"
GOLDEN_DIR = SPEC_DIR / "writer-golden"
CASES = sorted(p for p in GOLDEN_DIR.iterdir() if p.is_dir())
GOLDEN_PRODUCER = {"name": "lmnr-golden", "version": "0"}


def _load_case(case: Path) -> tuple[list[SpanRecord], BuildOptions]:
    data = json.loads((case / "spans.json").read_text())
    options = BuildOptions(producer=dict(GOLDEN_PRODUCER), **data.get("options", {}))
    return [SpanRecord.from_json(s) for s in data["spans"]], options


def _build_case(case: Path) -> dict[str, Any]:
    records, options = _load_case(case)
    trajectory = build_trajectory(records, options)
    assert trajectory is not None
    return trajectory


def _walk(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    """The trajectory and all its embedded subagents."""
    result = [trajectory]
    for sub in trajectory.get("subagent_trajectories") or []:
        result += _walk(sub)
    return result


# -- golden cases ---------------------------------------------------------------


def test_golden_cases_exist():
    assert len(CASES) >= 6


@pytest.mark.parametrize("case", CASES, ids=lambda p: p.name)
def test_golden_case(case: Path):
    trajectory = _build_case(case)
    expected_path = case / "trajectory.json"
    rendered = json.dumps(trajectory, indent=2, ensure_ascii=False) + "\n"
    if os.environ.get("LMNR_ATIF_UPDATE_GOLDEN") == "1":
        _ = expected_path.write_text(rendered)
    assert json.loads(expected_path.read_text()) == trajectory


@pytest.mark.parametrize("case", CASES, ids=lambda p: p.name)
def test_golden_output_matches_schema(case: Path):
    schema = json.loads((SPEC_DIR / "lmnr-atif.schema.json").read_text())
    validator = jsonschema.Draft202012Validator(schema)
    errors = [e.message for e in validator.iter_errors(_build_case(case))]
    assert errors == []


@pytest.mark.parametrize("case", CASES, ids=lambda p: p.name)
def test_golden_output_is_valid_atif(case: Path):
    """`uv run --with harbor==0.25.0 pytest tests/test_lmnr_atif_builder.py` runs it."""
    trajectory_module = pytest.importorskip("harbor.models.trajectories.trajectory")
    trajectory_module.Trajectory.model_validate(_build_case(case))


def _comparable(messages: list[dict[str, Any]]) -> list[Any]:
    """What the model saw, for comparing a rebuilt input with the real one.

    The leading run of system messages is compared as joined text (an override
    joins several system messages into one); reasoning is ignored.
    """
    system: list[str] = []
    rest = list(messages)
    while rest and rest[0].get("role") in ("system", "developer"):
        system.append(content_text(rest.pop(0).get("content")))
    return ["\n\n".join(system)] + [
        [
            m.get("role"),
            m.get("content") if m.get("content") is not None else "",
            [
                [c.get("tool_call_id"), c.get("function_name"), c.get("arguments")]
                for c in m.get("tool_calls") or []
            ],
            m.get("tool_call_id"),
        ]
        for m in rest
    ]


@pytest.mark.parametrize("case", CASES, ids=lambda p: p.name)
def test_reader_rebuilds_every_call_input(case: Path):
    """The strongest check of the writer: reading the trajectory back with the
    spec's rules (§6.3, §6.4) gives each LLM call exactly the input it had."""
    records, _options = _load_case(case)
    trajectory = _build_case(case)
    by_span = {r.span_id: r for r in records}
    checked = 0

    def check(doc: dict[str, Any], parent: TrajectoryExtension | None) -> None:
        nonlocal checked
        ext = TrajectoryExtension.read(doc, parent=parent)
        assert ext is not None
        steps = reader.steps_of(doc)
        for index, step in enumerate(steps):
            if step.get("source") != "agent" or step.get("llm_call_count") == 0:
                continue
            record = by_span[step["extra"]["lmnr"]["span_id"]]
            raw_input = json.loads(record.attributes["gen_ai.input.messages"])
            actual, _output, _reason = normalize_call(raw_input, None)
            if record.attributes.get("gen_ai.prompt.0.role") == "system" and not any(
                m.get("role") == "system" for m in actual
            ):
                actual = [
                    {"role": "system", "content": record.attributes["gen_ai.prompt.0.content"]},
                    *actual,
                ]
            rebuilt = reader.input_messages(steps, index, ext)
            assert _comparable(rebuilt) == _comparable(actual), (doc["agent"]["name"], index)
            checked += 1
        for sub in doc.get("subagent_trajectories") or []:
            check(sub, ext)

    check(trajectory, None)
    llm_records = [r for r in records if r.span_type == "LLM"]
    nested = {r.parent_span_id for r in llm_records}
    assert checked == len([r for r in llm_records if r.span_id not in nested])


@pytest.mark.parametrize("case", CASES, ids=lambda p: p.name)
def test_golden_output_is_deterministic(case: Path):
    records, options = _load_case(case)
    assert build_trajectory(list(reversed(records)), options) == _build_case(case)


# -- builder ----------------------------------------------------------------------


def _record(
    span_id: int,
    parent: int | None,
    name: str,
    span_type: str,
    start: float,
    end: float,
    attributes: dict[str, Any] | None = None,
    error: dict[str, str] | None = None,
) -> SpanRecord:
    def sid(n: int) -> str:
        return f"00000000-0000-0000-0000-{n:012x}"

    base = parse_time_ns("2026-10-28T09:00:00.000000Z")
    return SpanRecord(
        trace_id="11111111-2222-4333-8444-555555555555",
        span_id=sid(span_id),
        parent_span_id=sid(parent) if parent is not None else None,
        name=name,
        span_type=span_type,
        start_time=format_time(base + int(start * 1e9)),
        end_time=format_time(base + int(end * 1e9)),
        attributes=attributes or {},
        error=error,
    )


def _llm(
    span_id: int, start: float, end: float, messages: list[Any], output: Any, **attrs: Any
) -> SpanRecord:
    return _record(
        span_id,
        1,
        "openai.chat",
        "LLM",
        start,
        end,
        {
            "gen_ai.system": "openai",
            "gen_ai.request.model": "gpt-5.5",
            "gen_ai.input.messages": json.dumps(messages),
            "gen_ai.output.messages": json.dumps(output),
            **attrs,
        },
    )


def test_no_llm_or_tool_spans_means_no_trajectory():
    assert build_trajectory([_record(1, None, "root", "DEFAULT", 0, 1)]) is None
    assert build_trajectory([]) is None


def test_span_record_json_round_trip():
    record = _record(2, 1, "x", "TOOL", 0, 1, {"a": 1}, {"type": "E", "message": "m"})
    assert SpanRecord.from_json(record.to_json()) == record
    assert SpanRecord.from_json(json.loads(json.dumps(record.to_json()))) == record


def test_time_formatting_keeps_microseconds():
    value = "2026-10-28T09:00:01.123456Z"
    assert format_time(parse_time_ns(value)) == value
    assert parse_time_ns("2026-10-28T09:00:01.123456+00:00") == parse_time_ns(value)


def test_default_producer_is_this_sdk():
    from lmnr.version import __version__

    trajectory = build_trajectory(
        [
            _record(1, None, "root", "DEFAULT", 0, 2),
            _llm(
                2,
                0,
                1,
                [{"role": "user", "content": "hi"}],
                [{"role": "assistant", "content": "yo"}],
            ),
        ]
    )
    assert trajectory is not None
    assert trajectory["extra"]["lmnr"]["producer"] == {
        "name": "lmnr-python",
        "version": __version__,
    }


def test_concurrent_calls_keep_their_own_inputs():
    """Two calls of one agent started at the same time: each step is followed by
    its own new messages' call, and the second gets the exact input it had."""
    first = [{"role": "user", "content": "Summarize A"}]
    second = [{"role": "user", "content": "Summarize B"}]
    trajectory = build_trajectory(
        [
            _record(1, None, "fanout", "DEFAULT", 0, 3),
            _llm(2, 1, 2, first, [{"role": "assistant", "content": "A is..."}]),
            _llm(3, 1, 2.5, second, [{"role": "assistant", "content": "B is..."}]),
        ]
    )
    assert trajectory is not None
    steps = trajectory["steps"]
    assert [(s["source"], s["message"]) for s in steps] == [
        ("user", "Summarize A"),
        ("agent", "A is..."),
        ("user", "Summarize B"),
        ("agent", "B is..."),
    ]
    assert steps[3]["extra"]["lmnr"]["input_messages"] == second
    ext = TrajectoryExtension.read(trajectory)
    assert reader.input_messages(steps, 3, ext) == second


def test_tool_span_matched_by_tool_call_id_attribute():
    output = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "c1", "function": {"name": "search", "arguments": '{"q": "a"}'}},
                {"id": "c2", "function": {"name": "search", "arguments": '{"q": "b"}'}},
            ],
        }
    ]
    trajectory = build_trajectory(
        [
            _record(1, None, "root", "DEFAULT", 0, 5),
            _llm(2, 0, 1, [{"role": "user", "content": "go"}], output),
            # Started in the opposite order; the call id decides.
            _record(
                3,
                1,
                "search",
                "TOOL",
                1.1,
                1.5,
                {"gen_ai.tool.call.id": "c2", "lmnr.span.output": '"B"'},
            ),
            _record(
                4,
                1,
                "search",
                "TOOL",
                1.2,
                1.6,
                {"gen_ai.tool.call.id": "c1", "lmnr.span.output": '"A"'},
            ),
        ]
    )
    assert trajectory is not None
    step = trajectory["steps"][1]
    results = {r["source_call_id"]: r["content"] for r in step["observation"]["results"]}
    assert results == {"c1": "A", "c2": "B"}
    spans = {c["tool_call_id"]: c["extra"]["lmnr"]["span_id"] for c in step["tool_calls"]}
    assert spans["c1"].endswith("4") and spans["c2"].endswith("3")


def test_tool_result_from_span_only_forces_an_exact_next_input():
    """The next call didn't receive the tool result, so rebuilding its input
    from the observation would be wrong; the writer writes it out."""
    call = {"id": "c1", "function": {"name": "lookup", "arguments": "{}"}}
    assistant = {"role": "assistant", "content": None, "tool_calls": [call]}
    trajectory = build_trajectory(
        [
            _record(1, None, "root", "DEFAULT", 0, 5),
            _llm(2, 0, 1, [{"role": "user", "content": "go"}], [assistant]),
            _record(3, 1, "lookup", "TOOL", 1.1, 1.2, {"lmnr.span.output": '"found"'}),
            _llm(
                4,
                2,
                3,
                [{"role": "user", "content": "go"}, assistant, {"role": "user", "content": "and?"}],
                [{"role": "assistant", "content": "done"}],
            ),
        ]
    )
    assert trajectory is not None
    steps = trajectory["steps"]
    assert steps[1]["observation"]["results"][0]["content"] == "found"
    assert "input_messages" in steps[3]["extra"]["lmnr"]
    ext = TrajectoryExtension.read(trajectory)
    assert [m["role"] for m in reader.input_messages(steps, 3, ext)] == [
        "user",
        "assistant",
        "user",
    ]


def test_inline_images_go_through_the_media_sink():
    stored: list[tuple[str, str]] = []

    def sink(data: str, media_type: str) -> str:
        stored.append((data, media_type))
        return "images/abc.png"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
            ],
        }
    ]
    records = [
        _record(1, None, "root", "DEFAULT", 0, 2),
        _llm(2, 0, 1, messages, [{"role": "assistant", "content": "A cat."}]),
    ]
    trajectory = build_trajectory(records, BuildOptions(media=sink))
    assert trajectory is not None
    assert stored == [("iVBORw0KGgo=", "image/png")]
    assert trajectory["steps"][0]["message"] == [
        {"type": "text", "text": "What is this?"},
        {"type": "image", "source": {"media_type": "image/png", "path": "images/abc.png"}},
    ]
    # Without a sink, a placeholder keeps the message readable.
    plain = build_trajectory(records)
    assert plain is not None
    assert plain["steps"][0]["message"][1] == {"type": "text", "text": "[image: image/png]"}


def test_options_are_written_to_the_root():
    trajectory = build_trajectory(
        [
            _record(1, None, "root", "DEFAULT", 0, 2),
            _llm(
                2,
                0,
                1,
                [{"role": "user", "content": "hi"}],
                [{"role": "assistant", "content": "yo"}],
            ),
        ],
        BuildOptions(
            producer=dict(GOLDEN_PRODUCER),
            release="abc123",
            project_id="proj-1",
            exported=True,
            complete=False,
        ),
    )
    assert trajectory is not None
    assert trajectory["agent"]["version"] == "abc123"
    lmnr = trajectory["extra"]["lmnr"]
    assert (lmnr["project_id"], lmnr["exported"], lmnr["complete"]) == ("proj-1", True, False)


# -- normalizer ---------------------------------------------------------------------


def test_anthropic_tool_results_are_split_out_of_user_messages():
    messages, _output, _reason = normalize_call(
        [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "s"},
                    {"type": "redacted_thinking", "data": "x"},
                    {"type": "tool_use", "id": "t1", "name": "ls", "input": {"path": "."}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "a.txt",
                        "is_error": True,
                    },
                    {"type": "text", "text": "<system-reminder>be brief</system-reminder>"},
                    {"type": "text", "text": "thanks", "cache_control": {"type": "ephemeral"}},
                ],
            },
        ],
        None,
    )
    assert messages == [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "hmm",
            "tool_calls": [
                {"tool_call_id": "t1", "function_name": "ls", "arguments": {"path": "."}}
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "a.txt", "_is_error": True},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<system-reminder>be brief</system-reminder>"},
                {"type": "text", "text": "thanks"},
            ],
        },
    ]


def test_openai_malformed_arguments_keep_the_raw_string():
    _messages, output, reason = normalize_call(
        [],
        [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": '{"a": '},
                        },
                        {
                            "id": "c2",
                            "type": "function",
                            "function": {"name": "g", "arguments": "[1, 2]"},
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            },
            {"index": 1, "message": {"role": "assistant", "content": "second choice"}},
        ],
    )
    assert reason == "tool_calls"
    assert output == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "tool_call_id": "c1",
                "function_name": "f",
                "arguments": {},
                "extra": {"lmnr": {"raw_arguments": '{"a": '}},
            },
            {
                "tool_call_id": "c2",
                "function_name": "g",
                "arguments": {},
                "extra": {"lmnr": {"raw_arguments": "[1, 2]"}},
            },
        ],
    }


def test_responses_items_merge_into_one_assistant_turn():
    messages, output, _reason = normalize_call(
        [
            {"role": "developer", "content": [{"type": "input_text", "text": "Be terse."}]},
            {"role": "user", "content": "weather?"},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "look it up"}]},
            {"type": "function_call", "call_id": "c1", "name": "weather", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
            {"type": "web_search_call", "id": "ws_1", "status": "completed"},
        ],
        [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Sunny."}],
            }
        ],
    )
    assert messages == [
        {"role": "developer", "content": "Be terse."},
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "look it up",
            "tool_calls": [{"tool_call_id": "c1", "function_name": "weather", "arguments": {}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
    ]
    assert output == {"role": "assistant", "content": "Sunny."}


def test_gemini_calls_without_ids_get_stable_ids_across_input_and_output():
    call = {"role": "model", "parts": [{"functionCall": {"name": "add", "args": {"a": 1}}}]}
    response = {
        "role": "user",
        "parts": [{"functionResponse": {"name": "add", "response": {"result": 2}}}],
    }
    messages, output, reason = normalize_call(
        [{"role": "user", "parts": [{"text": "add"}]}, call, response],
        [
            {
                "content": {
                    "role": "model",
                    "parts": [{"function_call": {"name": "add", "args": {"a": 2}}}],
                },
                "finishReason": "STOP",
            }
        ],
    )
    assert messages[1]["tool_calls"][0]["tool_call_id"] == "add_1"
    assert messages[1]["tool_calls"][0]["extra"] == {"lmnr": {"synthetic_id": True}}
    assert messages[2] == {"role": "tool", "tool_call_id": "add_1", "content": "2"}
    assert output is not None
    assert output["tool_calls"][0]["tool_call_id"] == "add_2"
    assert reason == "STOP"


def test_plain_strings_and_unknown_shapes_degrade_to_text():
    messages, output, _reason = normalize_call("just a prompt", "just an answer")
    assert messages == [{"role": "user", "content": "just a prompt"}]
    assert output == {"role": "assistant", "content": "just an answer"}
    assert normalize_call(None, None) == ([], None, None)
    messages, _output, _reason = normalize_call(
        [{"role": "user", "content": {"weird": True}}], None
    )
    assert messages == [{"role": "user", "content": '{"weird": true}'}]


def test_tool_definitions_in_every_shape_become_openai_functions():
    schema = {"type": "object", "properties": {}}
    tools = normalize_tools(
        [
            {"type": "function", "function": {"name": "a", "parameters": schema, "strict": True}},
            {"type": "function", "name": "b", "parameters": schema, "strict": True},
            {"name": "c", "description": "C.", "input_schema": schema, "cache_control": {}},
            {"function_declarations": [{"name": "d", "parameters": schema}]},
            {"name": "e", "inputSchema": schema},
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
            {"no": "name"},
        ]
    )
    assert tools == [
        {"type": "function", "function": {"name": "a", "parameters": schema}},
        {"type": "function", "function": {"name": "b", "parameters": schema}},
        {"type": "function", "function": {"name": "c", "description": "C.", "parameters": schema}},
        {"type": "function", "function": {"name": "d", "parameters": schema}},
        {"type": "function", "function": {"name": "e", "parameters": schema}},
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
    ]
    assert normalize_tools(None) == []
