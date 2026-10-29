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


def test_failed_agent_no_tool_spawned_counts_as_an_error():
    hi = [{"role": "user", "content": "hi"}]
    yo = [{"role": "assistant", "content": "yo"}]
    helper_call = _llm(4, 2, 3, hi, yo)
    trajectory = build_trajectory(
        [
            _record(1, None, "root", "DEFAULT", 0, 5),
            _llm(2, 0, 1, hi, yo),
            _record(
                3,
                1,
                "helper",
                "DEFAULT",
                2,
                4,
                {"gen_ai.agent.name": "helper"},
                {"type": "RuntimeError", "message": "boom"},
            ),
            _record(4, 3, helper_call.name, "LLM", 2, 3, helper_call.attributes),
        ]
    )
    assert trajectory is not None
    (result,) = trajectory["steps"][-1]["observation"]["results"]
    assert result["extra"]["lmnr"]["status"] == "error"
    assert trajectory["final_metrics"]["extra"]["lmnr"]["errors"] == 1


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


def _tool_call_output(call_id: str, name: str, arguments: dict[str, Any]) -> list[Any]:
    function = {"name": name, "arguments": json.dumps(arguments)}
    return [{"role": "assistant", "tool_calls": [{"id": call_id, "function": function}]}]


def _results(step: dict[str, Any]) -> list[Any]:
    return [r["content"] for r in step.get("observation", {}).get("results", [])]


def test_concurrent_calls_keep_their_own_tool_spans():
    """The second call starts before the first one's tool runs; that must not
    cut the first call's window, nor let it take the second call's tool."""
    trajectory = build_trajectory(
        [
            _record(1, None, "root", "DEFAULT", 0, 10),
            _llm(
                2, 0, 1, [{"role": "user", "content": "A"}], _tool_call_output("a", "s", {"q": 1})
            ),
            _llm(
                3,
                0.5,
                1.5,
                [{"role": "user", "content": "B"}],
                _tool_call_output("b", "s", {"q": 2}),
            ),
            _record(
                4,
                1,
                "s",
                "TOOL",
                1.6,
                2,
                {"lmnr.span.input": '{"q": 2}', "lmnr.span.output": '"B"'},
            ),
            _record(
                5,
                1,
                "s",
                "TOOL",
                1.7,
                2,
                {"lmnr.span.input": '{"q": 1}', "lmnr.span.output": '"A"'},
            ),
        ]
    )
    assert trajectory is not None
    steps = trajectory["steps"]
    assert [(s["source"], _results(s)) for s in steps] == [
        ("user", []),
        ("agent", ["A"]),
        ("user", []),
        ("agent", ["B"]),
    ]


def test_later_system_prompt_without_a_first_one_is_an_override():
    """With no system prompt on the first call, a later call's leading system
    message is that call's system prompt, not a message in the history."""
    hi = {"role": "user", "content": "hi"}
    yo = {"role": "assistant", "content": "yo"}
    second = [{"role": "system", "content": "S"}, hi, yo, {"role": "user", "content": "more"}]
    trajectory = build_trajectory(
        [
            _record(1, None, "root", "DEFAULT", 0, 10),
            _llm(2, 0, 1, [hi], [yo]),
            _llm(3, 2, 3, second, [{"role": "assistant", "content": "ok"}]),
        ]
    )
    assert trajectory is not None
    steps = trajectory["steps"]
    assert [s["source"] for s in steps] == ["user", "agent", "user", "agent"]
    assert steps[3]["extra"]["lmnr"]["system_prompt"] == "S"
    assert "input_messages" not in steps[3]["extra"]["lmnr"]
    ext = TrajectoryExtension.read(trajectory)
    assert reader.input_messages(steps, 3, ext) == second


def test_repeated_generated_tool_call_ids_keep_their_own_results():
    """Two Gemini conversations in one agent both generate `convert_1`; each
    call's observation is the result its own next call received."""

    def content(role: str, *parts: dict[str, Any]) -> dict[str, Any]:
        return {"role": role, "parts": list(parts)}

    def call(x: int) -> dict[str, Any]:
        return content("model", {"function_call": {"name": "convert", "args": {"x": x}}})

    def result(value: str) -> dict[str, Any]:
        response = {"name": "convert", "response": {"output": value}}
        return content("user", {"function_response": response})

    records = [_record(1, None, "root", "DEFAULT", 0, 20)]
    for n, (prompt, value) in enumerate([("one", "ONE"), ("two", "TWO")]):
        ask = content("user", {"text": prompt})
        done = content("model", {"text": f"done {prompt}"})
        records += [
            _llm(10 * n + 2, 10 * n, 10 * n + 1, [ask], [{"content": call(n)}]),
            _llm(
                10 * n + 3,
                10 * n + 2,
                10 * n + 3,
                [ask, call(n), result(value)],
                [{"content": done}],
            ),
        ]
    trajectory = build_trajectory(records)
    assert trajectory is not None
    observed = [_results(s) for s in trajectory["steps"] if s["source"] == "agent"]
    assert observed == [["ONE"], [], ["TWO"], []]


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


def test_output_after_a_prefilled_assistant_turn_stays_separate():
    prefill = {"role": "assistant", "content": [{"type": "text", "text": "{"}]}
    output = {"role": "assistant", "content": [{"type": "text", "text": '"a": 1}'}]}
    messages, message, reason = normalize_call(
        [{"role": "user", "content": "json please"}, prefill],
        [{**output, "stop_reason": "end_turn"}],
    )
    assert messages == [
        {"role": "user", "content": "json please"},
        {"role": "assistant", "content": "{"},
    ]
    assert message == {"role": "assistant", "content": '"a": 1}'}
    assert reason == "end_turn"


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


# -- shapes only the TypeScript SDK records -------------------------------------------


def test_ai_sdk_prompt_parts():
    messages, output, _reason = normalize_call(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": [{"type": "text", "text": "weather?"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "reasoning", "text": "use the tool"},
                    {
                        "type": "tool-call",
                        "toolCallId": "c1",
                        "toolName": "weather",
                        "input": {"city": "Paris"},
                        "providerOptions": {"openai": {}},
                    },
                    {"type": "tool-call", "toolCallId": "c2", "toolName": "weather", "input": "{}"},
                ],
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool-result",
                        "toolCallId": "c1",
                        "toolName": "weather",
                        "output": {"type": "json", "value": {"temp_c": 18}},
                    },
                    {
                        "type": "tool-result",
                        "toolCallId": "c2",
                        "toolName": "weather",
                        "output": {"type": "error-text", "value": "no city"},
                    },
                    {"type": "tool-approval-response", "approvalId": "a1", "approved": True},
                ],
            },
        ],
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "source", "sourceType": "url", "id": "s1", "url": "https://x"},
                    {"type": "text", "text": "18C."},
                ],
            }
        ],
    )
    assert messages == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "use the tool",
            "tool_calls": [
                {"tool_call_id": "c1", "function_name": "weather", "arguments": {"city": "Paris"}},
                {"tool_call_id": "c2", "function_name": "weather", "arguments": {}},
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": '{"temp_c": 18}'},
        {"role": "tool", "tool_call_id": "c2", "content": "no city", "_is_error": True},
    ]
    assert output == {"role": "assistant", "content": "18C."}


@pytest.mark.parametrize(
    ("output", "content", "is_error"),
    [
        ({"type": "text", "value": "done"}, "done", False),
        ({"type": "json", "value": ["a", "é"]}, '["a", "é"]', False),
        ({"type": "json", "value": "quoted"}, '"quoted"', False),
        ({"type": "error-json", "value": {"code": 1}}, '{"code": 1}', True),
        ({"type": "execution-denied", "reason": "not allowed"}, "not allowed", False),
        (
            {
                "type": "content",
                "value": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            },
            [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            False,
        ),
    ],
)
def test_ai_sdk_tool_result_outputs(output: dict[str, Any], content: Any, is_error: bool):
    part = {"type": "tool-result", "toolCallId": "c1", "toolName": "t", "output": output}
    messages, _output, _reason = normalize_call([{"role": "tool", "content": [part]}], None)
    expected: dict[str, Any] = {"role": "tool", "tool_call_id": "c1", "content": content}
    if is_error:
        expected["_is_error"] = True
    assert messages == [expected]


def test_ai_sdk_provider_executed_tool_result_in_output():
    _messages, output, _reason = normalize_call(
        [],
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool-call",
                        "toolCallId": "ws",
                        "toolName": "web_search",
                        "input": "{}",
                        "providerExecuted": True,
                    },
                    {
                        "type": "tool-result",
                        "toolCallId": "ws",
                        "toolName": "web_search",
                        "result": {"hits": 0},
                        "isError": True,
                    },
                ],
            }
        ],
    )
    assert output == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"tool_call_id": "ws", "function_name": "web_search", "arguments": {}}],
    }


def test_ai_sdk_file_parts():
    stored: list[tuple[str, str]] = []

    def sink(data: str, media_type: str) -> str:
        stored.append((data, media_type))
        return "images/x.png"

    messages, _output, _reason = normalize_call(
        [
            {
                "role": "user",
                "content": [
                    {"type": "file", "mediaType": "image/png", "data": "iVBORw0KGgo="},
                    {
                        "type": "file",
                        "mediaType": "image/jpeg",
                        "data": {"type": "url", "url": "https://x/cat.jpg"},
                    },
                    {
                        "type": "file",
                        "mediaType": "image/png",
                        "data": {"type": "data", "data": "AA=="},
                    },
                    {
                        "type": "file",
                        "mediaType": "application/pdf",
                        "filename": "a.pdf",
                        "data": "x",
                    },
                ],
            }
        ],
        None,
        sink,
    )
    assert stored == [("iVBORw0KGgo=", "image/png"), ("AA==", "image/png")]
    assert messages[0]["content"] == [
        {"type": "image", "source": {"media_type": "image/png", "path": "images/x.png"}},
        {"type": "image", "source": {"media_type": "image/jpeg", "path": "https://x/cat.jpg"}},
        {"type": "image", "source": {"media_type": "image/png", "path": "images/x.png"}},
        {"type": "text", "text": "[file: a.pdf]"},
    ]


def test_otel_semconv_parts():
    messages, output, reason = normalize_call(
        [
            {"role": "system", "parts": [{"type": "text", "content": "Be brief."}]},
            {"role": "user", "parts": [{"type": "text", "content": "weather?"}]},
            {
                "role": "assistant",
                "parts": [
                    {"type": "reasoning", "content": "use the tool"},
                    {
                        "type": "tool_call",
                        "id": "c1",
                        "name": "weather",
                        "arguments": {"city": "Paris"},
                    },
                ],
            },
            {
                "role": "tool",
                "parts": [{"type": "tool_call_response", "id": "c1", "response": "18C"}],
            },
            {
                "role": "user",
                "parts": [{"type": "uri", "mime_type": "image/png", "uri": "https://x/a.png"}],
            },
        ],
        [
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": "18C."}],
                "finish_reason": "stop",
            }
        ],
    )
    assert messages == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "use the tool",
            "tool_calls": [
                {"tool_call_id": "c1", "function_name": "weather", "arguments": {"city": "Paris"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "18C"},
        {
            "role": "user",
            "content": [
                {"type": "image", "source": {"media_type": "image/png", "path": "https://x/a.png"}}
            ],
        },
    ]
    assert output == {"role": "assistant", "content": "18C."}
    assert reason == "stop"


def test_chat_tool_calls_with_top_level_arguments():
    _messages, output, _reason = normalize_call(
        [],
        [
            {
                "role": "model",
                "content": "Checking.",
                "tool_calls": [{"id": "c1", "name": "weather", "arguments": '{"city": "Paris"}'}],
            }
        ],
    )
    assert output is not None
    assert output["tool_calls"] == [
        {"tool_call_id": "c1", "function_name": "weather", "arguments": {"city": "Paris"}}
    ]


@pytest.mark.parametrize(
    "attributes",
    [
        {"ai.toolCall.id": "c1", "ai.toolCall.name": "lookup"},
        {"gen_ai.tool.call.id": "c1"},
        {"gen_ai.tool.name": "lookup"},
        {"ai.toolCall.name": "lookup"},
    ],
)
def test_tool_spans_are_matched_by_their_tool_name_and_call_id(attributes: dict[str, Any]):
    """AI SDK (`ai.tool <name>`) and OTel GenAI (`execute_tool <name>`) TOOL
    span names decorate the tool's name, so the attributes decide."""
    trajectory = build_trajectory(
        [
            _record(1, None, "root", "DEFAULT", 0, 5),
            _llm(
                2, 0, 1, [{"role": "user", "content": "go"}], _tool_call_output("c1", "lookup", {})
            ),
            _record(
                3,
                1,
                "ai.tool lookup",
                "TOOL",
                1.1,
                1.2,
                {"lmnr.span.output": '"found"', **attributes},
            ),
        ]
    )
    assert trajectory is not None
    assert [s["source"] for s in trajectory["steps"]] == ["user", "agent"]
    assert _results(trajectory["steps"][1]) == ["found"]


def test_unmatched_and_implicit_subagent_tool_spans_use_the_tool_name():
    hi = [{"role": "user", "content": "hi"}]
    yo = [{"role": "assistant", "content": "yo"}]
    helper_call = _llm(5, 3, 4, hi, yo)
    trajectory = build_trajectory(
        [
            _record(1, None, "root", "DEFAULT", 0, 10),
            _llm(2, 0, 1, hi, yo),
            _record(3, 1, "execute_tool search", "TOOL", 2, 2.5, {"gen_ai.tool.name": "search"}),
            _record(4, 1, "ai.tool research", "TOOL", 3, 5, {"ai.toolCall.name": "research"}),
            _record(5, 4, helper_call.name, "LLM", 3, 4, helper_call.attributes),
        ]
    )
    assert trajectory is not None
    names = [s["tool_calls"][0]["function_name"] for s in trajectory["steps"][2:]]
    assert names == ["search", "research"]
    assert trajectory["subagent_trajectories"][0]["agent"]["name"] == "research"
