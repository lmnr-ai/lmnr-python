"""Tests for the lmnr-atif spec artifacts and its reader (`lmnr.sdk.trajectory`)."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from lmnr.sdk.trajectory import reader
from lmnr.sdk.trajectory.reader import TrajectoryExtension
from lmnr.sdk.trajectory.spec import SPEC_VERSION

SPEC_DIR = Path(__file__).resolve().parents[1] / "spec" / "lmnr-atif"
FIXTURES = sorted((SPEC_DIR / "fixtures").glob("*.json"))


def _load(name: str) -> dict[str, Any]:
    return json.loads((SPEC_DIR / "fixtures" / name).read_text())


def _schema() -> dict[str, Any]:
    return json.loads((SPEC_DIR / "lmnr-atif.schema.json").read_text())


def _schema_errors(doc: dict[str, Any]) -> list[str]:
    validator = jsonschema.Draft202012Validator(_schema())
    return [e.message for e in validator.iter_errors(doc)]


# -- spec artifacts ---------------------------------------------------------


def test_fixtures_exist():
    assert len(FIXTURES) >= 4


def test_spec_version_matches_every_artifact():
    """The constant, the schema, the doc and the fixtures must agree (spec §2)."""
    assert _schema()["$id"] == f"urn:lmnr-atif:{SPEC_VERSION}"
    doc = (SPEC_DIR / "lmnr-atif.md").read_text()
    assert f"- **Spec version:** {SPEC_VERSION}\n" in doc
    assert f"### {SPEC_VERSION}\n" in doc
    for path in FIXTURES:
        trajectory = json.loads(path.read_text())
        assert trajectory["extra"]["lmnr"]["spec_version"] == SPEC_VERSION, path.name
        for sub in trajectory.get("subagent_trajectories") or []:
            assert sub["extra"]["lmnr"]["spec_version"] == SPEC_VERSION, path.name


def test_schema_is_valid_json_schema():
    jsonschema.Draft202012Validator.check_schema(_schema())


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_fixture_matches_schema(path: Path):
    assert _schema_errors(json.loads(path.read_text())) == []


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_fixture_is_valid_atif(path: Path):
    """Validated with Harbor's own models when harbor is installed (Python 3.12+).

    `uv run --with harbor==0.25.0 pytest tests/test_lmnr_atif.py` runs it locally.
    """
    trajectory_module = pytest.importorskip("harbor.models.trajectories.trajectory")
    trajectory_module.Trajectory.model_validate(json.loads(path.read_text()))


def test_schema_rejects_unknown_keys_and_other_versions():
    doc = _load("basic-agent-loop.json")

    typo = copy.deepcopy(doc)
    typo["steps"][2]["extra"]["lmnr"]["start"] = typo["steps"][2]["extra"]["lmnr"].pop("start_time")
    assert _schema_errors(typo)

    newer = copy.deepcopy(doc)
    newer["extra"]["lmnr"]["spec_version"] = "0.2.0"
    assert _schema_errors(newer)

    no_producer = copy.deepcopy(doc)
    del no_producer["extra"]["lmnr"]["producer"]
    assert _schema_errors(no_producer)

    bad_time = copy.deepcopy(doc)
    bad_time["steps"][2]["extra"]["lmnr"]["end_time"] = "2026-10-28T12:00:01Z"
    assert _schema_errors(bad_time)


def test_schema_checks_step_fields_by_step_kind():
    doc = _load("errors-and-dispatch.json")
    # `synthetic` belongs to `llm_call_count: 0` steps only.
    llm_step = copy.deepcopy(doc)
    llm_step["steps"][1]["extra"]["lmnr"]["synthetic"] = True
    assert _schema_errors(llm_step)
    # A message step can't carry LLM-call fields.
    user_step = copy.deepcopy(doc)
    user_step["steps"][0]["extra"]["lmnr"]["provider"] = "openai"
    assert _schema_errors(user_step)


def test_fixture_def_keys_are_sha256_of_canonical_json():
    """Keys are opaque to readers, but our fixtures follow the SHOULD in §6.6."""
    import hashlib

    for path in FIXTURES:
        defs = json.loads(path.read_text())["extra"]["lmnr"].get("defs") or {}
        for table in defs.values():
            for key, value in table.items():
                canonical = json.dumps(
                    value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                )
                assert key == "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


# -- versioning ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("version", "reference", "expected"),
    [
        ("0.1.0", "0.1.0", True),
        ("0.1.7", "0.1.0", True),
        ("0.1.0", "0.1.3", True),
        ("0.2.0", "0.1.0", False),
        ("0.0.9", "0.1.0", False),
        ("1.0.0", "0.1.0", False),
        ("1.4.2", "1.0.0", True),
        ("2.0.0", "1.9.0", False),
        ("0.1.0-rc.1", "0.1.0", True),
        ("0.1", "0.1.0", False),
        ("lmnr-atif/1", "0.1.0", False),
        (None, "0.1.0", False),
        (1, "0.1.0", False),
    ],
)
def test_is_compatible(version: Any, reference: str, expected: bool):
    assert reader.is_compatible(version, reference) is expected


def test_read_returns_none_for_plain_or_unreadable_atif():
    doc = _load("basic-agent-loop.json")
    assert TrajectoryExtension.read(doc) is not None

    plain = copy.deepcopy(doc)
    del plain["extra"]
    assert TrajectoryExtension.read(plain) is None

    newer = copy.deepcopy(doc)
    newer["extra"]["lmnr"]["spec_version"] = "0.2.0"
    assert TrajectoryExtension.read(newer) is None

    missing = copy.deepcopy(doc)
    del missing["extra"]["lmnr"]["spec_version"]
    assert TrajectoryExtension.read(missing) is None

    old_atif = copy.deepcopy(doc)
    old_atif["schema_version"] = "ATIF-v1.6"
    assert TrajectoryExtension.read(old_atif) is None

    assert TrajectoryExtension.read("not a trajectory") is None


def test_subagent_inherits_parent_version_and_defs():
    doc = _load("subagents.json")
    root = TrajectoryExtension.read(doc)
    assert root is not None

    researcher = copy.deepcopy(doc["subagent_trajectories"][0])
    del researcher["extra"]["lmnr"]["spec_version"]
    sub = TrajectoryExtension.read(researcher, parent=root)
    assert sub is not None
    assert sub.spec_version == SPEC_VERSION
    assert TrajectoryExtension.read(researcher) is None

    # The researcher's `$ref` lives in the root's defs (§6.6 ancestor lookup).
    tools = reader.tool_definitions(researcher, researcher["steps"][2], sub)
    assert tools is not None
    assert [t["function"]["name"] for t in tools] == ["web_search"]

    bare = {"steps": [{"step_id": 1, "source": "user", "message": "hi"}]}
    assert TrajectoryExtension.read(bare, parent=root) is not None


def test_unresolved_ref_is_absent():
    doc = _load("overrides.json")
    ext = TrajectoryExtension.read(doc)
    assert ext is not None
    assert ext.resolve({"$ref": "sha256:missing"}, "system_prompts") is None
    assert ext.resolve("inline", "system_prompts") == "inline"
    assert ext.resolve(None, "system_prompts") is None


# -- resolution -----------------------------------------------------------------


def test_system_prompt_resolution():
    doc = _load("overrides.json")
    ext = TrajectoryExtension.read(doc)
    steps = reader.steps_of(doc)
    assert reader.leading_system_count(steps) == 2

    # Default: the leading system steps, one message each, roles preserved.
    assert reader.system_prompt(steps, 3, ext) == [
        {"role": "system", "content": "You are a travel agent."},
        {"role": "developer", "content": "Always quote prices in EUR."},
    ]
    # Override through a $ref; not sticky.
    assert reader.system_prompt(steps, 4, ext) == [
        {
            "role": "system",
            "content": doc["extra"]["lmnr"]["defs"]["system_prompts"][
                doc["steps"][4]["extra"]["lmnr"]["system_prompt"]["$ref"]
            ],
        }
    ]
    assert len(reader.system_prompt(steps, 6, ext)) == 2
    # A mid-history system step (step 8) never changes the system prompt.
    assert len(reader.system_prompt(steps, 9, ext)) == 2

    # Plain ATIF ignores overrides and keeps roles as `system`.
    assert reader.system_prompt(steps, 4, None) == [
        {"role": "system", "content": "You are a travel agent."},
        {"role": "system", "content": "Always quote prices in EUR."},
    ]


def test_explicit_empty_system_prompt_and_tools():
    doc = _load("overrides.json")
    doc["steps"][4]["extra"]["lmnr"]["system_prompt"] = ""
    ext = TrajectoryExtension.read(doc)
    steps = reader.steps_of(doc)
    assert reader.system_prompt(steps, 4, ext) == []
    # Step 5 offers no tools (`[]`), unlike the agent default.
    assert reader.tool_definitions(doc, steps[4], ext) == []
    assert reader.tool_definitions(doc, steps[4], None) == doc["agent"]["tool_definitions"]


def test_tool_definitions_resolution():
    doc = _load("overrides.json")
    ext = TrajectoryExtension.read(doc)
    steps = reader.steps_of(doc)
    default = reader.tool_definitions(doc, steps[3], ext)
    assert default is not None
    assert [t["function"]["name"] for t in default] == ["search_flights", "search_hotels"]
    narrowed = reader.tool_definitions(doc, steps[6], ext)
    assert narrowed is not None
    assert [t["function"]["name"] for t in narrowed] == ["search_hotels"]

    no_tools = {"agent": {"name": "a", "version": "1"}, "steps": []}
    assert reader.tool_definitions(no_tools, {"source": "agent"}, None) is None


def test_input_messages_from_history():
    doc = _load("basic-agent-loop.json")
    ext = TrajectoryExtension.read(doc)
    steps = reader.steps_of(doc)
    messages = reader.input_messages(steps, 3, ext)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool"]
    assistant = messages[2]
    assert assistant["content"] == ""
    assert assistant["reasoning_content"].startswith("The user wants a refund")
    assert assistant["tool_calls"][0]["function_name"] == "get_order"
    assert messages[3] == {
        "role": "tool",
        "tool_call_id": "toolu_01",
        "content": '{"order_id": "A-1042", "status": "shipped"}',
    }
    assert [m["role"] for m in reader.input_messages(steps, 2, ext)] == ["system", "user"]


def test_input_messages_override_replaces_history():
    doc = _load("overrides.json")
    ext = TrajectoryExtension.read(doc)
    steps = reader.steps_of(doc)
    messages = reader.input_messages(steps, 9, ext)
    assert messages == [
        {"role": "system", "content": "You are a travel agent."},
        {"role": "developer", "content": "Always quote prices in EUR."},
        {"role": "system", "content": "Earlier turns were compacted into a summary."},
        {"role": "user", "content": doc["steps"][8]["message"]},
    ]
    # A plain-ATIF reader gets the whole linear history instead.
    plain = reader.input_messages(steps, 9, None)
    assert len(plain) > len(messages)
    assert plain[-2:] == [
        {"role": "system", "content": "Earlier turns were compacted into a summary."},
        {"role": "user", "content": doc["steps"][8]["message"]},
    ]


def test_input_messages_override_is_the_base_for_later_steps():
    """A step after an override extends it, not the whole history (§6.4 rule 2)."""
    doc = _load("overrides.json")
    ext = TrajectoryExtension.read(doc)
    steps = reader.steps_of(doc)
    assert "input_messages" not in steps[11]["extra"]["lmnr"]
    assert reader.input_messages(steps, 11, ext) == [
        {"role": "system", "content": "You are a travel agent."},
        {"role": "developer", "content": "Always quote prices in EUR."},
        {"role": "system", "content": "Earlier turns were compacted into a summary."},
        {"role": "user", "content": doc["steps"][8]["message"]},
        {"role": "assistant", "content": doc["steps"][9]["message"]},
        {"role": "user", "content": "Any cheaper one?"},
    ]
    # Steps before the override still see the whole history.
    assert len(reader.input_messages(steps, 6, ext)) == 7


def test_synthetic_steps_add_no_messages():
    doc = _load("errors-and-dispatch.json")
    ext = TrajectoryExtension.read(doc)
    steps = reader.steps_of(doc)
    assert reader.is_synthetic(steps[2], ext)
    assert reader.step_messages(steps[2], ext) == []
    roles = [m["role"] for m in reader.input_messages(steps, 3, ext)]
    assert roles == ["user", "assistant", "tool", "tool"]
    # Without the extension, the dispatch step looks like any agent step.
    assert len(reader.input_messages(steps, 3, None)) == 6


def test_observation_without_call_id_is_a_user_message():
    step = {
        "step_id": 1,
        "source": "agent",
        "message": "ls",
        "observation": {"results": [{"content": "file.txt"}]},
    }
    assert reader.step_messages(step, None) == [
        {"role": "assistant", "content": "ls"},
        {"role": "user", "content": "file.txt"},
    ]


# -- errors and usage ---------------------------------------------------------------


def test_step_and_tool_errors():
    doc = _load("errors-and-dispatch.json")
    ext = TrajectoryExtension.read(doc)
    steps = reader.steps_of(doc)
    assert reader.step_error(steps[1], ext) is None
    assert reader.step_error(steps[3], ext) == {
        "type": "RateLimitError",
        "message": "Error code: 429 - rate_limit_error",
    }
    assert reader.step_error(steps[3], None) is None

    results = reader.observation_results(steps[1])
    assert reader.result_error(results[0], ext) == {
        "type": "TimeoutError",
        "message": "tests did not finish in 60s",
    }
    assert reader.result_error(reader.observation_results(steps[2])[0], ext) is None


@pytest.mark.parametrize("flag", ["tool_result_is_error", "is_error"])
def test_tool_errors_from_harbor_flags(flag: str):
    result = {"source_call_id": "c1", "content": "boom", "extra": {flag: True}}
    assert reader.result_error(result, None) == {"type": "ToolError", "message": ""}
    assert reader.result_error({**result, "extra": {flag: False}}, None) is None


def test_usage_extension_with_fallbacks():
    doc = _load("basic-agent-loop.json")
    ext = TrajectoryExtension.read(doc)
    metrics = doc["steps"][2]["metrics"]
    assert reader.usage_extension(metrics, ext) == {
        "cache_creation_tokens": 300,
        "input_cost_usd": 0.00132,
        "output_cost_usd": 0.00096,
    }
    assert reader.usage_extension(metrics, None) == {}

    harbor_metrics = {
        "prompt_tokens": 10,
        "extra": {"reasoning_tokens": 7, "cache_creation_input_tokens": 3},
    }
    assert reader.usage_extension(harbor_metrics, None) == {
        "reasoning_tokens": 7,
        "cache_creation_tokens": 3,
    }
    assert reader.usage_extension(None, ext) == {}


def test_spec_doc_field_tables_match_schema():
    """Every key in the doc's field tables is in the schema, and vice versa."""
    doc = (SPEC_DIR / "lmnr-atif.md").read_text()
    defs = _schema()["$defs"]
    sections = {
        "### 4.1": "trajectoryLmnr",
        "### 4.2": "llmStepLmnr",
        "### 4.3": "syntheticStepLmnr",
        "### 4.4": "messageStepLmnr",
        "### 4.5": "toolCallLmnr",
        "### 4.6": "observationResultLmnr",
        "### 4.7": "metricsLmnr",
        "### 4.8": "finalMetricsLmnr",
        "### 4.9": "subagentRefLmnr",
    }
    for heading, name in sections.items():
        section = doc.split(heading, 1)[1].split("\n### ", 1)[0].split("\n## ", 1)[0]
        documented: set[str] = set()
        for cell in re.findall(r"^\| ([^|]+) \|", section, flags=re.MULTILINE):
            documented.update(re.findall(r"`([a-z_]+)`", cell))
        assert documented == set(defs[name]["properties"]), heading
