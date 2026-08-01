"""Tests for Claude Code settings.json env handling with the auto-instrumentation proxy."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.claude_agent import (
    proxy as claude_proxy,
    utils as claude_utils,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.claude_agent.utils import (
    read_claude_settings_env,
    read_claude_user_settings_env,
    resolve_target_url_from_env,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.claude_agent.wrappers import (
    apply_settings_env_proxy_override,
    restore_options_settings,
    update_options_env_for_proxy,
    wrap_transport_connect,
)
from lmnr_claude_code_proxy import ProxyServer


class MockOptions:
    def __init__(self, env=None, settings=None, cwd=None):
        self.env = env or {}
        self.settings = settings
        self.cwd = cwd


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_ORIGINAL_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_RESOURCE", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_USE_FOUNDRY", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_USE_BEDROCK", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_USE_VERTEX", raising=False)
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)


def test_read_claude_user_settings_env(tmp_path, monkeypatch, clean_env):
    """settings.json env block is loaded from CLAUDE_CONFIG_DIR."""
    config_dir = tmp_path / "claude"
    config_dir.mkdir()
    settings = {
        "env": {
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:30233",
            "ANTHROPIC_API_KEY": "test-key",
        },
        "model": "claude-sonnet-4-5",
    }
    (config_dir / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

    env = read_claude_user_settings_env()
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:30233"
    assert env["ANTHROPIC_API_KEY"] == "test-key"


def test_read_claude_user_settings_env_missing_file(tmp_path, monkeypatch, clean_env):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "missing"))
    assert read_claude_user_settings_env() == {}


def test_resolve_target_url_from_settings_when_not_in_process_env(
    tmp_path, monkeypatch, clean_env
):
    """Upstream from settings.json is used when process env has no base URL."""
    config_dir = tmp_path / "claude"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:30233"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

    assert resolve_target_url_from_env({}) == "http://127.0.0.1:30233"


def test_options_env_beats_settings_for_target_url(tmp_path, monkeypatch, clean_env):
    config_dir = tmp_path / "claude"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://settings.example"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

    assert (
        resolve_target_url_from_env({"ANTHROPIC_BASE_URL": "http://options.example"})
        == "http://options.example"
    )


def test_apply_settings_env_proxy_override_injects_flag_settings(clean_env):
    options = MockOptions()
    update_options_env_for_proxy(
        options, "http://127.0.0.1:45667", "http://127.0.0.1:30233"
    )
    apply_settings_env_proxy_override(options, "http://127.0.0.1:45667")

    assert options.settings is not None
    parsed = json.loads(options.settings)
    assert parsed["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:45667"


def test_apply_settings_env_proxy_override_merges_existing_json_settings(clean_env):
    options = MockOptions(
        settings=json.dumps(
            {
                "permissions": {"defaultMode": "acceptEdits"},
                "env": {"ANTHROPIC_MODEL": "claude-opus-4-5"},
            }
        )
    )
    update_options_env_for_proxy(
        options, "http://127.0.0.1:45667", "https://api.anthropic.com"
    )
    apply_settings_env_proxy_override(options, "http://127.0.0.1:45667")

    parsed = json.loads(options.settings)
    assert parsed["permissions"]["defaultMode"] == "acceptEdits"
    assert parsed["env"]["ANTHROPIC_MODEL"] == "claude-opus-4-5"
    assert parsed["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:45667"


def test_apply_settings_env_proxy_override_merges_settings_file(
    tmp_path, monkeypatch, clean_env
):
    settings_path = tmp_path / "extra-settings.json"
    settings_path.write_text(
        json.dumps({"env": {"FOO": "bar"}, "model": "claude-sonnet-4-5"}),
        encoding="utf-8",
    )
    options = MockOptions(settings=str(settings_path))
    update_options_env_for_proxy(
        options, "http://127.0.0.1:45667", "https://api.anthropic.com"
    )
    apply_settings_env_proxy_override(options, "http://127.0.0.1:45667")

    parsed = json.loads(options.settings)
    assert parsed["model"] == "claude-sonnet-4-5"
    assert parsed["env"]["FOO"] == "bar"
    assert parsed["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:45667"


def test_restore_options_settings(clean_env):
    options = MockOptions(settings=None)
    apply_settings_env_proxy_override(options, "http://127.0.0.1:45667")
    assert options.settings is not None
    restore_options_settings(options, None)
    assert options.settings is None

    options.settings = '{"env":{}}'
    restore_options_settings(options, "/tmp/original.json")
    assert options.settings == "/tmp/original.json"


def test_wrap_transport_connect_overrides_settings_base_url(
    tmp_path, monkeypatch, clean_env
):
    """
    Repro for lmnr-ai/lmnr#2167: ANTHROPIC_BASE_URL in settings.json must not
    prevent the CLI from using the local auto-instrumentation proxy.
    """
    from claude_agent_sdk._internal.transport.subprocess_cli import (
        SubprocessCLITransport,
    )

    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.claude_agent import (
        wrappers as claude_wrappers,
        span_utils,
    )

    config_dir = tmp_path / "claude"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        json.dumps(
            {
                "env": {
                    "ANTHROPIC_BASE_URL": "http://127.0.0.1:30233",
                    "ANTHROPIC_API_KEY": "test-key",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    # Process env may also have the custom URL (as in the issue repro).
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:30233")

    transport = MagicMock(spec=SubprocessCLITransport)
    transport._options = MockOptions(env={})

    captured = {}

    async def mock_connect(*args, **kwargs):
        captured["env"] = dict(transport._options.env)
        captured["settings"] = transport._options.settings
        return None

    original_connect = AsyncMock(side_effect=mock_connect)
    wrapper = wrap_transport_connect({"original": original_connect})

    # Patch names bound in wrappers (from .proxy import ...).
    with (
        patch.object(claude_wrappers, "create_proxy_for_transport") as mock_create,
        patch.object(claude_wrappers, "start_proxy") as mock_start,
        patch.object(span_utils, "publish_span_context_for_transport"),
    ):
        mock_proxy = MagicMock(spec=ProxyServer)
        mock_create.return_value = mock_proxy
        mock_start.return_value = "http://127.0.0.1:45667"

        asyncio.run(wrapper(original_connect, transport, (), {}))

        # Proxy must forward to the settings/custom upstream, not default Anthropic.
        mock_start.assert_called_once_with(
            mock_proxy, target_url="http://127.0.0.1:30233"
        )

    assert captured["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:45667"
    assert captured["env"]["ANTHROPIC_ORIGINAL_BASE_URL"] == "http://127.0.0.1:30233"

    settings = json.loads(captured["settings"])
    assert settings["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:45667"

    # Disk settings remain unchanged (no mutation of user config).
    on_disk = json.loads((config_dir / "settings.json").read_text(encoding="utf-8"))
    assert on_disk["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:30233"


def test_foundry_from_settings_enables_proxy_base_url(
    tmp_path, monkeypatch, clean_env
):
    config_dir = tmp_path / "claude"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        json.dumps(
            {
                "env": {
                    "CLAUDE_CODE_USE_FOUNDRY": "1",
                    "ANTHROPIC_FOUNDRY_BASE_URL": "https://foundry.example.com",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

    assert resolve_target_url_from_env({}) == "https://foundry.example.com"

    options = MockOptions()
    update_options_env_for_proxy(
        options, "http://127.0.0.1:45667", "https://foundry.example.com"
    )
    apply_settings_env_proxy_override(options, "http://127.0.0.1:45667")

    assert options.env[claude_utils.FOUNDRY_BASE_URL_ENV] == "http://127.0.0.1:45667"
    parsed = json.loads(options.settings)
    assert parsed["env"][claude_utils.FOUNDRY_BASE_URL_ENV] == "http://127.0.0.1:45667"


def test_resolve_target_url_from_project_settings(tmp_path, monkeypatch, clean_env):
    """Project .claude/settings.json base URL is used before default Anthropic."""
    # Isolate user settings so only project layer applies.
    user_dir = tmp_path / "user-claude"
    user_dir.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(user_dir))

    project = tmp_path / "project"
    claude_dir = project / ".claude"
    claude_dir.mkdir(parents=True)
    (claude_dir / "settings.json").write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://team-gateway.example/v1"}}),
        encoding="utf-8",
    )

    assert (
        resolve_target_url_from_env({}, cwd=project)
        == "https://team-gateway.example/v1"
    )


def test_local_settings_override_project_and_user(tmp_path, monkeypatch, clean_env):
    """Claude priority: local > project > user for settings env keys."""
    user_dir = tmp_path / "user-claude"
    user_dir.mkdir()
    (user_dir / "settings.json").write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://user.example"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(user_dir))

    project = tmp_path / "project"
    claude_dir = project / ".claude"
    claude_dir.mkdir(parents=True)
    (claude_dir / "settings.json").write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://project.example"}}),
        encoding="utf-8",
    )
    (claude_dir / "settings.local.json").write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://local.example"}}),
        encoding="utf-8",
    )

    env = read_claude_settings_env(project)
    assert env["ANTHROPIC_BASE_URL"] == "https://local.example"
    assert resolve_target_url_from_env({}, cwd=project) == "https://local.example"

    # Without local file, project wins over user.
    (claude_dir / "settings.local.json").unlink()
    assert resolve_target_url_from_env({}, cwd=project) == "https://project.example"


def test_project_settings_used_as_proxy_upstream(tmp_path, monkeypatch, clean_env):
    """
    Flag-level proxy injection must still forward to project settings base URL.

    Regression for Bugbot: resolve only from user settings while always injecting
    flag ANTHROPIC_BASE_URL caused team gateways in project settings to misroute
    to default Anthropic.
    """
    from claude_agent_sdk._internal.transport.subprocess_cli import (
        SubprocessCLITransport,
    )

    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.claude_agent import (
        wrappers as claude_wrappers,
        span_utils,
    )

    user_dir = tmp_path / "user-claude"
    user_dir.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(user_dir))

    project = tmp_path / "project"
    claude_dir = project / ".claude"
    claude_dir.mkdir(parents=True)
    (claude_dir / "settings.json").write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://team-gateway.example/v1"}}),
        encoding="utf-8",
    )

    transport = MagicMock(spec=SubprocessCLITransport)
    transport._options = MockOptions(env={}, cwd=str(project))

    captured = {}

    async def mock_connect(*args, **kwargs):
        captured["env"] = dict(transport._options.env)
        captured["settings"] = transport._options.settings
        return None

    original_connect = AsyncMock(side_effect=mock_connect)
    wrapper = wrap_transport_connect({"original": original_connect})

    with (
        patch.object(claude_wrappers, "create_proxy_for_transport") as mock_create,
        patch.object(claude_wrappers, "start_proxy") as mock_start,
        patch.object(span_utils, "publish_span_context_for_transport"),
    ):
        mock_proxy = MagicMock(spec=ProxyServer)
        mock_create.return_value = mock_proxy
        mock_start.return_value = "http://127.0.0.1:45667"

        asyncio.run(wrapper(original_connect, transport, (), {}))

        mock_start.assert_called_once_with(
            mock_proxy, target_url="https://team-gateway.example/v1"
        )

    assert captured["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:45667"
    assert (
        captured["env"]["ANTHROPIC_ORIGINAL_BASE_URL"]
        == "https://team-gateway.example/v1"
    )


def test_relative_settings_path_resolves_against_options_cwd(
    tmp_path, monkeypatch, clean_env
):
    """Relative options.settings path is resolved vs options.cwd, not process cwd."""
    project = tmp_path / "project"
    project.mkdir()
    settings_path = project / "agent-settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "permissions": {"defaultMode": "acceptEdits"},
                "env": {"ANTHROPIC_MODEL": "claude-opus-4-5", "FOO": "from-file"},
            }
        ),
        encoding="utf-8",
    )

    # Process cwd is elsewhere; path is relative to options.cwd.
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)

    options = MockOptions(settings="agent-settings.json", cwd=str(project))
    update_options_env_for_proxy(
        options, "http://127.0.0.1:45667", "https://api.anthropic.com"
    )
    apply_settings_env_proxy_override(options, "http://127.0.0.1:45667")

    parsed = json.loads(options.settings)
    assert parsed["permissions"]["defaultMode"] == "acceptEdits"
    assert parsed["env"]["FOO"] == "from-file"
    assert parsed["env"]["ANTHROPIC_MODEL"] == "claude-opus-4-5"
    assert parsed["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:45667"


def test_missing_settings_path_preserves_original_path(tmp_path, monkeypatch, clean_env):
    """Missing settings file path must not be clobbered with proxy-only JSON."""
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)

    original_path = "missing-settings.json"
    options = MockOptions(settings=original_path, cwd=str(tmp_path / "no-such-project"))
    update_options_env_for_proxy(
        options, "http://127.0.0.1:45667", "https://api.anthropic.com"
    )
    apply_settings_env_proxy_override(options, "http://127.0.0.1:45667")

    assert options.settings == original_path
    # options.env still points at the proxy.
    assert options.env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:45667"
