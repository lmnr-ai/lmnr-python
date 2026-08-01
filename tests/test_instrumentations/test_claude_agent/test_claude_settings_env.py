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
    def __init__(self, env=None, settings=None):
        self.env = env or {}
        self.settings = settings


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
