"""Tests for Claude Code settings.json handling (lmnr-ai/lmnr#2167).

Claude Code resolves its ``env`` block from settings files with HIGHER priority
than the subprocess environment, so rewriting ``options.env`` alone leaves the
proxy idle when a user keeps ``ANTHROPIC_BASE_URL`` in ``~/.claude/settings.json``.
"""

import json

import pytest

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.claude_agent.utils import (
    build_proxy_flag_settings,
    read_claude_settings_env,
    resolve_target_url_from_env,
    FOUNDRY_BASE_URL_ENV,
    FOUNDRY_RESOURCE_ENV,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.claude_agent.wrappers import (
    apply_settings_proxy_override,
)

PROXY_URL = "http://127.0.0.1:45667"
UPSTREAM = "https://gateway.example.com"


class MockOptions:
    """Mock ClaudeAgentOptions."""

    def __init__(self, env=None, settings=None, cwd=None):
        self.env = env or {}
        self.settings = settings
        self.cwd = cwd


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    """Point Claude's user config dir and session cwd at empty temp dirs."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

    session_dir = tmp_path / "session"
    (session_dir / ".claude").mkdir(parents=True)
    monkeypatch.chdir(session_dir)

    for key in (
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_ORIGINAL_BASE_URL",
        "ANTHROPIC_FOUNDRY_BASE_URL",
        "ANTHROPIC_FOUNDRY_RESOURCE",
        "ANTHROPIC_BEDROCK_BASE_URL",
        "ANTHROPIC_VERTEX_BASE_URL",
        "CLAUDE_CODE_USE_FOUNDRY",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "HTTP_PROXY",
        "HTTPS_PROXY",
    ):
        monkeypatch.delenv(key, raising=False)

    return config_dir, session_dir


def write_settings(path, env):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"env": env}))


def test_reads_user_settings_env(isolated_settings):
    config_dir, _ = isolated_settings
    write_settings(config_dir / "settings.json", {"ANTHROPIC_BASE_URL": UPSTREAM})

    assert read_claude_settings_env()["ANTHROPIC_BASE_URL"] == UPSTREAM


def test_local_settings_outrank_project_and_user(isolated_settings):
    config_dir, session_dir = isolated_settings
    write_settings(config_dir / "settings.json", {"ANTHROPIC_BASE_URL": "https://user"})
    write_settings(
        session_dir / ".claude" / "settings.json",
        {"ANTHROPIC_BASE_URL": "https://project"},
    )
    write_settings(
        session_dir / ".claude" / "settings.local.json",
        {"ANTHROPIC_BASE_URL": "https://local"},
    )

    assert read_claude_settings_env()["ANTHROPIC_BASE_URL"] == "https://local"


def test_missing_and_malformed_settings_are_ignored(isolated_settings):
    config_dir, _ = isolated_settings
    (config_dir / "settings.json").write_text("{not json")

    assert read_claude_settings_env() == {}


def test_settings_base_url_becomes_proxy_upstream(isolated_settings):
    """The proxy must forward to the gateway the user configured in settings."""
    config_dir, _ = isolated_settings
    write_settings(config_dir / "settings.json", {"ANTHROPIC_BASE_URL": UPSTREAM})

    assert resolve_target_url_from_env({}) == UPSTREAM


def test_options_env_still_outranks_settings_for_upstream(isolated_settings):
    config_dir, _ = isolated_settings
    write_settings(config_dir / "settings.json", {"ANTHROPIC_BASE_URL": UPSTREAM})

    url = resolve_target_url_from_env({"ANTHROPIC_BASE_URL": "https://explicit"})

    assert url == "https://explicit"


def test_flag_settings_pin_base_url_to_proxy(isolated_settings):
    config_dir, _ = isolated_settings
    write_settings(config_dir / "settings.json", {"ANTHROPIC_BASE_URL": UPSTREAM})

    settings = json.loads(build_proxy_flag_settings(None, PROXY_URL))

    assert settings["env"]["ANTHROPIC_BASE_URL"] == PROXY_URL


def test_flag_settings_preserve_unrelated_user_settings(isolated_settings):
    """Only base URLs are rewritten; other keys merge from the lower layers."""
    existing = json.dumps({"permissions": {"allow": ["Bash(*)"]}, "env": {"FOO": "bar"}})

    settings = json.loads(build_proxy_flag_settings(existing, PROXY_URL))

    assert settings["permissions"] == {"allow": ["Bash(*)"]}
    assert settings["env"]["FOO"] == "bar"
    assert settings["env"]["ANTHROPIC_BASE_URL"] == PROXY_URL


def test_flag_settings_blank_redirecting_keys(isolated_settings):
    """Settings layers merge per key, so a bypass key must be blanked, not dropped."""
    config_dir, _ = isolated_settings
    write_settings(
        config_dir / "settings.json",
        {"ANTHROPIC_BASE_URL": UPSTREAM, "HTTPS_PROXY": "http://corp:8080"},
    )

    settings = json.loads(build_proxy_flag_settings(None, PROXY_URL))

    assert settings["env"]["HTTPS_PROXY"] == ""


def test_flag_settings_do_not_invent_provider_base_urls(isolated_settings):
    """A user who never configured Bedrock must not get a Bedrock base URL."""
    settings = json.loads(build_proxy_flag_settings(None, PROXY_URL))

    assert "ANTHROPIC_BEDROCK_BASE_URL" not in settings["env"]
    assert "ANTHROPIC_VERTEX_BASE_URL" not in settings["env"]


def test_flag_settings_pin_provider_base_url_when_configured(isolated_settings):
    config_dir, _ = isolated_settings
    write_settings(
        config_dir / "settings.json",
        {"CLAUDE_CODE_USE_BEDROCK": "1", "ANTHROPIC_BEDROCK_BASE_URL": UPSTREAM},
    )

    settings = json.loads(build_proxy_flag_settings(None, PROXY_URL))

    assert settings["env"]["ANTHROPIC_BEDROCK_BASE_URL"] == PROXY_URL


def test_flag_settings_pin_foundry_base_url_for_resource_only_setup(isolated_settings):
    """Foundry can be set up by resource alone, with no base-URL key.

    Blanking the resource without pinning a base URL would strip its only routing
    key and leave nothing for the CLI to talk to.
    """
    config_dir, _ = isolated_settings
    write_settings(
        config_dir / "settings.json",
        {"CLAUDE_CODE_USE_FOUNDRY": "1", "ANTHROPIC_FOUNDRY_RESOURCE": "myresource"},
    )

    settings = json.loads(build_proxy_flag_settings(None, PROXY_URL))

    assert settings["env"][FOUNDRY_BASE_URL_ENV] == PROXY_URL
    assert settings["env"][FOUNDRY_RESOURCE_ENV] == ""


def test_flag_settings_pin_provider_base_url_from_enabling_flag(isolated_settings):
    config_dir, _ = isolated_settings
    write_settings(config_dir / "settings.json", {"CLAUDE_CODE_USE_VERTEX": "1"})

    settings = json.loads(build_proxy_flag_settings(None, PROXY_URL))

    assert settings["env"]["ANTHROPIC_VERTEX_BASE_URL"] == PROXY_URL


def test_flag_settings_merge_a_settings_file_path(isolated_settings, tmp_path):
    settings_file = tmp_path / "custom.json"
    settings_file.write_text(json.dumps({"model": "sonnet", "env": {"KEEP": "1"}}))

    settings = json.loads(
        build_proxy_flag_settings(str(settings_file), PROXY_URL)
    )

    assert settings["model"] == "sonnet"
    assert settings["env"]["KEEP"] == "1"
    assert settings["env"]["ANTHROPIC_BASE_URL"] == PROXY_URL


def test_unreadable_settings_path_is_left_untouched(isolated_settings):
    """Never replace a path we can't read — the CLI may still resolve it."""
    assert build_proxy_flag_settings("/nonexistent/settings.json", PROXY_URL) is None


def test_apply_override_leaves_unreadable_path_in_place(isolated_settings):
    options = MockOptions(settings="/nonexistent/settings.json")

    assert apply_settings_proxy_override(options, PROXY_URL) is None
    assert options.settings == "/nonexistent/settings.json"


def test_apply_override_returns_snapshot_for_restoration(isolated_settings):
    original = json.dumps({"model": "sonnet"})
    options = MockOptions(settings=original)

    snapshot = apply_settings_proxy_override(options, PROXY_URL)

    assert snapshot == (original,)
    assert json.loads(options.settings)["env"]["ANTHROPIC_BASE_URL"] == PROXY_URL

    options.settings = snapshot[0]
    assert options.settings == original


def test_apply_override_restores_none_settings(isolated_settings):
    """A previously-unset settings field must go back to None, not an empty blob."""
    options = MockOptions()

    snapshot = apply_settings_proxy_override(options, PROXY_URL)

    assert snapshot == (None,)
    options.settings = snapshot[0]
    assert options.settings is None


def test_on_disk_settings_are_never_modified(isolated_settings):
    config_dir, _ = isolated_settings
    settings_path = config_dir / "settings.json"
    write_settings(settings_path, {"ANTHROPIC_BASE_URL": UPSTREAM})
    before = settings_path.read_text()

    apply_settings_proxy_override(MockOptions(), PROXY_URL)

    assert settings_path.read_text() == before


def test_issue_2167_end_to_end(isolated_settings):
    """The reported scenario: base URL in user settings, proxy must still win."""
    config_dir, _ = isolated_settings
    write_settings(
        config_dir / "settings.json",
        {"ANTHROPIC_BASE_URL": UPSTREAM, "ANTHROPIC_API_KEY": "sk-user"},
    )
    options = MockOptions()

    target_url = resolve_target_url_from_env(options.env)
    apply_settings_proxy_override(options, PROXY_URL)

    # Proxy forwards to the user's real gateway...
    assert target_url == UPSTREAM
    # ...and the CLI is redirected to the proxy at the highest settings layer.
    assert json.loads(options.settings)["env"]["ANTHROPIC_BASE_URL"] == PROXY_URL
    # The user's API key is untouched, so it still merges in from their settings.
    assert "ANTHROPIC_API_KEY" not in json.loads(options.settings)["env"]
