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


def test_setting_sources_gates_which_layers_are_read(isolated_settings):
    """The CLI honors options.setting_sources, so we must too.

    Reading a layer the CLI was told to ignore would resolve an upstream the CLI
    never uses, pointing the proxy at an unintended host.
    """
    config_dir, session_dir = isolated_settings
    write_settings(config_dir / "settings.json", {"ANTHROPIC_BASE_URL": "https://user"})
    write_settings(
        session_dir / ".claude" / "settings.json",
        {"ANTHROPIC_BASE_URL": "https://project"},
    )

    assert (
        read_claude_settings_env(setting_sources=["user"])["ANTHROPIC_BASE_URL"]
        == "https://user"
    )
    assert (
        read_claude_settings_env(setting_sources=["project"])["ANTHROPIC_BASE_URL"]
        == "https://project"
    )


def test_empty_setting_sources_disables_on_disk_settings(isolated_settings):
    config_dir, _ = isolated_settings
    write_settings(config_dir / "settings.json", {"ANTHROPIC_BASE_URL": UPSTREAM})

    assert read_claude_settings_env(setting_sources=[]) == {}


def test_upstream_ignores_a_layer_the_cli_will_not_load(isolated_settings):
    config_dir, session_dir = isolated_settings
    write_settings(config_dir / "settings.json", {})
    write_settings(
        session_dir / ".claude" / "settings.json", {"ANTHROPIC_BASE_URL": UPSTREAM}
    )

    # The CLI only loads user settings, so the project gateway must be invisible.
    url = resolve_target_url_from_env({}, setting_sources=["user"])

    assert url == "https://api.anthropic.com"


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


def test_lowercase_proxy_var_is_blanked(isolated_settings):
    """Claude Code reads the lowercase spelling too (and prefers it).

    Handling only the uppercase form lets a lowercase corporate proxy in settings
    divert traffic away from our proxy.
    """
    config_dir, _ = isolated_settings
    write_settings(
        config_dir / "settings.json",
        {"ANTHROPIC_BASE_URL": UPSTREAM, "https_proxy": "http://corp:8080"},
    )

    settings = json.loads(build_proxy_flag_settings(None, PROXY_URL))

    assert settings["env"]["https_proxy"] == ""


def test_lowercase_settings_proxy_var_does_not_shadow_the_gateway(isolated_settings):
    config_dir, _ = isolated_settings
    write_settings(
        config_dir / "settings.json",
        {"ANTHROPIC_BASE_URL": UPSTREAM, "https_proxy": "http://corp:8080"},
    )

    assert resolve_target_url_from_env({}) == UPSTREAM


def test_lowercase_process_env_proxy_var_is_the_upstream(isolated_settings, monkeypatch):
    """Pre-existing semantics: a proxy var in the real env IS the target."""
    monkeypatch.setenv("https_proxy", "http://corp:8080")

    assert resolve_target_url_from_env({}) == "http://corp:8080"


@pytest.mark.parametrize("value", ["1", "true", "True", " on ", "yes"])
def test_provider_flag_truthy_values_pin_the_base_url(isolated_settings, value):
    """The CLI accepts 1/true/yes/on — verified against the bundled binary.

    Missing one would blank the Foundry resource without pinning a base URL,
    stripping the only routing key and breaking the run.
    """
    config_dir, _ = isolated_settings
    write_settings(
        config_dir / "settings.json",
        {"CLAUDE_CODE_USE_FOUNDRY": value, "ANTHROPIC_FOUNDRY_RESOURCE": "myres"},
    )

    settings = json.loads(build_proxy_flag_settings(None, PROXY_URL))

    assert settings["env"][FOUNDRY_BASE_URL_ENV] == PROXY_URL
    assert settings["env"][FOUNDRY_RESOURCE_ENV] == ""


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_provider_flag_falsy_values_do_not_pin_a_base_url(isolated_settings, value):
    config_dir, _ = isolated_settings
    write_settings(config_dir / "settings.json", {"CLAUDE_CODE_USE_VERTEX": value})

    settings = json.loads(build_proxy_flag_settings(None, PROXY_URL))

    assert "ANTHROPIC_VERTEX_BASE_URL" not in settings["env"]


def test_settings_proxy_var_does_not_shadow_the_gateway(isolated_settings):
    """HTTP_PROXY / HTTPS_PROXY are forward proxies, not API bases.

    They outrank every base URL, so taking them from settings would make a
    corporate proxy shadow the gateway configured right beside it and the
    interception proxy would forward API calls to the wrong host.
    """
    config_dir, _ = isolated_settings
    write_settings(
        config_dir / "settings.json",
        {"ANTHROPIC_BASE_URL": UPSTREAM, "HTTPS_PROXY": "http://corp:8080"},
    )

    assert resolve_target_url_from_env({}) == UPSTREAM


def test_settings_proxy_var_alone_falls_back_to_default(isolated_settings):
    config_dir, _ = isolated_settings
    write_settings(config_dir / "settings.json", {"HTTPS_PROXY": "http://corp:8080"})

    assert resolve_target_url_from_env({}) == "https://api.anthropic.com"


def test_process_env_proxy_var_is_still_the_upstream(isolated_settings, monkeypatch):
    """Pre-existing behavior: a proxy var in the real env IS the target."""
    config_dir, _ = isolated_settings
    write_settings(config_dir / "settings.json", {"ANTHROPIC_BASE_URL": UPSTREAM})
    monkeypatch.setenv("HTTPS_PROXY", "http://corp:8080")

    assert resolve_target_url_from_env({}) == "http://corp:8080"


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


def test_foundry_resource_is_blanked_when_enabled_only_in_settings(
    isolated_settings, monkeypatch
):
    """The resource is mutually exclusive with the base URL we pin.

    The CLI hard-fails with "baseURL and resource are mutually exclusive" if both
    are live, so a resource left in the process env must still be blanked.
    """
    config_dir, _ = isolated_settings
    write_settings(config_dir / "settings.json", {"CLAUDE_CODE_USE_FOUNDRY": "1"})
    monkeypatch.setenv(FOUNDRY_RESOURCE_ENV, "stray-resource")

    settings = json.loads(build_proxy_flag_settings(None, PROXY_URL))

    assert settings["env"][FOUNDRY_BASE_URL_ENV] == PROXY_URL
    assert settings["env"][FOUNDRY_RESOURCE_ENV] == ""


def test_process_env_proxy_var_is_blanked_in_flag_settings(isolated_settings, monkeypatch):
    """A proxy var in the process env must not route the CLI around our proxy."""
    monkeypatch.setenv("HTTPS_PROXY", "http://corp:8080")

    settings = json.loads(build_proxy_flag_settings(None, PROXY_URL))

    assert settings["env"]["HTTPS_PROXY"] == ""


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


def test_malformed_settings_file_is_left_untouched(isolated_settings, tmp_path):
    """An existing but unparseable file must not be replaced.

    Emitting a proxy-only blob here would silently drop the model / permissions
    the user configured for this run.
    """
    settings_file = tmp_path / "trailing_comma.json"
    settings_file.write_text('{"model": "sonnet", "permissions": {"allow": ["Bash(*)"]},}')

    assert build_proxy_flag_settings(str(settings_file), PROXY_URL) is None


def test_non_object_settings_file_is_left_untouched(isolated_settings, tmp_path):
    settings_file = tmp_path / "array.json"
    settings_file.write_text('["a", "b"]')

    assert build_proxy_flag_settings(str(settings_file), PROXY_URL) is None


def test_non_object_inline_settings_is_left_untouched(isolated_settings):
    assert build_proxy_flag_settings("{}[]", PROXY_URL) is None


def test_valid_empty_settings_file_still_gets_the_proxy(isolated_settings, tmp_path):
    """A genuinely empty settings object is readable — it must NOT be skipped."""
    settings_file = tmp_path / "empty.json"
    settings_file.write_text("{}")

    settings = build_proxy_flag_settings(str(settings_file), PROXY_URL)

    assert json.loads(settings)["env"]["ANTHROPIC_BASE_URL"] == PROXY_URL


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
