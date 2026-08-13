import os
import threading

import pytest

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.claude_agent.utils import (
    PROXY_ENV_KEYS,
)

# ProxyServer names its health-check thread f"proxy-monitor-{port}".
_MONITOR_PREFIX = "proxy-monitor-"


def _force_delenv(monkeypatch, key: str) -> None:
    """Delete ``key`` for the test and ensure undo deletes it if it was absent.

    ``monkeypatch.delenv(raising=False)`` is a no-op when the key is missing, so a
    test that later sets it (e.g. via ``setup_proxy_env``) would otherwise leak
    into the next test. ``setenv("")`` then ``delenv`` registers a proper undo.
    """
    if key in os.environ:
        monkeypatch.delenv(key)
    else:
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)


@pytest.fixture(autouse=True)
def isolate_claude_settings(tmp_path_factory, monkeypatch):
    """Keep every test away from the developer's real Claude settings.

    ``resolve_target_url_from_env`` / ``setup_proxy_env`` / ``read_claude_settings_env``
    all fall back to ``~/.claude/settings.json`` when ``CLAUDE_CONFIG_DIR`` is unset.
    Maintainers commonly keep ``CLAUDE_CODE_USE_BEDROCK`` (and AWS credentials) there,
    which silently flips upstream resolution to Bedrock and breaks tests that expect
    the Anthropic default — including ones that pass in isolation.

    Redirect the user config dir to an empty temp path. Tests that need a populated
    settings file set their own ``CLAUDE_CONFIG_DIR`` via ``isolated_settings``.
    """
    config_dir = tmp_path_factory.mktemp("claude-config")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

    # Also clear process-env knobs the resolver consults before / alongside
    # settings. Sandboxes and developer shells often export these.
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
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_REGION",
        "AWS_PROFILE",
        *PROXY_ENV_KEYS,
    ):
        _force_delenv(monkeypatch, key)


@pytest.fixture(autouse=True)
def cleanup_claude_proxy():
    """Clean up Claude proxy server before and after each test."""
    # Clean up before test
    _cleanup_proxy()
    yield
    # Clean up after test
    _cleanup_proxy()


@pytest.fixture(autouse=True)
def no_leaked_proxy_monitor_threads():
    """Fail the test that leaves a live proxy health-monitor thread behind.

    ``ProxyServer.run_server`` spawns a daemon ``proxy-monitor-<port>`` thread that
    GETs ``/lmnr-internal/health`` every second forever. ``_cleanup_proxy`` only stops
    the module-level singleton, so a per-transport instance started by mistake — e.g.
    by patching ``proxy.create_proxy_for_transport`` instead of the name
    ``wrappers`` actually calls — survives this whole session. Those stray requests
    then land inside unrelated tests that spy on ``httpx.Client.send`` globally
    (``test_openai/traces/test_chat.py`` and friends), which fail roughly whenever a
    1s tick hits their few-millisecond window. Catch the leak at its source instead.
    """
    before = {t.name for t in threading.enumerate() if _MONITOR_PREFIX in t.name}
    yield
    leaked = sorted(
        t.name
        for t in threading.enumerate()
        if _MONITOR_PREFIX in t.name and t.name not in before
    )
    assert not leaked, (
        f"test leaked live proxy health-monitor thread(s): {leaked}. A real proxy "
        "server was started and never stopped — patch "
        "`wrappers.create_proxy_for_transport` / `wrappers.start_proxy` (where the "
        "names are looked up), or stop the proxy you started."
    )


def start_claude_proxy():
    """Start the Claude proxy server if it's not running."""
    try:
        from lmnr_claude_code_proxy import run_server

        run_server()
    except Exception:
        # Ignore errors if the proxy couldn't be started
        pass


def _cleanup_proxy():
    """Stop the Claude proxy server if it's running."""
    try:
        from lmnr_claude_code_proxy import stop_server

        stop_server()
    except Exception:
        # Ignore errors if the proxy wasn't running or module not available
        pass
