import os

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.claude_agent import proxy as claude_proxy


def _patch_proxy_helpers(monkeypatch, port=45667):
    calls = {}

    def fake_run_server(target_url, port):
        calls["target_url"] = target_url
        calls["port"] = port

    monkeypatch.setattr(claude_proxy, "run_server", fake_run_server)
    monkeypatch.setattr(claude_proxy, "stop_server", lambda: None)
    monkeypatch.setattr(
        claude_proxy, "_find_available_port", lambda start_port, attempts: port
    )
    monkeypatch.setattr(claude_proxy, "_wait_for_port", lambda port: True)
    return calls, f"http://127.0.0.1:{port}"


def test_foundry_base_url_overrides_target(monkeypatch):
    calls, proxy_url = _patch_proxy_helpers(monkeypatch)
    monkeypatch.setenv("CLAUDE_CODE_USE_FOUNDRY", "1")
    monkeypatch.setenv(
        "ANTHROPIC_FOUNDRY_BASE_URL", "https://foundry.example/anthropic/"
    )
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_RESOURCE", raising=False)

    result = claude_proxy.start_or_connect_to_proxy()

    assert result == proxy_url
    assert calls["target_url"] == "https://foundry.example/anthropic"
    assert os.environ["ANTHROPIC_FOUNDRY_BASE_URL"] == proxy_url

    claude_proxy.release_proxy()

    assert os.environ["ANTHROPIC_FOUNDRY_BASE_URL"] == "https://foundry.example/anthropic/"


def test_foundry_resource_builds_target_url(monkeypatch):
    calls, proxy_url = _patch_proxy_helpers(monkeypatch)
    monkeypatch.setenv("CLAUDE_CODE_USE_FOUNDRY", "1")
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_BASE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_RESOURCE", "my-resource")

    result = claude_proxy.start_or_connect_to_proxy()

    assert result == proxy_url
    assert (
        calls["target_url"]
        == "https://my-resource.services.ai.azure.com/anthropic"
    )
    assert os.environ["ANTHROPIC_FOUNDRY_BASE_URL"] == proxy_url

    claude_proxy.release_proxy()

    assert "ANTHROPIC_FOUNDRY_BASE_URL" not in os.environ


def test_foundry_missing_config_fails(monkeypatch):
    calls, _ = _patch_proxy_helpers(monkeypatch)
    monkeypatch.setenv("CLAUDE_CODE_USE_FOUNDRY", "1")
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_RESOURCE", raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

    result = claude_proxy.start_or_connect_to_proxy()

    assert result is None
    assert "target_url" not in calls
    assert os.environ["ANTHROPIC_BASE_URL"] == "https://api.anthropic.com"
    assert "ANTHROPIC_FOUNDRY_BASE_URL" not in os.environ
