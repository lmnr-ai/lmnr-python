"""Shared utilities for Claude Agent instrumentation."""

import json
import os
import re
import socket
import time
from pathlib import Path
from typing import Any

from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)
# Constants
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
FOUNDRY_BASE_URL_ENV = "ANTHROPIC_FOUNDRY_BASE_URL"
FOUNDRY_RESOURCE_ENV = "ANTHROPIC_FOUNDRY_RESOURCE"
FOUNDRY_USE_ENV = "CLAUDE_CODE_USE_FOUNDRY"
BEDROCK_BASE_URL_ENV = "ANTHROPIC_BEDROCK_BASE_URL"
BEDROCK_USE_ENV = "CLAUDE_CODE_USE_BEDROCK"
BEDROCK_AWS_REGION_ENV = "AWS_REGION"

# Vertex AI configuration constants
VERTEX_BASE_URL_ENV = "ANTHROPIC_VERTEX_BASE_URL"
VERTEX_USE_ENV = "CLAUDE_CODE_USE_VERTEX"

# Base-URL keys that must point at our proxy in the flag-settings layer.
PROXY_BASE_URL_ENV_KEYS = (
    "ANTHROPIC_BASE_URL",
    FOUNDRY_BASE_URL_ENV,
    BEDROCK_BASE_URL_ENV,
    VERTEX_BASE_URL_ENV,
)

# Keys that must be blanked in the flag-settings layer: they would otherwise
# redirect the CLI away from our proxy. Removing them from the flag layer is not
# enough — settings layers merge per key, so a lower layer's value would win.
PROXY_NEUTRALIZED_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    FOUNDRY_RESOURCE_ENV,
)


def is_truthy_env(value: str | None) -> bool:
    """Check if environment variable value is truthy (equals '1')."""
    return value == "1"


def _read_settings_file(path: Path) -> dict[str, Any]:
    """Load a Claude settings JSON file; empty dict when missing or invalid."""
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _settings_env_block(data: dict[str, Any]) -> dict[str, str]:
    """Normalize a settings object's ``env`` block to a str -> str mapping."""
    env = data.get("env")
    if not isinstance(env, dict):
        return {}
    return {str(k): str(v) for k, v in env.items() if v is not None}


def read_claude_settings_env(cwd: str | Path | None = None) -> dict[str, str]:
    """
    Read the merged ``env`` block from Claude Code's on-disk settings layers.

    Claude Code applies these to the CLI session with HIGHER priority than the
    subprocess environment, so a user with ``ANTHROPIC_BASE_URL`` in
    ``~/.claude/settings.json`` silently bypasses our proxy. We read them to
    resolve the real upstream and to detect conflicts.

    Precedence (highest first): local project, shared project, user.
    """
    session_cwd = Path(cwd).expanduser() if cwd is not None else Path.cwd()
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    user_dir = Path(config_dir).expanduser() if config_dir else Path.home() / ".claude"

    merged: dict[str, str] = {}
    for path in (
        user_dir / "settings.json",
        session_cwd / ".claude" / "settings.json",
        session_cwd / ".claude" / "settings.local.json",
    ):
        merged.update(_settings_env_block(_read_settings_file(path)))
    return merged


def snapshot_env(keys: list[str]) -> tuple[dict[str, str | None], set[str]]:
    """
    Snapshot environment variables.

    Returns:
        Tuple of (snapshot dict, set of keys that were present)
    """
    snapshot: dict[str, str | None] = {}
    set_keys: set[str] = set()
    for key in keys:
        if key in os.environ:
            set_keys.add(key)
            snapshot[key] = os.environ.get(key)
        else:
            snapshot[key] = None
    return snapshot, set_keys


def restore_env(snapshot: dict[str, str | None], set_keys: set[str]) -> None:
    """
    Restore environment variables from snapshot.

    Args:
        snapshot: Dictionary of variable names to values
        set_keys: Set of keys that were originally present
    """
    for key, value in snapshot.items():
        if key in set_keys:
            os.environ[key] = value if value is not None else ""
        else:
            os.environ.pop(key, None)


def is_port_open(port: int, timeout: float = 0.5) -> bool:
    """
    Check if a port is currently accepting connections.

    Args:
        port: Port number to check
        timeout: Connection timeout in seconds

    Returns:
        True if port is open, False otherwise
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        try:
            sock.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


def wait_for_port(port: int, timeout: float = 5.0) -> bool:
    """
    Wait for a port to become available (accepting connections).

    Args:
        port: Port number to wait for
        timeout: Maximum time to wait in seconds

    Returns:
        True if port became available, False if timeout
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_port_open(port, timeout=0.2):
            return True
        time.sleep(0.1)
    return False


def _get_region_from_aws_config(profile: str) -> str | None:
    """Read region for a given profile from ~/.aws/config."""
    config_path = os.path.expanduser("~/.aws/config")
    try:
        with open(config_path, "r") as f:
            content = f.read()
    except OSError:
        return None

    profile_header = "default" if profile == "default" else f"profile {profile}"
    match = re.search(
        rf"\[{re.escape(profile_header)}\][^\[]*?^\s*region\s*=\s*([^\s\n]+)",
        content,
        re.MULTILINE | re.DOTALL,
    )
    return match.group(1) if match else None


def resolve_target_url_from_env(
    env_dict: dict[str, str],
    fallback: str = DEFAULT_ANTHROPIC_BASE_URL,
    *,
    cwd: str | Path | None = None,
) -> str | None:
    """
    Resolve target URL from environment dictionary with os.environ fallback.

    This is the single source of truth for determining the target URL for the proxy.

    Resolution order (highest to lowest priority):
    1. HTTPS_PROXY - if set, use as target (our proxy will forward to it)
    2. HTTP_PROXY - if set, use as target (our proxy will forward to it)
    3. Third-party provider URLs (e.g., Foundry, Bedrock, Vertex):
       - If CLAUDE_CODE_USE_FOUNDRY is truthy:
         - Use ANTHROPIC_FOUNDRY_BASE_URL, or
         - Construct from ANTHROPIC_FOUNDRY_RESOURCE
       - If CLAUDE_CODE_USE_BEDROCK is truthy:
         - Use ANTHROPIC_BEDROCK_BASE_URL, or
         - Construct from AWS_REGION env var, or
         - Construct by reading region from ~/.aws/config via AWS_PROFILE
       - If CLAUDE_CODE_USE_VERTEX is truthy:
         - Use ANTHROPIC_VERTEX_BASE_URL, or
         - Fall back to https://aiplatform.googleapis.com/v1
    4. ANTHROPIC_BASE_URL - standard Anthropic API base URL
    5. Fall back to default (https://api.anthropic.com)

    For each environment variable, checks env_dict first, then os.environ, then
    Claude Code's on-disk settings ``env`` block. The settings layer is checked
    last as a source for the *upstream* URL, but note it wins over process env
    inside the CLI itself — see ``build_proxy_flag_settings``.

    Args:
        env_dict: Dictionary of environment variables (e.g., from options.env)
        fallback: Fallback URL if no other source found (default: DEFAULT_ANTHROPIC_BASE_URL)
        cwd: Session root used to locate project settings (``options.cwd``)

    Returns:
        Resolved target URL, or None if provider is misconfigured
    """
    settings_env = read_claude_settings_env(cwd)

    # Helper: options.env, then os.environ, then Claude settings env
    def get_env_value(key: str) -> str | None:
        return env_dict.get(key) or os.environ.get(key) or settings_env.get(key)

    # 1. Check for HTTPS_PROXY (highest priority)
    https_proxy = get_env_value("HTTPS_PROXY")
    if https_proxy:
        return https_proxy.rstrip("/")

    # 2. Check for HTTP_PROXY
    http_proxy = get_env_value("HTTP_PROXY")
    if http_proxy:
        return http_proxy.rstrip("/")

    # 3. Check for third-party providers (Foundry)
    foundry_enabled = is_truthy_env(get_env_value(FOUNDRY_USE_ENV))
    if foundry_enabled:
        # Try to get Foundry base URL first
        foundry_base_url = get_env_value(FOUNDRY_BASE_URL_ENV)
        if foundry_base_url:
            return foundry_base_url.rstrip("/")

        # Try to construct from resource
        foundry_resource = get_env_value(FOUNDRY_RESOURCE_ENV)
        if foundry_resource:
            return f"https://{foundry_resource}.services.ai.azure.com/anthropic"

        # Foundry is enabled but misconfigured
        logger.error(
            "%s is set but neither %s nor %s is configured. "
            "Microsoft Foundry requires one of these values.",
            FOUNDRY_USE_ENV,
            FOUNDRY_BASE_URL_ENV,
            FOUNDRY_RESOURCE_ENV,
        )
        return None

    # 3b. Check for Bedrock
    bedrock_enabled = is_truthy_env(get_env_value(BEDROCK_USE_ENV))
    if bedrock_enabled:
        bedrock_base_url = get_env_value(BEDROCK_BASE_URL_ENV)
        if bedrock_base_url:
            return bedrock_base_url.rstrip("/")

        region = get_env_value(BEDROCK_AWS_REGION_ENV)
        if not region:
            aws_profile = get_env_value("AWS_PROFILE") or "default"
            region = _get_region_from_aws_config(aws_profile)

        if region:
            return f"https://bedrock-runtime.{region}.amazonaws.com"

        logger.error(
            "%s is set but could not determine AWS region. "
            "Set %s or configure a region in ~/.aws/config for the active profile.",
            BEDROCK_USE_ENV,
            BEDROCK_AWS_REGION_ENV,
        )
        return None

    # 3c. Check for Vertex AI
    vertex_enabled = is_truthy_env(get_env_value(VERTEX_USE_ENV))
    if vertex_enabled:
        # Unlike Foundry or Bedrock, we don't parse project or region config because
        # they affect the URL path, not the base URL, so CC handles this internally
        vertex_base_url = get_env_value(VERTEX_BASE_URL_ENV)
        if vertex_base_url:
            return vertex_base_url.rstrip("/")
        return "https://aiplatform.googleapis.com/v1"

    # 4. Check for ANTHROPIC_BASE_URL
    anthropic_base_url = get_env_value("ANTHROPIC_BASE_URL")
    if anthropic_base_url:
        return anthropic_base_url.rstrip("/")

    # 5. Use fallback
    return fallback


def build_proxy_flag_settings(
    existing: str | None,
    proxy_url: str,
    *,
    cwd: str | Path | None = None,
) -> str | None:
    """
    Build the ``--settings`` value that forces the CLI through our proxy.

    Claude Code resolves ``env`` from its settings layers with higher priority
    than the subprocess environment, so rewriting ``options.env`` alone leaves a
    user with ``ANTHROPIC_BASE_URL`` in ``~/.claude/settings.json`` talking
    straight to their upstream and the proxy sees zero traffic (lmnr#2167).
    ``--settings`` is the highest user-controlled layer, so writing the proxy URL
    there wins without ever touching the user's files on disk.

    Layers merge per key, so keys we simply omit keep their lower-layer value.
    Redirecting keys are therefore blanked rather than dropped.

    Returns the JSON string to assign to ``options.settings``, or ``None`` when
    the caller's existing value is a file path we could not read (in that case
    the path must be left alone so the CLI can still resolve it itself).
    """
    settings_obj: dict[str, Any] = {}
    if existing:
        stripped = existing.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
            except ValueError:
                return None
            if isinstance(parsed, dict):
                settings_obj = parsed
        else:
            path = Path(stripped).expanduser()
            if not path.is_absolute() and cwd is not None:
                path = Path(cwd).expanduser() / path
            if not path.is_file():
                return None
            settings_obj = _read_settings_file(path)

    env_block = settings_obj.get("env")
    env_dict: dict[str, str] = dict(env_block) if isinstance(env_block, dict) else {}

    settings_env = read_claude_settings_env(cwd)
    for key in PROXY_BASE_URL_ENV_KEYS:
        # Only pin the keys that are actually in play, so we never introduce a
        # provider base URL the user never configured.
        if key == "ANTHROPIC_BASE_URL" or key in settings_env or key in env_dict:
            env_dict[key] = proxy_url
    for key in PROXY_NEUTRALIZED_ENV_KEYS:
        if key in settings_env or key in env_dict:
            env_dict[key] = ""

    settings_obj["env"] = env_dict
    return json.dumps(settings_obj)


def setup_proxy_env(proxy_url: str) -> dict[str, str | None]:
    """
    Configure global environment to use proxy for custom transports.

    This is only used for custom (non-SubprocessCLITransport) transports
    where we can't control environment variable passing. We set ANTHROPIC_ORIGINAL_BASE_URL
    so the proxy server knows where to forward requests.

    Also removes HTTP_PROXY and HTTPS_PROXY from global env
    since our proxy will handle forwarding to them.

    Args:
        proxy_url: Proxy base URL (e.g., "http://127.0.0.1:45667")

    Returns:
        Dictionary of original env values for restoration
    """
    snapshot: dict[str, str | None] = {
        "ANTHROPIC_BASE_URL": os.environ.get("ANTHROPIC_BASE_URL"),
        "ANTHROPIC_ORIGINAL_BASE_URL": os.environ.get("ANTHROPIC_ORIGINAL_BASE_URL"),
        "HTTP_PROXY": os.environ.get("HTTP_PROXY"),
        "HTTPS_PROXY": os.environ.get("HTTPS_PROXY"),
    }

    # Store original target URL in ANTHROPIC_ORIGINAL_BASE_URL if not already set
    # This is used by the proxy to know where to forward requests
    if "ANTHROPIC_ORIGINAL_BASE_URL" not in os.environ:
        target = resolve_target_url_from_env({})  # Check only os.environ
        if target:
            os.environ["ANTHROPIC_ORIGINAL_BASE_URL"] = target
            snapshot["ANTHROPIC_ORIGINAL_BASE_URL"] = None  # Was not set

    # Set proxy URL
    os.environ["ANTHROPIC_BASE_URL"] = proxy_url

    # Remove HTTP_PROXY and HTTPS_PROXY (our proxy will forward to them)
    for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY"]:
        os.environ.pop(proxy_var, None)

    # Handle Foundry-specific env vars
    if is_truthy_env(os.environ.get(FOUNDRY_USE_ENV)):
        snapshot[FOUNDRY_BASE_URL_ENV] = os.environ.get(FOUNDRY_BASE_URL_ENV)
        snapshot[FOUNDRY_RESOURCE_ENV] = os.environ.get(FOUNDRY_RESOURCE_ENV)

        os.environ[FOUNDRY_BASE_URL_ENV] = proxy_url
        os.environ.pop(FOUNDRY_RESOURCE_ENV, None)

    # Handle Bedrock-specific env vars
    if is_truthy_env(os.environ.get(BEDROCK_USE_ENV)):
        snapshot[BEDROCK_BASE_URL_ENV] = os.environ.get(BEDROCK_BASE_URL_ENV)
        os.environ[BEDROCK_BASE_URL_ENV] = proxy_url

    # Handle Vertex AI-specific env vars
    if is_truthy_env(os.environ.get(VERTEX_USE_ENV)):
        snapshot[VERTEX_BASE_URL_ENV] = os.environ.get(VERTEX_BASE_URL_ENV)
        os.environ[VERTEX_BASE_URL_ENV] = proxy_url

    return snapshot
