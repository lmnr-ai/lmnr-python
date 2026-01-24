"""Shared utilities for Claude Agent instrumentation."""

import os
import socket
import time
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)
# Constants
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
FOUNDRY_BASE_URL_ENV = "ANTHROPIC_FOUNDRY_BASE_URL"
FOUNDRY_RESOURCE_ENV = "ANTHROPIC_FOUNDRY_RESOURCE"
FOUNDRY_USE_ENV = "CLAUDE_CODE_USE_FOUNDRY"


def is_truthy_env(value: str | None) -> bool:
    """Check if environment variable value is truthy (equals '1')."""
    return value == "1"


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


def resolve_target_url_from_env(
    env_dict: dict[str, str], fallback: str = DEFAULT_ANTHROPIC_BASE_URL
) -> str | None:
    """
    Resolve target URL from environment dictionary with os.environ fallback.

    This is the single source of truth for determining the target URL for the proxy.

    Resolution order:
    1. Check if CLAUDE_CODE_USE_FOUNDRY is truthy
       - If yes, use ANTHROPIC_FOUNDRY_BASE_URL or construct from ANTHROPIC_FOUNDRY_RESOURCE
       - BASE_URL and RESOURCE are mutually exclusive (BASE_URL takes precedence)
    2. Otherwise, use ANTHROPIC_BASE_URL
    3. Fall back to default

    For each environment variable, checks env_dict first, then os.environ as fallback.

    Args:
        env_dict: Dictionary of environment variables (e.g., from options.env)
        fallback: Fallback URL if no other source found (default: DEFAULT_ANTHROPIC_BASE_URL)

    Returns:
        Resolved target URL, or None if provider is misconfigured
    """

    # Helper to get value from env_dict first, then os.environ
    def get_env_value(key: str) -> str | None:
        return env_dict.get(key) or os.environ.get(key)

    # Check if foundry is enabled
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

    # Not using foundry - check for ANTHROPIC_BASE_URL
    anthropic_base_url = get_env_value("ANTHROPIC_BASE_URL")
    if anthropic_base_url:
        return anthropic_base_url

    # Use fallback
    return fallback


def setup_proxy_env(proxy_url: str) -> dict[str, str | None]:
    """
    Configure global environment to use proxy for custom transports.

    This is only used for custom (non-SubprocessCLITransport) transports
    where we can't control environment variable passing. We set ANTHROPIC_ORIGINAL_BASE_URL
    so the proxy server knows where to forward requests.

    Args:
        proxy_url: Proxy base URL (e.g., "http://127.0.0.1:45667")

    Returns:
        Dictionary of original env values for restoration
    """
    snapshot: dict[str, str | None] = {
        "ANTHROPIC_BASE_URL": os.environ.get("ANTHROPIC_BASE_URL"),
        "ANTHROPIC_ORIGINAL_BASE_URL": os.environ.get("ANTHROPIC_ORIGINAL_BASE_URL"),
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

    # Handle Foundry-specific env vars
    if is_truthy_env(os.environ.get(FOUNDRY_USE_ENV)):
        snapshot[FOUNDRY_BASE_URL_ENV] = os.environ.get(FOUNDRY_BASE_URL_ENV)
        snapshot[FOUNDRY_RESOURCE_ENV] = os.environ.get(FOUNDRY_RESOURCE_ENV)

        os.environ[FOUNDRY_BASE_URL_ENV] = proxy_url
        os.environ.pop(FOUNDRY_RESOURCE_ENV, None)

    return snapshot
