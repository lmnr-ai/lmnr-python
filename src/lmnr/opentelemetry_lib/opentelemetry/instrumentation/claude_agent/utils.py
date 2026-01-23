"""Shared utilities for Claude Agent instrumentation."""

import os
import socket
import time
from typing import Optional

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


def foundry_target_url() -> str | None:
    """
    Get Microsoft Foundry target URL if configured.

    Returns:
        Foundry URL or None if not configured/invalid
    """
    if not is_truthy_env(os.environ.get(FOUNDRY_USE_ENV)):
        return None

    base_url = os.environ.get(FOUNDRY_BASE_URL_ENV)
    if base_url:
        return base_url.rstrip("/")

    resource = os.environ.get(FOUNDRY_RESOURCE_ENV)
    if resource:
        return f"https://{resource}.services.ai.azure.com/anthropic"

    # Log error but return None (handled by caller)
    from lmnr.sdk.log import get_default_logger

    logger = get_default_logger(__name__)
    logger.error(
        "%s is set but neither %s nor %s is configured. "
        "Microsoft Foundry requires one of these values.",
        FOUNDRY_USE_ENV,
        FOUNDRY_BASE_URL_ENV,
        FOUNDRY_RESOURCE_ENV,
    )
    return None


def extract_3p_target_url() -> tuple[bool, Optional[str]]:
    """
    Extract target URL for third-party Anthropic API providers.

    Returns:
        Tuple of (provider_enabled, target_url):
        - (False, None): No third-party provider configured
        - (True, url): Provider configured and valid
        - (True, None): Provider configured but invalid
    """
    # Microsoft Foundry
    if is_truthy_env(os.environ.get(FOUNDRY_USE_ENV)):
        target_url = foundry_target_url()
        return (True, target_url)

    # TODO: Amazon Bedrock support
    # TODO: Google Vertex AI support

    return (False, None)


def resolve_target_url(fallback: str = DEFAULT_ANTHROPIC_BASE_URL) -> str | None:
    """
    Resolve the target URL for the proxy, checking providers and environment.

    Args:
        fallback: Fallback URL if no other source found

    Returns:
        Resolved target URL, or None if provider is misconfigured
    """
    # Check for third-party providers first
    provider_enabled, target_url = extract_3p_target_url()
    if provider_enabled:
        return target_url  # Can be None if misconfigured

    # Fallback to environment or default
    return (
        os.environ.get("ANTHROPIC_ORIGINAL_BASE_URL")
        or os.environ.get("ANTHROPIC_BASE_URL")
        or fallback
    )


def setup_proxy_env(proxy_url: str) -> dict[str, str | None]:
    """
    Configure environment to use proxy, returning snapshot for restoration.

    Args:
        proxy_url: Proxy base URL (e.g., "http://127.0.0.1:45667")

    Returns:
        Dictionary of original env values for restoration
    """
    snapshot: dict[str, str | None] = {
        "ANTHROPIC_BASE_URL": os.environ.get("ANTHROPIC_BASE_URL"),
        "ANTHROPIC_ORIGINAL_BASE_URL": os.environ.get("ANTHROPIC_ORIGINAL_BASE_URL"),
    }

    # Determine target URL before proxying
    if "ANTHROPIC_ORIGINAL_BASE_URL" not in os.environ:
        target = resolve_target_url()
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
