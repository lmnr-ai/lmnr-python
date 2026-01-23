from __future__ import annotations

import atexit
import os
import socket
import threading
import time
from typing import Optional

from lmnr_claude_code_proxy import run_server, set_current_trace, stop_server

from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_CC_PROXY_PORT = 45667
CC_PROXY_PORT_ATTEMPTS = 5

# Third-party API providers
# Microsoft Foundry
FOUNDRY_BASE_URL_ENV = "ANTHROPIC_FOUNDRY_BASE_URL"
FOUNDRY_RESOURCE_ENV = "ANTHROPIC_FOUNDRY_RESOURCE"
FOUNDRY_USE_ENV = "CLAUDE_CODE_USE_FOUNDRY"
# TODO: Amazon Bedrock.
# TODO: Google Vertex AI.


class _ProxyState:
    """Encapsulates proxy server state and environment snapshot."""

    def __init__(self):
        self.lock = threading.Lock()
        self.port: int | None = None
        self.base_url: str | None = None
        self.target_url: str | None = None
        self.shutdown_registered: bool = False
        self.env_snapshot: dict[str, str | None] | None = None
        self.env_snapshot_set: set[str] | None = None

    def reset(self):
        """Reset all state except lock and shutdown registration."""
        self.port = None
        self.base_url = None
        self.target_url = None
        self.env_snapshot = None
        self.env_snapshot_set = None

    def is_running(self) -> bool:
        """Check if we're tracking a running proxy."""
        return self.port is not None and self.base_url is not None


_proxy_state = _ProxyState()


def _find_available_port(start_port: int, attempts: int) -> Optional[int]:
    for offset in range(attempts):
        candidate = start_port + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", candidate))
            except OSError:
                continue
        return candidate
    return None


def _wait_for_port(port: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            try:
                sock.connect(("127.0.0.1", port))
                return True
            except OSError:
                time.sleep(0.1)
    return False


def _is_port_open(port: int) -> bool:
    """Check if a port is currently accepting connections (non-blocking)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        try:
            sock.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _stop_cc_proxy_locked():
    try:
        stop_server()
    except Exception as e:
        logger.debug("Unable to stop cc-proxy: %s", e)

    # Restore environment variables to the pre-proxy state to avoid leaks.
    if (
        _proxy_state.env_snapshot is not None
        and _proxy_state.env_snapshot_set is not None
    ):
        for key, value in _proxy_state.env_snapshot.items():
            if key in _proxy_state.env_snapshot_set:
                os.environ[key] = value if value is not None else ""
            else:
                os.environ.pop(key, None)

    _proxy_state.reset()


def _stop_cc_proxy():
    with _proxy_state.lock:
        _stop_cc_proxy_locked()


def _register_proxy_shutdown():
    if not _proxy_state.shutdown_registered:
        atexit.register(_stop_cc_proxy)
        _proxy_state.shutdown_registered = True


def get_cc_proxy_base_url() -> str | None:
    return _proxy_state.base_url


def _is_truthy_env(value: str | None) -> bool:
    return value == "1"


def _foundry_target_url() -> str | None:
    if not _is_truthy_env(os.environ.get(FOUNDRY_USE_ENV)):
        return None

    base_url = os.environ.get(FOUNDRY_BASE_URL_ENV)
    if base_url:
        return base_url.rstrip("/")

    resource = os.environ.get(FOUNDRY_RESOURCE_ENV)
    if resource:
        return f"https://{resource}.services.ai.azure.com/anthropic"

    logger.error(
        "%s is set but neither %s nor %s is configured. "
        "Microsoft Foundry requires one of these values.",
        FOUNDRY_USE_ENV,
        FOUNDRY_BASE_URL_ENV,
        FOUNDRY_RESOURCE_ENV,
    )
    return None


def _snapshot_env(keys: list[str]) -> tuple[dict[str, str | None], set[str]]:
    # Track value + presence so we can restore or delete keys accurately.
    snapshot: dict[str, str | None] = {}
    set_keys: set[str] = set()
    for key in keys:
        if key in os.environ:
            set_keys.add(key)
            snapshot[key] = os.environ.get(key)
        else:
            snapshot[key] = None
    return snapshot, set_keys


def _extract_3p_target_url() -> tuple[bool, Optional[str]]:
    """
    Extract the target URL for third-party Anthropic API providers.

    Returns a tuple of (provider_enabled, target_url):
    - (False, None): No third-party provider is configured
    - (True, url): A provider is configured and valid
    - (True, None): A provider is configured but invalid (should abort)
    """
    # Microsoft Foundry
    if _is_truthy_env(os.environ.get(FOUNDRY_USE_ENV)):
        target_url = _foundry_target_url()
        return (True, target_url)

    # TODO: Amazon Bedrock support
    # TODO: Google Vertex AI support

    return (False, None)


def _find_existing_proxy() -> Optional[int]:
    """
    Check if a proxy is already running on the default port range.
    Returns the port if found, None otherwise.
    """
    for offset in range(CC_PROXY_PORT_ATTEMPTS):
        port = DEFAULT_CC_PROXY_PORT + offset
        if _is_port_open(port):
            return port
    return None


def _capture_env_snapshot_if_needed():
    """Capture environment snapshot if not already done."""
    if _proxy_state.env_snapshot is None:
        _proxy_state.env_snapshot, _proxy_state.env_snapshot_set = _snapshot_env(
            [
                "ANTHROPIC_BASE_URL",
                "ANTHROPIC_ORIGINAL_BASE_URL",
                FOUNDRY_BASE_URL_ENV,
                FOUNDRY_RESOURCE_ENV,
            ]
        )


def _setup_proxy_env(proxy_base_url: str):
    """Configure environment variables to route through the proxy."""
    os.environ["ANTHROPIC_BASE_URL"] = proxy_base_url
    if _is_truthy_env(os.environ.get(FOUNDRY_USE_ENV)):
        os.environ[FOUNDRY_BASE_URL_ENV] = proxy_base_url
        if FOUNDRY_RESOURCE_ENV in os.environ:
            os.environ.pop(FOUNDRY_RESOURCE_ENV, None)


def _resolve_target_url() -> Optional[str]:
    """
    Resolve the target URL for the proxy.
    Returns None if a provider is explicitly configured but invalid.
    """
    # Check for third-party providers first
    provider_enabled, target_url = _extract_3p_target_url()
    if provider_enabled:
        return target_url  # Can be None if provider is misconfigured

    # Fallback to Anthropic base URL chain
    if _proxy_state.target_url:
        return _proxy_state.target_url

    if _proxy_state.env_snapshot:
        # Use snapshot to avoid circular reference to proxy URL
        return (
            _proxy_state.env_snapshot.get("ANTHROPIC_ORIGINAL_BASE_URL")
            or _proxy_state.env_snapshot.get("ANTHROPIC_BASE_URL")
            or DEFAULT_ANTHROPIC_BASE_URL
        )

    return (
        os.environ.get("ANTHROPIC_ORIGINAL_BASE_URL")
        or os.environ.get("ANTHROPIC_BASE_URL")
        or DEFAULT_ANTHROPIC_BASE_URL
    )


def _connect_to_tracked_proxy() -> Optional[str]:
    """
    Check if we already have a running proxy that we're tracking.
    Returns the proxy URL if valid, None otherwise.
    """
    if not _proxy_state.is_running():
        return None

    if _is_port_open(_proxy_state.port):
        logger.debug("Reusing existing tracked proxy on: %s", _proxy_state.base_url)
        return _proxy_state.base_url

    # Port is no longer open; reset state
    logger.debug("Previously tracked proxy is no longer running")
    _proxy_state.reset()
    return None


def _connect_to_existing_proxy(port: int) -> Optional[str]:
    """
    Connect to an existing untracked proxy (e.g., from container reuse).
    Returns the proxy URL.
    """
    # Capture environment snapshot
    _capture_env_snapshot_if_needed()

    # Resolve and preserve the upstream target URL
    if _proxy_state.target_url is None:
        target_url = _resolve_target_url()
        if target_url is None:
            # Provider is misconfigured
            return None
        _proxy_state.target_url = target_url
        os.environ.setdefault("ANTHROPIC_ORIGINAL_BASE_URL", target_url)

    # Only set state after successful validation
    proxy_base_url = f"http://127.0.0.1:{port}"
    _proxy_state.port = port
    _proxy_state.base_url = proxy_base_url

    # Configure environment to use proxy
    _setup_proxy_env(proxy_base_url)
    _register_proxy_shutdown()

    logger.info("Connected to existing proxy on: %s", proxy_base_url)
    return proxy_base_url


def _start_new_proxy() -> Optional[str]:
    """
    Start a new proxy server.
    Returns the proxy URL on success, None on failure.
    """
    # Find available port
    port = _find_available_port(DEFAULT_CC_PROXY_PORT, CC_PROXY_PORT_ATTEMPTS)
    if port is None:
        logger.warning("Unable to allocate port for cc-proxy.")
        return None

    # Resolve target URL
    target_url = _resolve_target_url()
    if target_url is None:
        # Provider explicitly configured but invalid
        return None

    original_target_url = _proxy_state.target_url
    _proxy_state.target_url = target_url

    # Capture environment snapshot
    _capture_env_snapshot_if_needed()

    # Preserve the upstream URL
    os.environ.setdefault("ANTHROPIC_ORIGINAL_BASE_URL", target_url)

    # Start the server
    try:
        run_server(target_url, port=port)
    except OSError as exc:  # pragma: no cover
        logger.warning("Unable to start cc-proxy: %s", exc)
        _proxy_state.target_url = original_target_url
        return None

    # Wait for server to be ready
    if not _wait_for_port(port):
        logger.warning("cc-proxy failed to start on port %s", port)
        stop_server()
        _proxy_state.target_url = original_target_url
        return None

    # Update state
    proxy_base_url = f"http://127.0.0.1:{port}"
    _proxy_state.port = port
    _proxy_state.base_url = proxy_base_url

    # Configure environment to use proxy
    _setup_proxy_env(proxy_base_url)
    _register_proxy_shutdown()

    logger.info("Started claude proxy server on: %s", proxy_base_url)
    return proxy_base_url


def start_or_connect_to_proxy() -> Optional[str]:
    """
    Start a new proxy or connect to an existing one.

    In serverless/lambda environments, containers may be reused and a proxy
    from a previous invocation might still be running. This function checks
    for an existing proxy first before starting a new one.
    """
    with _proxy_state.lock:
        # Try to reuse tracked proxy first
        if url := _connect_to_tracked_proxy():
            return url

        # Try to connect to existing untracked proxy (container reuse)
        if port := _find_existing_proxy():
            return _connect_to_existing_proxy(port)

        # Start new proxy
        return _start_new_proxy()


def release_proxy() -> None:
    with _proxy_state.lock:
        _stop_cc_proxy_locked()
        logger.debug("Released claude proxy server")


def set_trace_to_proxy(
    trace_id: str,
    span_id: str,
    project_api_key: str,
    span_path: list[str] = [],
    span_ids_path: list[str] = [],
    laminar_url: str = "https://api.lmnr.ai",
):
    set_current_trace(
        trace_id=trace_id,
        span_id=span_id,
        project_api_key=project_api_key,
        span_path=span_path,
        span_ids_path=span_ids_path,
        laminar_url=laminar_url,
    )
