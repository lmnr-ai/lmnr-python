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

_CC_PROXY_LOCK = threading.Lock()
_CC_PROXY_PORT: int | None = None
_CC_PROXY_BASE_URL: str | None = None
_CC_PROXY_TARGET_URL: str | None = None
_CC_PROXY_SHUTDOWN_REGISTERED = False
_CC_PROXY_ENV_SNAPSHOT: dict[str, str | None] | None = None
_CC_PROXY_ENV_SNAPSHOT_SET: set[str] | None = None


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
    global _CC_PROXY_PORT
    global _CC_PROXY_BASE_URL
    global _CC_PROXY_TARGET_URL
    global _CC_PROXY_ENV_SNAPSHOT
    global _CC_PROXY_ENV_SNAPSHOT_SET

    try:
        stop_server()
    except Exception as e:
        logger.debug("Unable to stop cc-proxy: %s", e)

    # Restore environment variables to the pre-proxy state to avoid leaks.
    if _CC_PROXY_ENV_SNAPSHOT is not None and _CC_PROXY_ENV_SNAPSHOT_SET is not None:
        for key, value in _CC_PROXY_ENV_SNAPSHOT.items():
            if key in _CC_PROXY_ENV_SNAPSHOT_SET:
                os.environ[key] = value if value is not None else ""
            else:
                os.environ.pop(key, None)

    _CC_PROXY_PORT = None
    _CC_PROXY_BASE_URL = None
    _CC_PROXY_TARGET_URL = None
    _CC_PROXY_ENV_SNAPSHOT = None
    _CC_PROXY_ENV_SNAPSHOT_SET = None


def _stop_cc_proxy():
    with _CC_PROXY_LOCK:
        _stop_cc_proxy_locked()


def _register_proxy_shutdown():
    global _CC_PROXY_SHUTDOWN_REGISTERED
    if not _CC_PROXY_SHUTDOWN_REGISTERED:
        atexit.register(_stop_cc_proxy)
        _CC_PROXY_SHUTDOWN_REGISTERED = True


def get_cc_proxy_base_url() -> str | None:
    return _CC_PROXY_BASE_URL


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


def extract_3p_target_url() -> tuple[bool, Optional[str]]:
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


def start_or_connect_to_proxy() -> Optional[str]:
    """
    Start a new proxy or connect to an existing one.

    In serverless/lambda environments, containers may be reused and a proxy
    from a previous invocation might still be running. This function checks
    for an existing proxy first before starting a new one.
    """
    with _CC_PROXY_LOCK:
        global _CC_PROXY_PORT
        global _CC_PROXY_BASE_URL
        global _CC_PROXY_TARGET_URL
        global _CC_PROXY_ENV_SNAPSHOT
        global _CC_PROXY_ENV_SNAPSHOT_SET

        # Check if we already have a running proxy that we're tracking
        if _CC_PROXY_PORT is not None and _CC_PROXY_BASE_URL is not None:
            if _is_port_open(_CC_PROXY_PORT):
                logger.debug(
                    "Reusing existing tracked proxy on: %s", _CC_PROXY_BASE_URL
                )
                return _CC_PROXY_BASE_URL
            else:
                # Port is no longer open; reset state
                logger.debug("Previously tracked proxy is no longer running")
                _CC_PROXY_PORT = None
                _CC_PROXY_BASE_URL = None

        # Check if there's an existing proxy running from a previous container reuse
        existing_port = _find_existing_proxy()
        if existing_port is not None:
            proxy_base_url = f"http://127.0.0.1:{existing_port}"
            _CC_PROXY_PORT = existing_port
            _CC_PROXY_BASE_URL = proxy_base_url
            os.environ["ANTHROPIC_BASE_URL"] = proxy_base_url
            if _is_truthy_env(os.environ.get(FOUNDRY_USE_ENV)):
                os.environ[FOUNDRY_BASE_URL_ENV] = proxy_base_url
                if FOUNDRY_RESOURCE_ENV in os.environ:
                    os.environ.pop(FOUNDRY_RESOURCE_ENV, None)
            _register_proxy_shutdown()
            logger.info("Connected to existing proxy on: %s", proxy_base_url)
            return proxy_base_url

        # No existing proxy found; start a new one
        port = _find_available_port(DEFAULT_CC_PROXY_PORT, CC_PROXY_PORT_ATTEMPTS)
        if port is None:
            logger.warning("Unable to allocate port for cc-proxy.")
            return None

        # Resolve the upstream target. Third-party providers override Anthropic when enabled.
        provider_enabled, target_url = extract_3p_target_url()
        if provider_enabled and target_url is None:
            # Provider explicitly configured but invalid; abort.
            return None

        if target_url is None:
            # Fallback to the Anthropic base URL chain if no provider is configured.
            target_url = (
                _CC_PROXY_TARGET_URL
                or os.environ.get("ANTHROPIC_ORIGINAL_BASE_URL")
                or os.environ.get("ANTHROPIC_BASE_URL")
                or DEFAULT_ANTHROPIC_BASE_URL
            )
        _CC_PROXY_TARGET_URL = target_url
        if _CC_PROXY_ENV_SNAPSHOT is None:
            # Capture env before any mutations so we can restore on shutdown.
            _CC_PROXY_ENV_SNAPSHOT, _CC_PROXY_ENV_SNAPSHOT_SET = _snapshot_env(
                [
                    "ANTHROPIC_BASE_URL",
                    "ANTHROPIC_ORIGINAL_BASE_URL",
                    FOUNDRY_BASE_URL_ENV,
                    FOUNDRY_RESOURCE_ENV,
                ]
            )
        # Preserve the upstream URL for any consumers that expect it.
        os.environ.setdefault("ANTHROPIC_ORIGINAL_BASE_URL", target_url)

        try:
            run_server(target_url, port=port)
        except OSError as exc:  # pragma: no cover
            logger.warning("Unable to start cc-proxy: %s", exc)
            return None

        if not _wait_for_port(port):
            logger.warning("cc-proxy failed to start on port %s", port)
            stop_server()
            return None

        proxy_base_url = f"http://127.0.0.1:{port}"
        _CC_PROXY_PORT = port
        _CC_PROXY_BASE_URL = proxy_base_url
        os.environ["ANTHROPIC_BASE_URL"] = proxy_base_url
        if _is_truthy_env(os.environ.get(FOUNDRY_USE_ENV)):
            # Foundry uses a separate base URL; route it through the proxy too.
            os.environ[FOUNDRY_BASE_URL_ENV] = proxy_base_url
            if FOUNDRY_RESOURCE_ENV in os.environ:
                os.environ.pop(FOUNDRY_RESOURCE_ENV, None)
        _register_proxy_shutdown()

        logger.info("Started claude proxy server on: %s", proxy_base_url)
        return proxy_base_url


def release_proxy() -> None:
    with _CC_PROXY_LOCK:
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
