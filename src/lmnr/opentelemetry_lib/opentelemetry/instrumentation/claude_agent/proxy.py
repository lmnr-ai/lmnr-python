from __future__ import annotations

import atexit
import os
import socket
import threading
import time
from typing import Optional

from lmnr.sdk.log import get_default_logger

from lmnr_claude_code_proxy import run_server, set_current_trace, stop_server

logger = get_default_logger(__name__)

DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_CC_PROXY_PORT = 45667
CC_PROXY_PORT_ATTEMPTS = 5

_CC_PROXY_LOCK = threading.Lock()
_CC_PROXY_PORT: int | None = None
_CC_PROXY_BASE_URL: str | None = None
_CC_PROXY_TARGET_URL: str | None = None
_CC_PROXY_SHUTDOWN_REGISTERED = False


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


def _stop_cc_proxy_locked():
    global _CC_PROXY_PORT, _CC_PROXY_BASE_URL

    try:
        stop_server()
    except Exception as e:
        logger.debug("Unable to stop cc-proxy: %s", e)

    if _CC_PROXY_TARGET_URL:
        os.environ["ANTHROPIC_BASE_URL"] = _CC_PROXY_TARGET_URL

    _CC_PROXY_PORT = None
    _CC_PROXY_BASE_URL = None


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


def start_proxy() -> Optional[str]:
    with _CC_PROXY_LOCK:
        global _CC_PROXY_PORT, _CC_PROXY_BASE_URL, _CC_PROXY_TARGET_URL

        port = _find_available_port(DEFAULT_CC_PROXY_PORT, CC_PROXY_PORT_ATTEMPTS)
        if port is None:
            logger.warning("Unable to allocate port for cc-proxy.")
            return None

        target_url = (
            _CC_PROXY_TARGET_URL
            or os.environ.get("ANTHROPIC_ORIGINAL_BASE_URL")
            or os.environ.get("ANTHROPIC_BASE_URL")
            or DEFAULT_ANTHROPIC_BASE_URL
        )
        _CC_PROXY_TARGET_URL = target_url
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
        _register_proxy_shutdown()

        logger.info("Started claude proxy server on: " + str(proxy_base_url))
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
