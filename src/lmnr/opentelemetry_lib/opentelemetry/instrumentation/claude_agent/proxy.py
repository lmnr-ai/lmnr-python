from __future__ import annotations

import atexit
import logging
import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_CC_PROXY_PORT = 45667
CC_PROXY_PORT_ATTEMPTS = 5

_CC_PROXY_LOCK = threading.Lock()
_CC_PROXY_PROCESS: subprocess.Popen | None = None
_CC_PROXY_PORT: int | None = None
_CC_PROXY_BASE_URL: str | None = None
_CC_PROXY_TARGET_URL: str | None = None
_CC_PROXY_SHUTDOWN_REGISTERED = False
_CC_PROXY_USAGE_COUNT = 0


def _resolve_cc_proxy_binary() -> Optional[str]:
    bundled_binary = Path(__file__).with_name("cc-proxy")
    if bundled_binary.exists() and os.access(bundled_binary, os.X_OK):
        return str(bundled_binary)

    which_result = shutil.which("cc-proxy")
    if which_result:
        return which_result
    return None


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
    global _CC_PROXY_PROCESS, _CC_PROXY_PORT, _CC_PROXY_BASE_URL

    if _CC_PROXY_PROCESS:
        if _CC_PROXY_PROCESS.poll() is None:
            try:
                _CC_PROXY_PROCESS.terminate()
                _CC_PROXY_PROCESS.wait(timeout=2)
            except subprocess.TimeoutExpired:
                _CC_PROXY_PROCESS.kill()
        _CC_PROXY_PROCESS = None

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
    return _CC_PROXY_BASE_URL if _CC_PROXY_PROCESS and _CC_PROXY_PROCESS.poll() is None else None
    
async def start_proxy() -> Optional[str]:
    binary_path = _resolve_cc_proxy_binary()
    if not binary_path:
        logger.debug("cc-proxy binary not found. Skipping proxy startup.")
        return None

    with _CC_PROXY_LOCK:
        global _CC_PROXY_PROCESS, _CC_PROXY_PORT, _CC_PROXY_BASE_URL, _CC_PROXY_TARGET_URL

        global _CC_PROXY_USAGE_COUNT

        if _CC_PROXY_PROCESS and _CC_PROXY_PROCESS.poll() is None and _CC_PROXY_BASE_URL:
            _CC_PROXY_USAGE_COUNT += 1
            return _CC_PROXY_BASE_URL

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
            process = subprocess.Popen(
                [binary_path, "--target-url", target_url, "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:  # pragma: no cover
            logger.warning("Unable to start cc-proxy: %s", exc)
            return None

        if not _wait_for_port(port):
            logger.warning("cc-proxy failed to start on port %s", port)
            process.terminate()
            return None

        proxy_base_url = f"http://127.0.0.1:{port}"
        _CC_PROXY_PROCESS = process
        _CC_PROXY_PORT = port
        _CC_PROXY_BASE_URL = proxy_base_url
        os.environ["ANTHROPIC_BASE_URL"] = proxy_base_url
        _register_proxy_shutdown()
        _CC_PROXY_USAGE_COUNT = 1

        logger.info("Started claude proxy server on: " + str(proxy_base_url))
        return proxy_base_url


async def release_proxy() -> None:
    with _CC_PROXY_LOCK:
        global _CC_PROXY_USAGE_COUNT
        if _CC_PROXY_USAGE_COUNT > 0:
            _CC_PROXY_USAGE_COUNT -= 1
            if _CC_PROXY_USAGE_COUNT == 0:
                _stop_cc_proxy_locked()
                logger.info("Released claude proxy server")
 