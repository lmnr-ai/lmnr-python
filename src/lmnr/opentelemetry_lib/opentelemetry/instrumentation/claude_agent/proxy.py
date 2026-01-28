"""Per-transport proxy management for Claude Agent instrumentation."""

from __future__ import annotations

import os
import threading

from lmnr_claude_code_proxy import ProxyServer

from lmnr.sdk.log import get_default_logger

from .utils import wait_for_port

logger = get_default_logger(__name__)

# Thread-safe port allocation
_PORT_LOCK = threading.Lock()
_DEFAULT_PORT = 45667
_NEXT_PORT = _DEFAULT_PORT
_ALLOCATED_PORTS: set[int] = set()
_DEFAULT_MAX_PORTS = 5000
_DEFAULT_MIN_PORTS = 10

# Maximum port range: can be configured via LMNR_CC_PROXY_MAX_PORTS env var
# Capped at 65535 - DEFAULT_PORT to stay within valid port range
_MAX_PORTS_ABSOLUTE = 65535 - _DEFAULT_PORT  # = 19868


def _get_max_ports() -> int:
    """Get the configured max ports from env, capped at absolute maximum."""
    try:
        env_val = os.environ.get("LMNR_CC_PROXY_MAX_PORTS")
        if env_val:
            configured = abs(int(env_val))
            return max(min(configured, _MAX_PORTS_ABSOLUTE), _DEFAULT_MIN_PORTS)
    except (ValueError, TypeError):
        pass
    return max(min(_DEFAULT_MAX_PORTS, _MAX_PORTS_ABSOLUTE), _DEFAULT_MIN_PORTS)


def _allocate_port() -> int:
    """Allocate next available port in thread-safe manner with wraparound."""
    with _PORT_LOCK:
        global _NEXT_PORT
        max_ports = _get_max_ports()

        # Try to find an available port, with wraparound
        attempts = 0
        while _NEXT_PORT in _ALLOCATED_PORTS:
            _NEXT_PORT = _DEFAULT_PORT + ((_NEXT_PORT - _DEFAULT_PORT + 1) % max_ports)
            attempts += 1
            if attempts >= max_ports:
                # All ports in range are allocated, expand beyond if possible
                _NEXT_PORT = _DEFAULT_PORT + attempts
                if _NEXT_PORT > 65535:
                    raise RuntimeError(
                        f"All {max_ports} proxy ports are in use. "
                        "Increase LMNR_CC_PROXY_MAX_PORTS or wait for ports to be released."
                    )

        port = _NEXT_PORT
        _ALLOCATED_PORTS.add(port)
        _NEXT_PORT = _DEFAULT_PORT + ((_NEXT_PORT - _DEFAULT_PORT + 1) % max_ports)
        return port


def _release_port(port: int) -> None:
    """Release port back to pool."""
    with _PORT_LOCK:
        _ALLOCATED_PORTS.discard(port)


def create_proxy_for_transport() -> ProxyServer:
    """
    Create a new ProxyServer instance for a Transport.

    Returns:
        ProxyServer instance (not yet started)
    """
    port = _allocate_port()
    proxy = ProxyServer(port=port)
    # Store port for cleanup
    proxy._allocated_port = port  # type: ignore
    return proxy


def start_proxy(proxy: ProxyServer, target_url: str, max_retries: int = 10) -> str:
    """
    Start a proxy server, retrying with different ports if occupied.

    Args:
        proxy: ProxyServer instance
        target_url: Upstream URL to proxy to (required)
        max_retries: Maximum number of port allocation attempts (default: 10)

    Returns:
        Proxy base URL (e.g., "http://127.0.0.1:45667")

    Raises:
        RuntimeError: If proxy fails to start after all retries
    """

    # Retry loop for port allocation
    for attempt in range(max_retries):
        current_port = proxy.port

        # Start server
        try:
            proxy.run_server(target_url)
        except Exception as e:
            logger.debug(
                "Failed to start proxy server on port %d (attempt %d/%d): %s",
                current_port,
                attempt + 1,
                max_retries,
                e,
            )

            # Release current port and allocate a new one
            if hasattr(proxy, "_allocated_port"):
                _release_port(proxy._allocated_port)  # type: ignore

            # Allocate new port for next attempt
            if attempt < max_retries - 1:
                new_port = _allocate_port()
                proxy.port = new_port
                proxy._allocated_port = new_port  # type: ignore
                continue
            else:
                # Last attempt failed
                raise RuntimeError(
                    f"Failed to start proxy after {max_retries} attempts. Last error on port {current_port}: {e}"
                ) from e

        # Wait for readiness
        if not wait_for_port(proxy.port, timeout=5.0):
            logger.debug(
                "Proxy failed readiness check on port %d (attempt %d/%d)",
                proxy.port,
                attempt + 1,
                max_retries,
            )
            try:
                proxy.stop_server()
            except Exception as e:
                logger.debug("Error stopping proxy on port %d: %s", proxy.port, e)
            finally:
                # Release current port and allocate a new one
                if hasattr(proxy, "_allocated_port"):
                    _release_port(proxy._allocated_port)  # type: ignore

            if attempt < max_retries - 1:
                new_port = _allocate_port()
                proxy.port = new_port
                proxy._allocated_port = new_port  # type: ignore
                continue
            else:
                raise RuntimeError(
                    f"Proxy failed to start after {max_retries} attempts (readiness check failed)"
                )

        proxy_url = f"http://127.0.0.1:{proxy.port}"
        logger.info("Started proxy server on: %s", proxy_url)
        return proxy_url

    # Should not reach here, but just in case
    raise RuntimeError(f"Failed to start proxy after {max_retries} attempts")


def stop_proxy(proxy: ProxyServer) -> None:
    """
    Stop proxy server and release resources.

    The Rust proxy handles timeouts internally, so this is safe to call
    from both sync and async contexts.
    """
    port = proxy.port
    try:
        proxy.stop_server()
        logger.debug("Stopped proxy server on port %d", port)
    except Exception as e:
        logger.debug("Error stopping proxy on port %d: %s", port, e)
    finally:
        if hasattr(proxy, "_allocated_port"):
            _release_port(proxy._allocated_port)  # type: ignore


def publish_span_context_to_proxy(
    proxy: ProxyServer,
    trace_id: str,
    span_id: str,
    project_api_key: str,
    span_path: list[str],
    span_ids_path: list[str],
    laminar_url: str,
) -> None:
    """Publish trace context to specific proxy instance."""
    try:
        proxy.set_current_trace(
            trace_id=trace_id,
            span_id=span_id,
            project_api_key=project_api_key,
            span_path=span_path,
            span_ids_path=span_ids_path,
            laminar_url=laminar_url,
        )
    except Exception as e:
        logger.debug("Failed to publish span context to proxy: %s", e)
