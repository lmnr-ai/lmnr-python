"""Wrapper functions for Claude Agent instrumentation."""

import os
from typing import Any

from lmnr import Laminar
from lmnr.sdk.log import get_default_logger

from opentelemetry.trace import Status, StatusCode

from .proxy import create_proxy_for_transport, start_proxy, stop_proxy
from .span_utils import (
    span_name,
    record_input,
    record_output,
    publish_span_context_for_transport,
)
from .utils import (
    setup_proxy_env,
    restore_env,
    resolve_target_url_from_env,
    is_truthy_env,
    FOUNDRY_BASE_URL_ENV,
    FOUNDRY_RESOURCE_ENV,
)

logger = get_default_logger(__name__)


def wrap_sync(to_wrap: dict[str, Any]):
    """Wrapper for synchronous methods."""

    def wrapper(wrapped, instance, args, kwargs):
        with Laminar.start_as_current_span(
            span_name(to_wrap),
            span_type=to_wrap.get("span_type", "DEFAULT"),
        ) as span:
            record_input(span, wrapped, args, kwargs)

            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:  # pylint: disable=broad-except
                span.set_status(Status(StatusCode.ERROR))
                span.record_exception(e)
                raise

            record_output(span, to_wrap, result)
            return result

    return wrapper


def wrap_async(to_wrap: dict[str, Any]):
    """Wrapper for async methods."""

    async def wrapper(wrapped, instance, args, kwargs):
        with Laminar.start_as_current_span(
            span_name(to_wrap),
            span_type=to_wrap.get("span_type", "DEFAULT"),
        ) as span:
            record_input(span, wrapped, args, kwargs)

            if to_wrap.get("should_publish_span_context"):
                # Get transport from instance (ClaudeSDKClient._transport)
                if hasattr(instance, "_transport"):
                    publish_span_context_for_transport(instance._transport)

            try:
                result = await wrapped(*args, **kwargs)
            except Exception as e:  # pylint: disable=broad-except
                span.set_status(Status(StatusCode.ERROR))
                span.record_exception(e)
                raise

            record_output(span, to_wrap, result)

            return result

    return wrapper


def wrap_async_gen(to_wrap: dict[str, Any]):
    """Wrapper for async generator methods (streaming)."""

    def wrapper(wrapped, instance, args, kwargs):
        async def generator():
            span = Laminar.start_span(
                span_name(to_wrap),
                span_type=to_wrap.get("span_type", "DEFAULT"),
            )
            collected = []
            async_iter = None

            if to_wrap.get("should_publish_span_context"):
                with Laminar.use_span(span):
                    # Get transport from instance (ClaudeSDKClient._transport)
                    if hasattr(instance, "_transport"):
                        publish_span_context_for_transport(instance._transport)

            try:
                with Laminar.use_span(span):
                    record_input(span, wrapped, args, kwargs)
                    async_source = wrapped(*args, **kwargs)
                    async_iter = (
                        async_source.__aiter__()
                        if hasattr(async_source, "__aiter__")
                        else async_source
                    )

                while True:
                    try:
                        with Laminar.use_span(
                            span, record_exception=False, set_status_on_exception=False
                        ):
                            item = await async_iter.__anext__()
                            collected.append(item)
                    except StopAsyncIteration:
                        break
                    yield item
            except Exception as e:  # pylint: disable=broad-except
                with Laminar.use_span(span):
                    span.set_status(Status(StatusCode.ERROR))
                    span.record_exception(e)
                raise
            finally:
                if async_iter and hasattr(async_iter, "aclose"):
                    try:
                        with Laminar.use_span(span):
                            await async_iter.aclose()
                    except Exception:  # pylint: disable=broad-except
                        pass
                with Laminar.use_span(span):
                    record_output(span, to_wrap, collected)
                    span.end()

        return generator()

    return wrapper


def wrap_transport_connect(to_wrap: dict[str, Any]):
    """Wrap Transport.connect to start proxy before connecting."""

    async def wrapper(wrapped, instance, args, kwargs):
        try:
            from claude_agent_sdk._internal.transport.subprocess_cli import (
                SubprocessCLITransport,
            )
        except (ImportError, ModuleNotFoundError):
            # If import fails, just call original without instrumentation
            logger.warning(
                "Failed to import SubprocessCLITransport, skipping proxy setup"
            )
            return await wrapped(*args, **kwargs)

        # Resolve target URL from transport's options.env (with os.environ fallback)
        # IMPORTANT: Read options.env BEFORE modifying options.env to avoid circular proxy config
        env_dict = instance._options.env if hasattr(instance, "_options") else {}
        target_url = resolve_target_url_from_env(env_dict)

        if target_url is None:
            raise RuntimeError("Invalid provider configuration")

        # Create and start proxy with resolved target URL
        proxy = create_proxy_for_transport()
        proxy_url = start_proxy(proxy, target_url=target_url)

        # Determine if this is a custom transport (not SubprocessCLITransport)
        # SubprocessCLITransport gets proxy config via options.env, custom ones need global env
        is_custom = not isinstance(instance, SubprocessCLITransport)

        # For custom transports, we need to set global env vars as fallback
        # since we can't control how they handle environment.
        # For SubprocessCLITransport, proxy config is in options.env
        if is_custom:
            original_env = setup_proxy_env(proxy_url)
            env_set_keys = {k for k, v in original_env.items() if v is not None}
        else:
            # For SubprocessCLITransport, update options.env with proxy config
            if hasattr(instance, "_options"):
                update_options_env_for_proxy(instance._options, proxy_url, target_url)

            original_env = {}
            env_set_keys = set()
            if FOUNDRY_RESOURCE_ENV in os.environ:
                original_env[FOUNDRY_RESOURCE_ENV] = os.environ[FOUNDRY_RESOURCE_ENV]
                env_set_keys.add(FOUNDRY_RESOURCE_ENV)
                os.environ.pop(FOUNDRY_RESOURCE_ENV)

        # Store context on instance
        context: dict[str, Any] = {
            "proxy": proxy,
            "proxy_url": proxy_url,
            "is_custom_transport": is_custom,
            "original_env": original_env,
            "env_set_keys": env_set_keys,
        }

        instance.__lmnr_context = context
        instance.__lmnr_wrapped = True

        # Connect transport
        try:
            result = await wrapped(*args, **kwargs)

            # After successful connection, publish current span context to proxy
            publish_span_context_for_transport(instance)

            return result
        except Exception:
            # If connect fails, clean up proxy
            stop_proxy(proxy)

            # Restore env (for both custom transports and SubprocessCLITransport)
            if original_env:
                restore_env(original_env, env_set_keys or set())

            delattr(instance, "__lmnr_context")
            raise

    return wrapper


def wrap_transport_close(to_wrap: dict[str, Any]):
    """Wrap Transport.close to stop proxy after closing."""

    async def wrapper(wrapped, instance, args, kwargs):
        from .proxy import stop_proxy

        try:
            # Close transport first
            return await wrapped(*args, **kwargs)
        finally:
            # Clean up proxy and restore environment if needed
            context: dict[str, Any] | None = getattr(instance, "__lmnr_context", None)
            if context:
                # Restore global env for both custom transports and SubprocessCLITransport
                # (SubprocessCLITransport might have had FOUNDRY_RESOURCE temporarily removed)
                if context.get("original_env"):
                    from .utils import restore_env

                    restore_env(
                        context.get("original_env", {}),
                        context.get("env_set_keys", set()),
                    )

                stop_proxy(context["proxy"])
                delattr(instance, "__lmnr_context")

    return wrapper


def update_options_env_for_proxy(options, proxy_url: str, target_url: str) -> None:
    """
    Update options.env to point subprocess to proxy.

    - Sets ANTHROPIC_BASE_URL to proxy URL
    - Sets ANTHROPIC_ORIGINAL_BASE_URL to target URL (for proxy to forward to)
    - If Foundry enabled:
        - sets ANTHROPIC_FOUNDRY_BASE_URL to proxy URL
        - Removes ANTHROPIC_FOUNDRY_RESOURCE from options.env (mutually exclusive)
    - ALL OTHER env vars passed intact

    Note: ANTHROPIC_FOUNDRY_RESOURCE from os.environ is handled separately in
    wrap_transport_connect by temporarily removing it before subprocess starts.

    Args:
        options: ClaudeAgentOptions instance with .env dict
        proxy_url: Proxy URL (e.g., "http://127.0.0.1:45667")
        target_url: Original target URL to forward to (e.g., "https://api.anthropic.com")
    """

    # Simple helper to check env
    def get_env_value(key: str) -> str | None:
        return options.env.get(key) or os.environ.get(key)

    foundry_enabled = is_truthy_env(get_env_value("CLAUDE_CODE_USE_FOUNDRY"))

    # Set proxy URL
    options.env["ANTHROPIC_BASE_URL"] = proxy_url

    # Store original target URL so proxy knows where to forward
    options.env["ANTHROPIC_ORIGINAL_BASE_URL"] = target_url

    # Remove FOUNDRY_RESOURCE from options.env (mutually exclusive with ANTHROPIC_BASE_URL)
    if FOUNDRY_RESOURCE_ENV in options.env:
        options.env.pop(FOUNDRY_RESOURCE_ENV)

    if foundry_enabled:
        if "CLAUDE_CODE_USE_FOUNDRY" not in options.env:
            options.env["CLAUDE_CODE_USE_FOUNDRY"] = "1"
        options.env[FOUNDRY_BASE_URL_ENV] = proxy_url


def wrap_query(to_wrap: dict[str, Any]):
    """Wrap query() function - handles custom transport wrapping."""

    def wrapper(wrapped, instance, args, kwargs):

        transport = kwargs.get("transport")

        # If user provided a custom transport, wrap it for proxy lifecycle
        # SubprocessCLITransport is already wrapped globally
        if transport:
            try:
                from claude_agent_sdk._internal.transport.subprocess_cli import (
                    SubprocessCLITransport,
                )

                # Only wrap non-SubprocessCLITransport (already globally wrapped)
                if not isinstance(transport, SubprocessCLITransport):
                    wrap_custom_transport_if_needed(transport)
            except (ImportError, ModuleNotFoundError):
                # If import fails, try wrapping anyway
                wrap_custom_transport_if_needed(transport)

        # Note: We don't pre-configure proxy URL here because we don't know the port
        # until the proxy is actually created in wrap_transport_connect

        # Continue with normal async gen wrapping (since query is streaming)
        async def generator():
            with Laminar.start_as_current_span(
                span_name(to_wrap),
                span_type=to_wrap.get("span_type", "DEFAULT"),
            ) as span:
                record_input(span, wrapped, args, kwargs)

                collected = []
                async_iter = None
                try:
                    async_iter = wrapped(*args, **kwargs)

                    # Note: Span context is published in wrap_transport_connect after connection

                    async for item in async_iter:

                        collected.append(item)

                        yield item
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR))
                    span.record_exception(e)
                    raise
                finally:
                    if async_iter and hasattr(async_iter, "aclose"):
                        try:
                            with Laminar.use_span(span):
                                await async_iter.aclose()
                        except Exception:
                            pass
                    record_output(span, to_wrap, collected)

        return generator()

    return wrapper


def wrap_client_init(to_wrap: dict[str, Any]):
    """Wrap ClaudeSDKClient.__init__ to handle custom transport wrapping."""

    def wrapper(wrapped, instance, args, kwargs):
        try:
            from claude_agent_sdk._internal.transport.subprocess_cli import (
                SubprocessCLITransport,
            )
        except (ImportError, ModuleNotFoundError):
            # If import fails, just call original without instrumentation
            logger.warning(
                "Failed to import SubprocessCLITransport, skipping proxy setup"
            )
            return wrapped(*args, **kwargs)

        # Extract transport from args/kwargs
        transport = None
        if args and len(args) > 1:
            transport = args[1]
        if "transport" in kwargs:
            transport = kwargs["transport"]

        # If user provided a custom transport, wrap it for proxy lifecycle
        # SubprocessCLITransport is already wrapped globally
        if transport and not isinstance(transport, SubprocessCLITransport):
            wrap_custom_transport_if_needed(transport)

        # Note: We don't pre-configure proxy URL here because we don't know the port
        # until the proxy is actually created in wrap_transport_connect

        # Call original init
        return wrapped(*args, **kwargs)

    return wrapper


def wrap_custom_transport_if_needed(transport):
    """
    Dynamically wrap custom transport's connect/close methods.

    Note: SubprocessCLITransport is already wrapped globally by the instrumentation
    and should be handled before calling this function to avoid double wrapping.
    """
    # Skip if already wrapped
    if hasattr(transport, "__lmnr_wrapped"):
        return

    # Mark transport as wrapped immediately to prevent double wrapping
    transport.__lmnr_wrapped = True

    # Create wrapper callables using the factory pattern
    connect_wrapper = wrap_transport_connect({"is_transport_connect": True})
    close_wrapper = wrap_transport_close({"is_transport_close": True})

    # Get original methods
    original_connect = transport.connect
    original_close = transport.close

    # Replace methods with wrapped versions
    async def wrapped_connect_custom():
        return await connect_wrapper(original_connect, transport, (), {})

    async def wrapped_close_custom():
        return await close_wrapper(original_close, transport, (), {})

    transport.connect = wrapped_connect_custom
    transport.close = wrapped_close_custom
