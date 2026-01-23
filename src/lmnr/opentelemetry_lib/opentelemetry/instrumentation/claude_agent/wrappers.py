"""Wrapper functions for Claude Agent instrumentation."""

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
    resolve_target_url,
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
            logger.warning("Failed to import SubprocessCLITransport, skipping proxy setup")
            return await wrapped(*args, **kwargs)

        # Create and start proxy
        proxy = create_proxy_for_transport()
        proxy_url = start_proxy(proxy)

        # Determine if this is a truly custom transport (not SubprocessCLITransport)
        # SubprocessCLITransport gets proxy config via options.env, custom ones need global env
        is_custom = not isinstance(instance, SubprocessCLITransport)

        # For truly custom transports, we need to set global env vars as fallback
        # since we can't control how they handle environment.
        # For SubprocessCLITransport, proxy config is already in options.env
        if is_custom:

            original_env = setup_proxy_env(proxy_url)
            env_set_keys = {k for k, v in original_env.items() if v is not None}
        else:
            original_env = None
            env_set_keys = None

        # Store context on instance
        context: dict[str, Any] = {
            "proxy": proxy,
            "proxy_url": proxy_url,
            "is_custom_transport": is_custom,
        }
        if is_custom:
            context["original_env"] = original_env
            context["env_set_keys"] = env_set_keys

        instance.__lmnr_context = context

        # Connect transport
        try:
            result = await wrapped(*args, **kwargs)

            # After successful connection, publish current span context to proxy
            publish_span_context_for_transport(instance)

            return result
        except Exception:
            # If connect fails, clean up proxy

            stop_proxy(proxy)

            # Restore env if custom transport
            if is_custom and original_env:

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
                # Restore global env only for custom transports
                if context.get("is_custom_transport"):
                    from .utils import restore_env

                    restore_env(
                        context.get("original_env", {}),
                        context.get("env_set_keys", set()),
                    )

                stop_proxy(context["proxy"])
                delattr(instance, "__lmnr_context")

    return wrapper


def get_proxy_url_for_options() -> str:
    """Get the proxy URL that will be used for the next transport connection."""
    from .proxy import _NEXT_PORT

    # Return the URL that will be allocated
    return f"http://127.0.0.1:{_NEXT_PORT}"


def update_options_env_for_proxy(options, proxy_url: str) -> None:
    """Update options.env dict with proxy environment variables."""

    # Determine original target URL
    if "ANTHROPIC_ORIGINAL_BASE_URL" not in options.env:
        target_url = resolve_target_url()
        if target_url:
            options.env["ANTHROPIC_ORIGINAL_BASE_URL"] = target_url

    # Set proxy URL
    options.env["ANTHROPIC_BASE_URL"] = proxy_url

    # Handle Foundry-specific env vars
    if is_truthy_env(options.env.get("CLAUDE_CODE_USE_FOUNDRY")):
        if FOUNDRY_BASE_URL_ENV in options.env:
            options.env[FOUNDRY_BASE_URL_ENV] = proxy_url
        # Remove FOUNDRY_RESOURCE_ENV if present
        options.env.pop(FOUNDRY_RESOURCE_ENV, None)


def wrap_query(to_wrap: dict[str, Any]):
    """Wrap query() function to set proxy env in options."""

    def wrapper(wrapped, instance, args, kwargs):
        # Signature: async def query(*, prompt: str, options: ClaudeAgentOptions | None = None, ...)

        options = kwargs.get("options")
        transport = kwargs.get("transport")

        # If user provided a custom transport, wrap it dynamically
        if transport:
            wrap_custom_transport_if_needed(transport)
        else:
            # For standard SubprocessCLITransport path, update options.env with proxy URL
            if options is None:
                # Create default options if none provided
                try:
                    from claude_agent_sdk.types import ClaudeAgentOptions

                    options = ClaudeAgentOptions()
                    kwargs["options"] = options
                except (ImportError, ModuleNotFoundError):
                    logger.warning("Failed to import ClaudeAgentOptions, skipping proxy setup")
                    pass

            # Get the proxy URL that will be allocated
            proxy_url = get_proxy_url_for_options()

            # Update options.env to include proxy configuration
            update_options_env_for_proxy(options, proxy_url)

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
                        try:
                            collected.append(item)
                        except StopAsyncIteration:
                            break
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
    """Wrap ClaudeSDKClient.__init__ to set proxy env in options."""

    def wrapper(wrapped, instance, args, kwargs):
        try:
            from claude_agent_sdk._internal.transport.subprocess_cli import (
                SubprocessCLITransport,
            )
        except (ImportError, ModuleNotFoundError):
            # If import fails, just call original without instrumentation
            logger.warning("Failed to import SubprocessCLITransport, skipping proxy setup")
            return wrapped(*args, **kwargs)

        # Extract options and transport
        options = None
        transport = None

        if args:
            if len(args) > 0:
                options = args[0]
            if len(args) > 1:
                transport = args[1]

        if "options" in kwargs:
            options = kwargs["options"]
        if "transport" in kwargs:
            transport = kwargs["transport"]

        if transport:
            # User provided a custom transport
            if isinstance(transport, SubprocessCLITransport):
                # It's a SubprocessCLITransport - we can update its options.env
                if hasattr(transport, "_options"):
                    transport_options = transport._options
                    proxy_url = get_proxy_url_for_options()
                    update_options_env_for_proxy(transport_options, proxy_url)

            # Wrap the transport for proxy lifecycle management
            wrap_custom_transport_if_needed(transport)
        else:
            # Standard path: no custom transport, will use SubprocessCLITransport created by client
            if options is None:
                # Create default options if none provided
                try:
                    from claude_agent_sdk.types import ClaudeAgentOptions

                    options = ClaudeAgentOptions()
                    if args:
                        args = (options, *args[1:])
                    else:
                        kwargs["options"] = options
                except (ImportError, ModuleNotFoundError):
                    logger.warning("Failed to import ClaudeAgentOptions, skipping proxy setup")
                    pass

            # Get the proxy URL that will be allocated
            proxy_url = get_proxy_url_for_options()

            # Update options.env to include proxy configuration
            update_options_env_for_proxy(options, proxy_url)

        # Call original init
        return wrapped(*args, **kwargs)

    return wrapper


def wrap_custom_transport_if_needed(transport):
    """Dynamically wrap custom transport's connect/close methods."""
    try:
        from claude_agent_sdk._internal.transport.subprocess_cli import (
            SubprocessCLITransport,
        )
    except (ImportError, ModuleNotFoundError):
        # If import fails, skip wrapping
        logger.warning("Failed to import SubprocessCLITransport, skipping transport wrapping")
        return

    # Skip if already wrapped or is SubprocessCLITransport (handled by instrumentation)
    if hasattr(transport, "__lmnr_context") or isinstance(
        transport, SubprocessCLITransport
    ):
        return

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
