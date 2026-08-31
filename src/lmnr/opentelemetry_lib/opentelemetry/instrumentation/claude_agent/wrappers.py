"""Wrapper functions for Claude Agent instrumentation."""

import asyncio
import os
from typing import Any, Sequence

from lmnr import Laminar
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.wrapper_helpers import (
    add_spec_wrapper,
)
from lmnr.sdk.log import get_default_logger

from opentelemetry.trace import Status, StatusCode

from .proxy import create_proxy_for_transport, start_proxy, stop_proxy, _release_port
from .types import ClaudeAgentSpec
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
    build_proxy_flag_settings,
    read_claude_settings_env,
    PROXY_BASE_URL_ENV_KEYS,
    PROXY_ENV_KEYS,
    FOUNDRY_BASE_URL_ENV,
    FOUNDRY_RESOURCE_ENV,
    FOUNDRY_USE_ENV,
    BEDROCK_BASE_URL_ENV,
    BEDROCK_USE_ENV,
    VERTEX_BASE_URL_ENV,
    VERTEX_USE_ENV,
)

logger = get_default_logger(__name__)

# Timeout for cleanup operations to prevent hanging on stuck underlying calls
DEFAULT_CLEANUP_TIMEOUT = 4.0


def wrap_sync(
    to_wrap: ClaudeAgentSpec,
    wrapped,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
):
    """Wrapper for synchronous methods."""

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


async def wrap_async(
    to_wrap: ClaudeAgentSpec,
    wrapped,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
):
    """Wrapper for async methods."""

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


def wrap_async_gen(
    to_wrap: ClaudeAgentSpec,
    wrapped,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
):
    """Wrapper for async generator methods (streaming)."""

    async def generator():
        span = Laminar.start_span(
            span_name(to_wrap),
            span_type=to_wrap.get("span_type", "DEFAULT"),
        )
        collected = []
        async_iter = None

        if to_wrap.get("should_publish_span_context"):
            with Laminar.use_span(span):
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
        except GeneratorExit:
            # User broke out of the loop - this is normal, don't record as error
            raise
        except asyncio.CancelledError:
            # Request was cancelled (e.g., FastAPI client disconnect)
            # Don't record as error, just propagate
            raise
        except Exception as e:  # pylint: disable=broad-except
            with Laminar.use_span(span):
                span.set_status(Status(StatusCode.ERROR))
                span.record_exception(e)
            raise
        finally:
            await _cleanup_async_iter(async_iter, span)
            with Laminar.use_span(span):
                record_output(span, to_wrap, collected)
            span.end()

    return generator()


async def _cleanup_async_iter(async_iter, span) -> None:
    """
    Clean up an async iterator with timeout and cancellation protection.

    Shields cleanup from cancellation and adds timeout to prevent hanging if
    underlying close operations get stuck. Swallows all exceptions since cleanup
    failures are expected in scenarios like early user breaks or closed transports.
    """
    if not async_iter or not hasattr(async_iter, "aclose"):
        return

    try:
        with Laminar.use_span(span):
            # Shield from cancellation and add timeout to prevent hanging
            await asyncio.wait_for(
                asyncio.shield(async_iter.aclose()), timeout=DEFAULT_CLEANUP_TIMEOUT
            )
    except BaseException:
        # Swallow all exceptions - cleanup failures are expected when:
        # - Subprocess already terminated (ProcessError)
        # - User broke out of generator early (GeneratorExit)
        # - Request was cancelled (CancelledError, TimeoutError)
        pass


async def wrap_transport_connect(
    to_wrap: ClaudeAgentSpec,
    wrapped,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
):
    """Wrap Transport.connect to start proxy before connecting."""
    try:
        from claude_agent_sdk._internal.transport.subprocess_cli import (
            SubprocessCLITransport,
        )
    except (ImportError, ModuleNotFoundError):
        logger.warning(
            "Failed to import SubprocessCLITransport, skipping proxy setup"
        )
        return await wrapped(*args, **kwargs)

    # Read options.env BEFORE modifying to avoid circular proxy config
    options = getattr(instance, "_options", None)
    env_dict = options.env if options is not None else {}
    session_cwd = getattr(options, "cwd", None)
    # Read the DOCUMENTED options.setting_sources only. Do not try to infer
    # the SDK's private skills-based default (_apply_skills_defaults narrows
    # this to user,project) — that is an internal we must not track. Reading
    # a layer the CLI ignored is a safe over-read for upstream resolution:
    # the flag layer still forces every base URL to the proxy.
    setting_sources = getattr(options, "setting_sources", None)
    target_url = resolve_target_url_from_env(
        env_dict,
        cwd=session_cwd,
        setting_sources=setting_sources,
        # options.settings is the highest layer the CLI reads and
        # apply_settings_proxy_override rewrites its base URLs to the proxy.
        settings=getattr(options, "settings", None),
    )

    if target_url is None:
        raise RuntimeError("Invalid provider configuration")

    proxy = create_proxy_for_transport()
    proxy_url = start_proxy(proxy, target_url=target_url)

    # Custom transports use global env, SubprocessCLITransport uses options.env
    is_custom = not isinstance(instance, SubprocessCLITransport)

    options_env_snapshot = {}
    original_settings: Any = None
    settings_overridden = False
    if is_custom:
        original_env = setup_proxy_env(proxy_url, session_cwd)
        env_set_keys = {k for k, v in original_env.items() if v is not None}
    else:
        if options is not None:
            options_env_snapshot = snapshot_options_env_for_proxy(options)
            update_options_env_for_proxy(options, proxy_url, target_url)
            original_settings = getattr(options, "settings", None)
            settings_overridden = apply_settings_proxy_override(
                options, proxy_url
            )

        original_env = {}
        env_set_keys = set()

        # Remove from os.environ (mutually exclusive with ANTHROPIC_BASE_URL)
        if FOUNDRY_RESOURCE_ENV in os.environ:
            original_env[FOUNDRY_RESOURCE_ENV] = os.environ[FOUNDRY_RESOURCE_ENV]
            env_set_keys.add(FOUNDRY_RESOURCE_ENV)
            os.environ.pop(FOUNDRY_RESOURCE_ENV)

        # Prevent subprocess from routing through corporate proxy
        for proxy_var in PROXY_ENV_KEYS:
            if proxy_var in os.environ:
                original_env[proxy_var] = os.environ[proxy_var]
                env_set_keys.add(proxy_var)
                os.environ.pop(proxy_var)

    context: dict[str, Any] = {
        "proxy": proxy,
        "proxy_url": proxy_url,
        "is_custom_transport": is_custom,
        "original_env": original_env,
        "env_set_keys": env_set_keys,
        "options_env_snapshot": options_env_snapshot,
        "original_settings": original_settings,
        "settings_overridden": settings_overridden,
    }

    instance.__lmnr_context = context
    instance.__lmnr_wrapped = True

    try:
        result = await wrapped(*args, **kwargs)
        publish_span_context_for_transport(instance)
        return result
    except Exception:
        stop_proxy(proxy)

        if original_env:
            restore_env(original_env, env_set_keys or set())

        if options_env_snapshot and options is not None:
            restore_options_env_from_snapshot(options, options_env_snapshot)

        if settings_overridden and options is not None:
            options.settings = original_settings

        try:
            delattr(instance, "__lmnr_context")
        except Exception:
            pass
        raise


async def wrap_transport_close(
    to_wrap: ClaudeAgentSpec,
    wrapped,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
):
    """Wrap Transport.close to stop proxy after closing."""
    try:
        return await wrapped(*args, **kwargs)
    finally:
        await _cleanup_transport_context(instance)


async def _cleanup_transport_context(instance) -> None:
    """
    Cleanup proxy and restore environment when transport closes.

    Shields from cancellation and adds timeout to prevent hanging. Runs proxy stop
    in thread pool to avoid blocking event loop while ensuring cleanup completes.
    """
    context: dict[str, Any] | None = getattr(instance, "__lmnr_context", None)
    if not context:
        return

    async def _do_cleanup():
        try:
            if context.get("original_env"):
                restore_env(
                    context.get("original_env", {}),
                    context.get("env_set_keys", set()),
                )

            options = getattr(instance, "_options", None)
            if context.get("settings_overridden") and options is not None:
                options.settings = context.get("original_settings")

            # Release port immediately to prevent leaks
            # Must happen before background cleanup in case event loop shuts down
            proxy = context.get("proxy")
            if proxy and hasattr(proxy, "_allocated_port"):
                _release_port(proxy._allocated_port)  # type: ignore
                # Prevent double-release in stop_proxy
                try:
                    delattr(proxy, "_allocated_port")
                except Exception:
                    pass

            if proxy:
                try:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, stop_proxy, proxy)
                except RuntimeError:
                    try:
                        stop_proxy(proxy)
                    except Exception:
                        pass
        finally:
            try:
                delattr(instance, "__lmnr_context")
            except Exception:
                pass

    try:
        await asyncio.wait_for(
            asyncio.shield(_do_cleanup()), timeout=DEFAULT_CLEANUP_TIMEOUT
        )
    except BaseException:
        # Swallow all exceptions - cleanup failures are expected
        pass


def snapshot_options_env_for_proxy(options) -> dict[str, str | None]:
    """
    Snapshot keys in options.env that will be modified by update_options_env_for_proxy.

    This enables restoration on error so retries work correctly.

    Args:
        options: ClaudeAgentOptions instance with .env dict

    Returns:
        Dictionary mapping keys to their original values (or None if not present)
    """
    # PROXY_ENV_KEYS covers both the upper and lower case spellings that
    # update_options_env_for_proxy pops; a key popped but not snapshotted is lost
    # from options.env when a failed connect restores.
    keys_to_snapshot = [
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_ORIGINAL_BASE_URL",
        *PROXY_ENV_KEYS,
        FOUNDRY_BASE_URL_ENV,
        FOUNDRY_RESOURCE_ENV,
        FOUNDRY_USE_ENV,
        BEDROCK_BASE_URL_ENV,
        BEDROCK_USE_ENV,
        VERTEX_BASE_URL_ENV,
        VERTEX_USE_ENV,
    ]

    snapshot = {}
    for key in keys_to_snapshot:
        snapshot[key] = options.env.get(key)

    return snapshot


def restore_options_env_from_snapshot(options, snapshot: dict[str, str | None]) -> None:
    """
    Restore options.env from a snapshot created by snapshot_options_env_for_proxy.

    Args:
        options: ClaudeAgentOptions instance with .env dict
        snapshot: Dictionary mapping keys to their original values
    """
    for key, value in snapshot.items():
        if value is None:
            # Key was not present originally, remove it
            options.env.pop(key, None)
        else:
            # Restore original value
            options.env[key] = value


def apply_settings_proxy_override(options, proxy_url: str) -> bool:
    """
    Point ``options.settings`` at the proxy so settings.json can't bypass it.

    Claude Code's settings ``env`` block outranks the subprocess environment, so
    ``options.env`` alone does not redirect a user whose base URL lives in
    ``~/.claude/settings.json`` (lmnr#2167). ``--settings`` is the highest
    user-controlled layer; the user's files on disk are never modified.

    Returns whether ``options.settings`` was rewritten. The caller snapshots the
    previous value itself, so restoring on error does not have to distinguish
    "the original was ``None``" from "we never touched it".
    """
    if not hasattr(options, "settings"):
        return False

    session_cwd = getattr(options, "cwd", None)
    setting_sources = getattr(options, "setting_sources", None)

    original = options.settings
    updated = build_proxy_flag_settings(
        original, proxy_url, cwd=session_cwd, setting_sources=setting_sources
    )
    if updated is None:
        logger.warning(
            "Could not read options.settings %r; Claude Code settings that define "
            "a base URL will bypass the Laminar proxy and produce no LLM spans.",
            original,
        )
        return False

    options.settings = updated

    conflicting = [
        key
        for key, value in read_claude_settings_env(
            session_cwd, setting_sources
        ).items()
        if key in PROXY_BASE_URL_ENV_KEYS and value not in ("", proxy_url)
    ]
    if conflicting:
        logger.info(
            "Claude Code settings define %s, which outranks the subprocess "
            "environment. Injected --settings so the Laminar proxy at %s still "
            "intercepts API traffic; your settings files were not modified.",
            ", ".join(sorted(conflicting)),
            proxy_url,
        )

    return True


def update_options_env_for_proxy(options, proxy_url: str, target_url: str) -> None:
    """
    Update options.env to point subprocess to proxy.

    - Sets ANTHROPIC_BASE_URL to proxy URL
    - Sets ANTHROPIC_ORIGINAL_BASE_URL to target URL (for proxy to forward to)
    - Removes HTTP_PROXY and HTTPS_PROXY from options.env
      since our proxy will handle forwarding to them
    - If Foundry enabled:
        - sets ANTHROPIC_FOUNDRY_BASE_URL to proxy URL
        - Removes ANTHROPIC_FOUNDRY_RESOURCE from options.env (mutually exclusive)
    - If Bedrock enabled:
        - sets ANTHROPIC_BEDROCK_BASE_URL to proxy URL
    - ALL OTHER env vars passed intact

    Note: For SubprocessCLITransport, HTTP_PROXY, HTTPS_PROXY, and ANTHROPIC_FOUNDRY_RESOURCE
    from os.environ are handled separately in wrap_transport_connect by temporarily removing
    them before subprocess starts (since subprocess inherits os.environ).

    Args:
        options: ClaudeAgentOptions instance with .env dict
        proxy_url: Proxy URL (e.g., "http://127.0.0.1:45667")
        target_url: Original target URL to forward to (e.g., "https://api.anthropic.com")
    """

    settings_env = read_claude_settings_env(
        getattr(options, "cwd", None), getattr(options, "setting_sources", None)
    )

    def get_env_value(key: str) -> str | None:
        return options.env.get(key) or os.environ.get(key) or settings_env.get(key)

    foundry_enabled = is_truthy_env(get_env_value("CLAUDE_CODE_USE_FOUNDRY"))

    options.env["ANTHROPIC_BASE_URL"] = proxy_url
    options.env["ANTHROPIC_ORIGINAL_BASE_URL"] = target_url

    for proxy_var in PROXY_ENV_KEYS:
        options.env.pop(proxy_var, None)

    if FOUNDRY_RESOURCE_ENV in options.env:
        options.env.pop(FOUNDRY_RESOURCE_ENV)

    if foundry_enabled:
        if "CLAUDE_CODE_USE_FOUNDRY" not in options.env:
            options.env["CLAUDE_CODE_USE_FOUNDRY"] = "1"
        options.env[FOUNDRY_BASE_URL_ENV] = proxy_url

    bedrock_enabled = is_truthy_env(get_env_value(BEDROCK_USE_ENV))
    if bedrock_enabled:
        if BEDROCK_USE_ENV not in options.env:
            options.env[BEDROCK_USE_ENV] = "1"
        options.env[BEDROCK_BASE_URL_ENV] = proxy_url

    vertex_enabled = is_truthy_env(get_env_value(VERTEX_USE_ENV))
    if vertex_enabled:
        if VERTEX_USE_ENV not in options.env:
            options.env[VERTEX_USE_ENV] = "1"
        options.env[VERTEX_BASE_URL_ENV] = proxy_url


def wrap_query(
    to_wrap: ClaudeAgentSpec,
    wrapped,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
):
    """Wrap query() function - handles custom transport wrapping."""
    transport = kwargs.get("transport")

    if transport:
        try:
            from claude_agent_sdk._internal.transport.subprocess_cli import (
                SubprocessCLITransport,
            )

            if not isinstance(transport, SubprocessCLITransport):
                wrap_custom_transport_if_needed(transport)
        except (ImportError, ModuleNotFoundError):
            wrap_custom_transport_if_needed(transport)

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

                async for item in async_iter:
                    collected.append(item)
                    yield item
            except GeneratorExit:
                raise
            except asyncio.CancelledError:
                raise
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR))
                span.record_exception(e)
                raise
            finally:
                await _cleanup_async_iter(async_iter, span)
                record_output(span, to_wrap, collected)

    return generator()


def wrap_client_init(
    to_wrap: ClaudeAgentSpec,
    wrapped,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
):
    """Wrap ClaudeSDKClient.__init__ to handle custom transport wrapping."""
    try:
        from claude_agent_sdk._internal.transport.subprocess_cli import (
            SubprocessCLITransport,
        )
    except (ImportError, ModuleNotFoundError):
        logger.warning(
            "Failed to import SubprocessCLITransport, skipping proxy setup"
        )
        return wrapped(*args, **kwargs)

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


def wrap_custom_transport_if_needed(transport):
    """
    Dynamically wrap custom transport's connect/close methods.

    Note: SubprocessCLITransport is already wrapped globally by the instrumentation
    and should be handled before calling this function to avoid double wrapping.
    """
    # Skip if already wrapped
    if hasattr(transport, "__lmnr_wrapped"):
        return

    transport.__lmnr_wrapped = True

    # A custom transport is wrapped per-instance at runtime, so it has no row in
    # WRAPPED_FUNCTIONS. `add_spec_wrapper` is the same adapter the instrumentor
    # uses, binding a spec so the handler still gets the wrapt-shaped call.
    # Neither transport handler reads the spec, so a minimal one is enough.
    connect_wrapper = add_spec_wrapper(
        wrap_transport_connect,
        ClaudeAgentSpec(
            package_name="",
            method_name="connect",
            is_async=True,
            wrapper_function=wrap_transport_connect,
        ),
    )
    close_wrapper = add_spec_wrapper(
        wrap_transport_close,
        ClaudeAgentSpec(
            package_name="",
            method_name="close",
            is_async=True,
            wrapper_function=wrap_transport_close,
        ),
    )

    original_connect = transport.connect
    original_close = transport.close

    async def wrapped_connect_custom():
        return await connect_wrapper(original_connect, transport, (), {})

    async def wrapped_close_custom():
        return await close_wrapper(original_close, transport, (), {})

    transport.connect = wrapped_connect_custom
    transport.close = wrapped_close_custom
