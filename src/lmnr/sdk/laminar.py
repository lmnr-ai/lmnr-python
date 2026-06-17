import asyncio
import atexit
import datetime
import json
import logging
import os
import re
import subprocess
import sys
import uuid
import warnings
from contextlib import contextmanager
from typing import Any, Callable, Generator, Literal

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.context import Context, get_value
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.trace import INVALID_TRACE_ID, Span, Status, StatusCode, use_span
from opentelemetry.util.types import AttributeValue
from typing_extensions import TypedDict

from lmnr.opentelemetry_lib import TracerManager
from lmnr.opentelemetry_lib.tracing import TracerWrapper
from lmnr.opentelemetry_lib.tracing.attributes import (
    ASSOCIATION_PROPERTIES,
    PARENT_SPAN_IDS_PATH,
    PARENT_SPAN_PATH,
    SESSION_ID,
    SPAN_TYPE,
    TRACE_TYPE,
    USER_ID,
    Attributes,
)
from lmnr.opentelemetry_lib.tracing.context import (
    CONTEXT_METADATA_KEY,
    CONTEXT_SESSION_ID_KEY,
    CONTEXT_TRACE_TYPE_KEY,
    CONTEXT_USER_ID_KEY,
    attach_context,
    detach_context,
    get_current_context,
    get_event_attributes_from_context,
    push_span_context,
    set_association_prop_context,
)
from lmnr.opentelemetry_lib.tracing.instruments import Instruments
from lmnr.opentelemetry_lib.tracing.processor import LaminarSpanProcessor
from lmnr.opentelemetry_lib.tracing.span import LaminarSpan
from lmnr.opentelemetry_lib.tracing.tracer import get_tracer_with_context
from lmnr.opentelemetry_lib.tracing.utils import set_association_props_in_context
from lmnr.sdk.utils import (
    from_env,
    get_otel_env_var,
    is_otel_attribute_value_type,
    json_dumps,
)

from .log import VerboseColorfulFormatter, get_level_from_env
from .types import (
    LaminarSpanContext,
    LaminarSpanType,
    SessionRecordingOptions,
    TraceType,
)


class ParsedParentSpanContext(TypedDict):
    """Parsed information from a parent span context."""

    otel_span_context: trace.SpanContext | None
    path: list[str]
    span_ids_path: list[str]
    user_id: str | None
    session_id: str | None
    trace_type: TraceType | None
    metadata: dict[str, Any] | None
    # The propagated debugger block, if any (used to arm the debug runtime on a
    # downstream run). None when the context carried no debug block.
    debug: Any | None


def _parse_parent_span_context(
    parent_span_context: LaminarSpanContext | dict | str | None,
    logger: logging.Logger,
) -> ParsedParentSpanContext:
    """Parse parent_span_context and extract all relevant information.

    Args:
        parent_span_context: Parent span context to parse
        logger: Logger for warnings

    Returns:
        ParsedParentSpanContext with otel_span_context, path, span_ids_path,
        user_id, session_id, trace_type, and metadata
    """
    if parent_span_context is None:
        return ParsedParentSpanContext(
            otel_span_context=None,
            path=[],
            span_ids_path=[],
            user_id=None,
            session_id=None,
            trace_type=None,
            metadata=None,
            debug=None,
        )

    path = []
    span_ids_path = []
    user_id = None
    session_id = None
    trace_type = None
    metadata = None
    debug = None
    laminar_span_context = None

    # Try to deserialize if dict or str
    if isinstance(parent_span_context, (dict, str)):
        try:
            laminar_span_context = LaminarSpanContext.deserialize(parent_span_context)
        except Exception:
            logger.warning(
                f"Could not deserialize parent_span_context: {parent_span_context}. "
                "Will use it as is."
            )
            laminar_span_context = parent_span_context
    else:
        laminar_span_context = parent_span_context

    # Extract path and association props from LaminarSpanContext
    if isinstance(laminar_span_context, LaminarSpanContext):
        path = laminar_span_context.span_path
        span_ids_path = laminar_span_context.span_ids_path
        user_id = laminar_span_context.user_id
        session_id = laminar_span_context.session_id
        if laminar_span_context.trace_type is not None:
            try:
                trace_type = (
                    TraceType(laminar_span_context.trace_type)
                    if isinstance(laminar_span_context.trace_type, str)
                    else laminar_span_context.trace_type
                )
            except (ValueError, TypeError):
                pass
        metadata = laminar_span_context.metadata
        debug = laminar_span_context.debug

    # Convert to OTEL span context
    try:
        otel_span_context = LaminarSpanContext.try_to_otel_span_context(
            laminar_span_context, logger
        )
    except ValueError as exc:
        logger.warning(f"Invalid span context provided: {exc}")
        return ParsedParentSpanContext(
            otel_span_context=None,
            path=path,
            span_ids_path=span_ids_path,
            user_id=user_id,
            session_id=session_id,
            trace_type=trace_type,
            metadata=metadata,
            debug=debug,
        )

    return ParsedParentSpanContext(
        otel_span_context=otel_span_context,
        path=path,
        span_ids_path=span_ids_path,
        user_id=user_id,
        session_id=session_id,
        trace_type=trace_type,
        metadata=metadata,
        debug=debug,
    )


class Laminar:
    __project_api_key: str | None = None
    __initialized: bool = False
    __base_http_url: str | None = None
    __global_metadata: dict[str, AttributeValue] = {}
    # The debug run's atexit pointer hook (a bound `runtime.emit_pointer`),
    # kept by reference so shutdown() can unregister it. atexit holds a strong
    # ref to whatever it registers, so without this an initialize()/shutdown()
    # loop (common in tests / notebooks) keeps every retired DebugRuntime — and
    # its replay cache — alive and uncollectable.
    __debug_exit_hook: Callable[[], None] | None = None
    # Base url / http port captured at initialize(), reused to build the clients
    # when the debug runtime is armed late from a propagated context (the
    # from-context path has no access to initialize()'s args).
    __base_url_for_debug: str | None = None
    __http_port_for_debug: int | None = None
    # Process-wide "run live" latch for v2 debugger replay (shared spec §7.3).
    # Set True on the first cache MISS so every later LLM call in this process
    # skips the cache endpoint and runs live; reset in shutdown(). Mirrors the
    # TS `Laminar.debugRunLive`.
    __debug_run_live: bool = False

    @classmethod
    def is_debug_run_live(cls) -> bool:
        """True once any LLM call in this run has seen a cache MISS."""
        return cls.__debug_run_live

    @classmethod
    def set_debug_run_live(cls, value: bool) -> None:
        """Latch (or reset) the process-wide debugger run-live flag."""
        cls.__debug_run_live = value

    @classmethod
    def initialize(
        cls,
        project_api_key: str | None = None,
        base_url: str | None = None,
        base_http_url: str | None = None,
        http_port: int | None = None,
        grpc_port: int | None = None,
        instruments: (
            list[Instruments] | set[Instruments] | tuple[Instruments] | None
        ) = None,
        disabled_instruments: (
            list[Instruments] | set[Instruments] | tuple[Instruments] | None
        ) = None,
        disable_batch: bool = False,
        max_export_batch_size: int | None = None,
        export_timeout_seconds: int | None = None,
        set_global_tracer_provider: bool = True,
        otel_logger_level: int = logging.ERROR,
        session_recording_options: SessionRecordingOptions | None = None,
        force_http: bool = False,
        metadata: dict[str, AttributeValue] | None = None,
    ):
        """Initialize Laminar context across the application.
        This method must be called before using any other Laminar methods or
        decorators.

        Args:
            project_api_key (str | None, optional): Laminar project api key.\
                You can generate one by going to the projects settings page on\
                the Laminar dashboard. If not specified, we will try to read\
                from the LMNR_PROJECT_API_KEY environment variable in os.environ\
                or in .env file. Defaults to None.
            base_url (str | None, optional): Laminar API url. Do NOT include\
                the port number, use `http_port` and `grpc_port`. If not\
                specified, defaults to https://api.lmnr.ai.
            base_http_url (str | None, optional): Laminar API http url. Only\
                set this if your Laminar backend HTTP is proxied through a\
                different host. If not specified, defaults to\
                https://api.lmnr.ai.
            http_port (int | None, optional): Laminar API http port. If not\
                specified, defaults to 443.
            grpc_port (int | None, optional): Laminar API grpc port. If not\
                specified, defaults to 8443.
            instruments (set[Instruments] | list[Instruments] | tuple[Instruments] | None, optional):
                Instruments to enable. Defaults to all instruments. You can pass\
                an empty set to disable all instruments. Read more:\
                https://docs.lmnr.ai/tracing/automatic-instrumentation
            disabled_instruments (set[Instruments] | list[Instruments] | tuple[Instruments] | None, optional):
                Instruments to disable. Defaults to None.
            disable_batch (bool, optional): If set to True, spans will be sent\
                immediately to the backend. Useful for debugging, but may cause\
                performance overhead in production. Defaults to False.
            max_export_batch_size (int | None, optional): Maximum number of spans\
                to export in a single batch. If not specified, defaults to 64\
                (lower than the OpenTelemetry default of 512). If you see\
                `DEADLINE_EXCEEDED` errors, try reducing this value.
            export_timeout_seconds (int | None, optional): Timeout for the OTLP\
                exporter. Defaults to 30 seconds (unlike the OpenTelemetry\
                default of 10 seconds). Defaults to None.
            set_global_tracer_provider (bool, optional): If set to True, the\
                Laminar tracer provider will be set as the global tracer provider.\
                OpenTelemetry allows only one tracer provider per app, so set this\
                to False, if you are using another tracing library. Setting this to\
                False may break some external instrumentations, e.g. LiteLLM.\
                Defaults to True.
            otel_logger_level (int, optional): OpenTelemetry logger level. Defaults\
                to logging.ERROR.
            session_recording_options (SessionRecordingOptions | None, optional): Options\
                for browser session recording. Currently supports 'mask_input'\
                (bool) to control whether input fields are masked during recording.\
                Defaults to None (uses default masking behavior).
            force_http (bool, optional): If set to True, the HTTP OTEL exporter will be\
                used instead of the gRPC OTEL exporter. Defaults to False.
        Raises:
            ValueError: If project API key is not set
        """
        if cls.is_initialized():
            cls.__logger.info(
                "Laminar is already initialized. Skipping initialization."
            )
            return

        cls.__project_api_key = project_api_key or from_env("LMNR_PROJECT_API_KEY")

        if (
            not cls.__project_api_key
            and not get_otel_env_var("ENDPOINT")
            and not get_otel_env_var("HEADERS")
        ):
            raise ValueError(
                "Please initialize the Laminar object with"
                " your project API key or set the LMNR_PROJECT_API_KEY"
                " environment variable in your environment or .env file"
            )

        cls._initialize_logger()

        url = base_url or from_env("LMNR_BASE_URL")
        if url:
            url = url.rstrip("/")
            if not url.startswith("http:") and not url.startswith("https:"):
                url = f"https://{url}"
            if match := re.search(r":(\d{1,5})$", url):
                url = url[: -len(match.group(0))]
                cls.__logger.info(f"Ignoring port in base URL: {match.group(1)}")
        http_url = base_http_url or url or "https://api.lmnr.ai"
        if not http_url.startswith("http:") and not http_url.startswith("https:"):
            http_url = f"https://{http_url}"
        if match := re.search(r":(\d{1,5})$", http_url):
            http_url = http_url[: -len(match.group(0))]
            if http_port is None:
                cls.__logger.info(f"Using HTTP port from base URL: {match.group(1)}")
                http_port = int(match.group(1))
            else:
                cls.__logger.info(f"Using HTTP port passed as an argument: {http_port}")

        # Capture the connection args for debug BEFORE flipping __initialized:
        # once initialization is marked done, span creation is live and a span
        # carrying a propagated debug block can arm the debug runtime (via
        # _arm_debug_runtime_from_context) in the window before _init_debug_runtime
        # runs below. That from-context path reads these static fields to build its
        # own cache clients, so leaving them None here would point the clients at
        # the default base URL instead of this initialize()'s base_url/http_port —
        # and first-wins would then pin that mis-targeted runtime. Set
        # unconditionally (even when local debug is off): a downstream span may
        # still arrive carrying a debug block. _init_debug_runtime re-sets these
        # (harmless) so its direct-call test path keeps working.
        cls.__base_url_for_debug = url
        cls.__http_port_for_debug = http_port

        cls.__initialized = True
        cls.__base_http_url = f"{http_url}:{http_port or 443}"
        env_metadata: dict[str, Any] = {}
        if env_metadata_str := os.getenv("LMNR_TRACE_METADATA"):
            try:
                env_metadata = json.loads(env_metadata_str)
            except Exception:
                pass
        cls.__global_metadata = {**env_metadata, **(metadata or {})}

        if not os.getenv("OTEL_ATTRIBUTE_COUNT_LIMIT"):
            # each message is at least 2 attributes: role and content,
            # but the default attribute limit is 128, so raise it
            os.environ["OTEL_ATTRIBUTE_COUNT_LIMIT"] = "10000"

        TracerManager.init(
            base_url=url,
            http_port=http_port or 443,
            port=grpc_port or 8443,
            project_api_key=cls.__project_api_key,
            instruments=set(instruments) if instruments is not None else None,
            block_instruments=(
                set(disabled_instruments) if disabled_instruments is not None else None
            ),
            disable_batch=disable_batch,
            max_export_batch_size=max_export_batch_size,
            timeout_seconds=export_timeout_seconds,
            set_global_tracer_provider=set_global_tracer_provider,
            otel_logger_level=otel_logger_level,
            session_recording_options=session_recording_options,
            force_http=force_http,
        )

        # Build the debug runtime only after tracing is up. It has no dependency
        # on TracerManager.init (which never reads the runtime or global
        # metadata), so running it here means a tracer-init failure aborts before
        # any debug side effects — backend session registration, the
        # `rollout.session_id` stamp, the atexit pointer hook — instead of
        # leaving them live on a process whose tracing never came up. It must
        # still precede the metadata-context attach below so `rollout.session_id`
        # lands on the OTEL context, and `_initialize_context_from_env` so the
        # runtime is registered before the inherited trace id is recorded.
        cls._init_debug_runtime(base_url=url, http_port=http_port)

        with get_tracer_with_context() as (tracer, isolated_context):
            new_ctx = context_api.set_value(
                CONTEXT_METADATA_KEY, cls.__global_metadata, isolated_context
            )
            attach_context(new_ctx)

        cls._initialize_context_from_env()

    @classmethod
    def _initialize_context_from_env(cls) -> None:
        """Attach upstream Laminar context from the environment, if provided."""
        env_context = os.getenv("LMNR_SPAN_CONTEXT")
        if not env_context:
            return

        try:
            laminar_context = LaminarSpanContext.deserialize(env_context)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            cls.__logger.warning(
                "LMNR_SPAN_CONTEXT is set but could not be deserialized: %s", exc
            )
            return

        # Arm the debug runtime from a debug block carried by LMNR_SPAN_CONTEXT
        # (first-wins, idempotent, no-op when already armed or no block present).
        # The span-creation funnels only see this block when a caller passes the
        # parent context explicitly; an LMNR_SPAN_CONTEXT-attached run instead
        # parents off the pushed context with `parent_span_context=None`, so
        # without arming here a propagated debug block would never activate the
        # replay cache / `rollout.session_id` on this downstream process. Done
        # before recording the inherited trace id so the runtime exists for it.
        cls._arm_debug_runtime_from_context(laminar_context.debug)

        try:
            otel_span_context = LaminarSpanContext.try_to_otel_span_context(
                laminar_context, cls.__logger
            )
        except ValueError as exc:
            cls.__logger.warning(
                "LMNR_SPAN_CONTEXT is set but invalid span context provided: %s", exc
            )
            return

        base_context = trace.set_span_in_context(
            trace.NonRecordingSpan(otel_span_context), get_current_context()
        )
        # Re-stamp global metadata onto the pushed context. A debug block in
        # LMNR_SPAN_CONTEXT was armed above, AFTER initialize() attached the
        # ambient metadata context, so `rollout.session_id` is now in
        # __global_metadata but not on the context auto-instrumentation spans
        # descend from. Refresh it here (idempotent for non-debug runs — the
        # value already on the context is the same global metadata).
        base_context = context_api.set_value(
            CONTEXT_METADATA_KEY, cls.__global_metadata, base_context
        )
        processor = TracerWrapper.instance._span_processor
        if isinstance(processor, LaminarSpanProcessor):
            processor.set_parent_path_info(
                otel_span_context.span_id,
                laminar_context.span_path,
                laminar_context.span_ids_path,
            )
        push_span_context(base_context)
        cls.__logger.debug("Initialized Laminar parent context from LMNR_SPAN_CONTEXT.")

        # On a debug run attached via LMNR_SPAN_CONTEXT, no span has a null parent
        # (everything descends from the injected context), so the processor's
        # root-span hook never fires. Record the inherited trace id here so the
        # run pointer (§5) isn't emitted with an empty trace_id.
        cls._record_debug_trace_id_from_env(otel_span_context)

    @classmethod
    def _record_debug_trace_id_from_env(
        cls, otel_span_context: trace.SpanContext
    ) -> None:
        """Record the trace id inherited from LMNR_SPAN_CONTEXT on the debug runtime.

        No-op when debug mode is off. Best-effort: never break initialization.
        """
        try:
            from lmnr.sdk.debug import get_runtime

            runtime = get_runtime()
            if runtime is None:
                return
            runtime.record_trace_id(str(uuid.UUID(int=otel_span_context.trace_id)))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            cls.__logger.debug("Failed to record debug trace id from env: %s", exc)

    @classmethod
    def _init_debug_runtime(cls, base_url: str | None, http_port: int | None) -> None:
        """Build the v2 debug runtime (shared spec §4) when LMNR_DEBUG is set.

        On a debug run the session id from the config is stamped into the global
        trace metadata as `rollout.session_id`, and a process-exit hook emits the
        run pointer once the root trace id is known. When debug mode is off this
        is a no-op and the SDK behaves exactly as before.

        Unlike v1 (which closed the client at the end of init), v2 retains BOTH a
        sync `LaminarClient` and an async `AsyncLaminarClient` on the runtime for
        the whole run — every live LLM call looks its input hash up in the
        server-side cache through them (sync providers via `client`, async
        providers via `async_client`). `shutdown()` closes them.
        """
        # Capture the connection args so a later context-armed runtime (built
        # deep in span creation, with no access to initialize()'s args) can
        # construct its own clients. Set unconditionally — even when local debug
        # is off, a downstream span may still arrive carrying a debug block.
        cls.__base_url_for_debug = base_url
        cls.__http_port_for_debug = http_port

        try:
            from lmnr.sdk.debug import (
                get_runtime,
                init_debug_runtime,
                reset_debug_runtime,
            )
            from lmnr.sdk.debug.config import _is_truthy

            # Debug mode off: bail before constructing the clients (and their
            # httpx pools), which would otherwise leak unclosed on every normal
            # initialize(). This is the same LMNR_DEBUG gate build_debug_config()
            # applies first; checking it directly avoids a redundant config build
            # here (which would mint a throwaway session uuid and re-read the
            # last-run file) that init_debug_runtime() then discards and rebuilds.
            if not _is_truthy(os.environ.get("LMNR_DEBUG")):
                return

            # LMNR_DEBUG is set, so env config owns this process. But initialize()
            # flips __initialized BEFORE calling this method, and the span funnels
            # gate only on is_initialized() — so in that window a span carrying a
            # propagated debug block could have armed a context runtime
            # (local_origin=False). init_debug_runtime() is idempotent and would
            # then return THAT runtime, whose clients differ from the env ones
            # built below, so the `runtime.client is not client` guard would bail —
            # skipping session registration, the browser open, and the pointer
            # hook the local run needs. Discard the context runtime (closing its
            # now-orphaned clients) and clear the one-shot flag so the env path
            # below builds a fresh local-origin runtime that owns the process.
            preempting = get_runtime()
            if preempting is not None and not preempting.local_origin:
                preempting.client.close()
                cls._close_debug_async_client(preempting.async_client)
                reset_debug_runtime()

            from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
            from lmnr.sdk.client.synchronous.sync_client import LaminarClient
            from lmnr.sdk.utils import get_frontend_url

            # Retained for the run's lifetime (NOT closed here) — the provider
            # wrappers reach back through the runtime to hit the cache endpoint
            # on every live call. shutdown() closes both.
            client = LaminarClient(
                base_url=base_url,
                project_api_key=cls.__project_api_key,
                port=http_port,
            )
            async_client = AsyncLaminarClient(
                base_url=base_url,
                project_api_key=cls.__project_api_key,
                port=http_port,
            )
            debugger_url = os.getenv("LMNR_FRONTEND_URL") or get_frontend_url(base_url)
            runtime = init_debug_runtime(
                client, async_client, debugger_url=debugger_url
            )
            if runtime is None or runtime.client is not client:
                # `runtime is None`: debug mode is off after all
                # (build_debug_config returned None). `runtime.client is not
                # client`: a propagated context armed the runtime in the narrow
                # window between the discard above and this build, so ours are
                # orphaned. The discard above clears the COMMON preempt case so
                # the local run still owns the process; this stays as a defensive
                # fallback for that residual race — close ours either way, since
                # the existing runtime retains ITS clients and leaving ours open
                # leaks httpx pools.
                client.close()
                cls._close_debug_async_client(async_client)
                return

            # Register the SDK-minted session id with the backend so the run
            # shows up in the UI. This idempotent upsert is what makes a bare
            # `LMNR_DEBUG=true` run (no replay) useful. Best-effort: a failure
            # here must never crash initialization, so it stays inside the
            # surrounding try/except. The backend returns the project id
            # (derived from the API key) so we can print the session URL.
            try:
                project_id = client.rollout_sessions.register(runtime.session_id)
                if project_id:
                    # Record the project id so the run pointer's debugger_url
                    # field carries the SAME full per-session URL we print
                    # here (single code path via debugger_session_url).
                    runtime.record_project_id(project_id)
                    session_url = runtime.debugger_session_url()
                    cls.__logger.info(
                        "Laminar debugger session: %s",
                        session_url,
                    )
                    if runtime.should_open_browser:
                        opener = (
                            "open"
                            if sys.platform == "darwin"
                            else "start"
                            if sys.platform == "win32"
                            else "xdg-open"
                        )
                        subprocess.Popen(
                            [opener, session_url],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
            except Exception as exc:
                cls.__logger.warning("Failed to register debug session: %s", exc)

            cls.__global_metadata = {
                **cls.__global_metadata,
                "rollout.session_id": runtime.session_id,
            }
            # Drop any prior hook first so a repeated initialize() (or an
            # init/shutdown loop) doesn't pin every retired DebugRuntime (and
            # its replay cache) alive via atexit's strong reference; keep this
            # one by reference for shutdown() to unregister.
            if cls.__debug_exit_hook is not None:
                atexit.unregister(cls.__debug_exit_hook)
            cls.__debug_exit_hook = runtime.emit_pointer
            atexit.register(cls.__debug_exit_hook)
        except Exception as exc:  # never let debug setup crash initialization
            cls.__logger.warning("Failed to initialize debug runtime: %s", exc)

    @classmethod
    def _arm_debug_runtime_from_context(cls, debug: Any) -> None:
        """Arm OR refresh the debug runtime from a propagated `DebugContext`.

        Called from the span-creation funnels when a parent `LaminarSpanContext`
        carrying an armed debug block parses — so a downstream service joins the
        upstream debug run regardless of how its spans originate
        (auto-instrumentation, manual observe, or an external library).

        The coordinates carried in the span context are DYNAMIC: a long-lived
        downstream service handling many requests must follow each request's
        session / replay-trace / cache-until, not freeze on the first context it
        ever saw. So the transport (the clients) is built once and reused, while
        the replay coordinates are refreshed in place on every new context. An
        env-origin runtime is the exception — it owns the process, so a
        propagated context never overrides it. A context-armed (downstream)
        runtime reuses the upstream session and may consult the cache, but —
        unlike the local-origin path — does NOT open the browser, print the
        session URL, or register an exit-time pointer hook (the origin owns
        those). It (re-)registers the session and re-stamps `rollout.session_id`
        only when the session id actually changes.

        Never raises: any failure leaves debug inert. Mirrors the TS
        `_armDebugRuntimeFromContext`.
        """
        try:
            from lmnr.sdk.debug import get_runtime

            # Pre-check the block before allocating clients, so a span carrying
            # no / an unarmed debug block costs nothing.
            if (
                debug is None
                or not getattr(debug, "enabled", False)
                or not getattr(debug, "session_id", None)
            ):
                return

            # An env-origin run owns the process and pins its own coordinates — a
            # propagated context never overrides it, so there is nothing to
            # refresh. Bail before any allocation.
            existing = get_runtime()
            if existing is not None and existing.local_origin:
                return

            from lmnr.sdk.debug import init_debug_runtime_from_context
            from lmnr.sdk.utils import get_frontend_url

            base_url = cls.__base_url_for_debug
            http_port = cls.__http_port_for_debug

            # Reuse the already-built clients when a context-armed runtime exists
            # — the transport is stable; only the dynamic coordinates (session /
            # replay / cache-until) move per request. Build clients only on first
            # arm so a long-lived downstream service doesn't allocate a pair per
            # span.
            if existing is not None:
                client = existing.client
                async_client = existing.async_client
                built_clients = False
            else:
                from lmnr.sdk.client.asynchronous.async_client import (
                    AsyncLaminarClient,
                )
                from lmnr.sdk.client.synchronous.sync_client import LaminarClient

                client = LaminarClient(
                    base_url=base_url,
                    project_api_key=cls.__project_api_key,
                    port=http_port,
                )
                async_client = AsyncLaminarClient(
                    base_url=base_url,
                    project_api_key=cls.__project_api_key,
                    port=http_port,
                )
                built_clients = True

            debugger_url = os.getenv("LMNR_FRONTEND_URL") or get_frontend_url(base_url)
            runtime, config_changed = init_debug_runtime_from_context(
                debug, client, async_client, debugger_url=debugger_url
            )
            if runtime is None or runtime.client is not client:
                # `runtime is None`: the block was unarmed. `runtime.client is
                # not client`: another caller won the first-arm race (or this is
                # a refresh that reused a pre-existing runtime's clients), so the
                # clients WE built here are orphaned. Close them only when we
                # built them — never close clients we borrowed from `existing`,
                # which the live runtime still uses. Leaving built ones open
                # leaks httpx pools.
                if built_clients:
                    client.close()
                    cls._close_debug_async_client(async_client)
                if runtime is None:
                    return

            # Only (re-)register + re-stamp when the propagated context describes
            # a fresh run — ANY moved coordinate (session id, replay trace id, or
            # cache-until needle), not just a new session id. A steady stream of
            # requests carrying the SAME coordinates must not spam the backend or
            # rewrite global metadata on every span. Register the (upstream)
            # session id so downstream cache lookups are accepted: idempotent
            # upsert, best-effort. Unlike the local-origin path we do NOT log the
            # URL or open a browser — the origin already did.
            if not config_changed:
                return

            # New replay coordinates mean a new run's cache state: clear the
            # process-wide run-live latch so the new run starts clean. Otherwise a
            # MISS latched by the PREVIOUS run would make every call in the new
            # one skip the cache and run live.
            cls.set_debug_run_live(False)

            try:
                runtime.client.rollout_sessions.register(runtime.session_id)
            except Exception as exc:
                cls.__logger.debug(
                    "Failed to register downstream debug session: %s", exc
                )

            cls.__global_metadata = {
                **cls.__global_metadata,
                "rollout.session_id": runtime.session_id,
            }
            # Re-stamp global metadata onto the ambient isolated context.
            # `__global_metadata` alone is not enough: `LaminarSpanProcessor`
            # reads `rollout.session_id` from `CONTEXT_METADATA_KEY` on the
            # parent context, so auto-instrumented spans (no explicit
            # start_span funnel) would omit it on a downstream joined run.
            # Mirrors the LMNR_SPAN_CONTEXT / env-init paths in initialize().
            refreshed = context_api.set_value(
                CONTEXT_METADATA_KEY, cls.__global_metadata, get_current_context()
            )
            attach_context(refreshed)
            # No atexit pointer hook: a downstream run must not emit the pointer.
        except Exception as exc:  # never let debug arming crash span creation
            cls.__logger.debug("Failed to arm debug runtime from context: %s", exc)

    @staticmethod
    def _close_debug_async_client(async_client: Any) -> None:
        """Best-effort close of the retained async cache client from sync code.

        `AsyncLaminarClient.close()` is a coroutine, but both call sites (the
        `_init_debug_runtime` bail path and `shutdown()`) are synchronous, so we
        drive it to completion with `asyncio.run()`. Everything is swallowed: a
        failed close of the cache client must never crash init or shutdown. If a
        loop is already running on this thread (async caller), `asyncio.run`
        raises `RuntimeError` — there is no safe synchronous way to await on the
        live loop, so we leave the close to the loop's own teardown.
        """
        try:
            asyncio.run(async_client.close())
        except Exception:
            pass

    @classmethod
    def is_initialized(cls):
        """Check if Laminar is initialized. A utility to make sure other
        methods are called after initialization.

        Returns:
            bool: True if Laminar is initialized, False otherwise
        """
        return cls.__initialized

    @classmethod
    def _initialize_logger(cls):
        cls.__logger = logging.getLogger(__name__)
        cls.__logger.setLevel(get_level_from_env())
        console_log_handler = logging.StreamHandler()
        console_log_handler.setFormatter(VerboseColorfulFormatter())
        cls.__logger.addHandler(console_log_handler)

    @classmethod
    def event(
        cls,
        name: str,
        attributes: dict[str, AttributeValue] | None = None,
        timestamp: datetime.datetime | int | None = None,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
    ):
        """Associate an event with the current span. This is a wrapper around
        `span.add_event()` that adds the event to the current span.

        Args:
            name (str): event name
            attributes (dict[str, AttributeValue] | None, optional): event attributes.
                Defaults to None.
            timestamp (datetime.datetime | int | None, optional): If int, must\
                be epoch nanoseconds. If not specified, relies on the underlying\
                OpenTelemetry implementation. Defaults to None.
        """
        if not cls.is_initialized():
            return

        if timestamp and isinstance(timestamp, datetime.datetime):
            timestamp = int(timestamp.timestamp() * 1e9)

        extra_attributes = get_event_attributes_from_context()

        # override the user_id and session_id from the context with the ones
        # passed as arguments
        if user_id is not None:
            extra_attributes["lmnr.event.user_id"] = user_id
        if session_id is not None:
            extra_attributes["lmnr.event.session_id"] = session_id

        current_span = trace.get_current_span(context=get_current_context())
        if current_span == trace.INVALID_SPAN:
            span = cls.start_span(name)
            span.add_event(name, {**(attributes or {}), **extra_attributes}, timestamp)
            span.end()
            return

        current_span.add_event(
            name, {**(attributes or {}), **extra_attributes}, timestamp
        )

    @classmethod
    @contextmanager
    def start_as_current_span(
        cls,
        name: str,
        input: Any = None,
        span_type: LaminarSpanType = "DEFAULT",
        context: Context | None = None,
        labels: list[str] | None = None,
        parent_span_context: LaminarSpanContext | None = None,
        tags: list[str] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, AttributeValue] | None = None,
        attributes: dict[str, AttributeValue] | None = None,
    ) -> Generator[LaminarSpan, None, None]:
        """Start a new span as the current span. Useful for manual
        instrumentation. If `span_type` is set to `"LLM"`, you should report
        usage and response attributes manually. See `Laminar.set_span_attributes`
        for more information.

        Usage example:
        ```python
        with Laminar.start_as_current_span("my_span", input="my_input") as span:
            await my_async_function()
            Laminar.set_span_output("my_output")`
        ```

        Args:
            name (str): name of the span
            input (Any, optional): input to the span. Will be sent as an\
                attribute, so must be json serializable. Defaults to None.
            span_type (Literal["DEFAULT", "LLM", "TOOL"], optional):\
                type of the span. If you use `"LLM"`, you should report usage\
                and response attributes manually. Defaults to "DEFAULT".
            context (Context | None, optional): raw OpenTelemetry context\
                to attach the span to. Defaults to None.
            parent_span_context (LaminarSpanContext | None, optional): parent\
                span context to use for the span. Useful for continuing traces\
                across services. If parent_span_context is a\
                raw OpenTelemetry span context, or if it is a dictionary or string\
                obtained from `Laminar.get_laminar_span_context_dict()` or\
                `Laminar.get_laminar_span_context_str()` respectively, it will be\
                converted to a `LaminarSpanContext` if possible. See also\
                `Laminar.serialize_span_context` for more information.
                Defaults to None.
            labels (list[str] | None, optional): [DEPRECATED] Use tags\
                instead. Labels to set for the span. Defaults to None.
            tags (list[str] | None, optional): tags to set for the span.
                Defaults to None.
            user_id (str | None, optional): user id to set for the trace.
                Defaults to None.
            session_id (str | None, optional): session id to set for the trace.
                Defaults to None.
            metadata (dict[str, AttributeValue] | None, optional): metadata to\
                set for the trace. Defaults to None.
            attributes (dict[str, AttributeValue] | None, optional): attributes to\
                set for the span. This function may override attributes by laminar\
                internal values, such as tags or metadata. Defaults to None.
        """

        if not cls.is_initialized():
            yield trace.NonRecordingSpan(
                trace.SpanContext(
                    trace_id=RandomIdGenerator().generate_trace_id(),
                    span_id=RandomIdGenerator().generate_span_id(),
                    is_remote=False,
                )
            )
            return

        with get_tracer_with_context() as (tracer, isolated_context):
            ctx = context or isolated_context

            # Parse parent_span_context and extract all info
            parsed = _parse_parent_span_context(parent_span_context, cls.__logger)

            # Arm the debug runtime from a propagated debug block (first-wins,
            # idempotent, no-op when already armed or no block present). Placed
            # here — deep in span creation, before any caching instrumentation
            # consults the runtime — so a downstream run joins the upstream debug
            # session regardless of how its spans originate.
            cls._arm_debug_runtime_from_context(parsed["debug"])

            # Arming a new session attaches a refreshed isolated context carrying
            # the newly-armed `rollout.session_id`. When we're building on that
            # isolated context (no explicit `context` arg), re-read it so the
            # metadata merge below reflects the armed session — `ctx` was snapshot
            # BEFORE arming, and since context wins over global in the merge, a
            # prior request's stale session id would otherwise override it.
            if context is None:
                ctx = get_current_context()

            # Set parent span in context if present
            if parsed["otel_span_context"] is not None:
                ctx = trace.set_span_in_context(
                    trace.NonRecordingSpan(parsed["otel_span_context"]), ctx
                )

            # Determine trace_type with proper priority
            trace_type = None
            if span_type in ["EVALUATION", "EXECUTOR", "EVALUATOR"]:
                trace_type = TraceType.EVALUATION
            elif parsed["trace_type"] is not None:
                trace_type = parsed["trace_type"]

            # Merge metadata: context (inherited) + global + parent + explicit (explicit wins)
            # Get metadata from context if it exists
            ctx_metadata = get_value(CONTEXT_METADATA_KEY, ctx) or {}
            # Merge with priority: global < context < parent < explicit
            merged_metadata = {
                **(cls.__global_metadata or {}),
                **(ctx_metadata or {}),
                **(parsed["metadata"] or {}),
                **(metadata or {}),
            }

            # Get association props from context (fallback values)
            ctx_user_id = get_value(CONTEXT_USER_ID_KEY, ctx)
            ctx_session_id = get_value(CONTEXT_SESSION_ID_KEY, ctx)

            # Merge user_id and session_id with priority: context < parent < explicit
            final_user_id = (
                user_id
                if user_id is not None
                else (
                    parsed["user_id"] if parsed["user_id"] is not None else ctx_user_id
                )
            )
            final_session_id = (
                session_id
                if session_id is not None
                else (
                    parsed["session_id"]
                    if parsed["session_id"] is not None
                    else ctx_session_id
                )
            )

            ctx = set_association_prop_context(
                trace_type=trace_type,
                user_id=final_user_id,
                session_id=final_session_id,
                metadata=merged_metadata if merged_metadata else None,
                context=ctx,
                # we need a token separately, so we manually attach the context
                attach=False,
            )
            ctx_token = context_api.attach(ctx)
            isolated_context_token = attach_context(ctx)
            label_props = {}
            try:
                if labels:
                    warnings.warn(
                        "`Laminar.start_as_current_span` `labels` is deprecated. Use `tags` instead.",
                        DeprecationWarning,
                    )
                    label_props = {f"{ASSOCIATION_PROPERTIES}.labels": labels}
            except Exception:
                cls.__logger.warning(
                    f"`start_as_current_span` Could not set labels: {labels}. "
                    "They will be propagated to the next span."
                )
            tag_props = {}
            if tags:
                if isinstance(tags, list) and all(isinstance(tag, str) for tag in tags):
                    tag_props = {f"{ASSOCIATION_PROPERTIES}.tags": tags}
                else:
                    cls.__logger.warning(
                        f"`start_as_current_span` Could not set tags: {tags}. Tags must be a list of strings. "
                        "Tags will be ignored."
                    )

            try:
                with tracer.start_as_current_span(
                    name,
                    context=ctx,
                    attributes={
                        **(attributes or {}),
                        SPAN_TYPE: span_type,
                        PARENT_SPAN_PATH: parsed["path"],
                        PARENT_SPAN_IDS_PATH: parsed["span_ids_path"],
                        **(label_props),
                        **(tag_props),
                        # Association properties are attached to context above
                        # and the relevant attributes are populated in the processor
                    },
                ) as span:
                    if not isinstance(span, LaminarSpan):
                        span = LaminarSpan(span)
                    span.set_input(input)
                    yield span
            finally:
                try:
                    detach_context(isolated_context_token)
                    context_api.detach(ctx_token)
                except Exception:
                    pass

    @classmethod
    def start_span(
        cls,
        name: str,
        input: Any = None,
        span_type: LaminarSpanType = "DEFAULT",
        context: Context | None = None,
        parent_span_context: LaminarSpanContext | None = None,
        labels: dict[str, str] | None = None,
        tags: list[str] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, AttributeValue] | None = None,
        attributes: dict[str, AttributeValue] | None = None,
    ) -> LaminarSpan | Span:
        """Start a new span. Useful for manual instrumentation.
        If `span_type` is set to `"LLM"`, you should report usage and response
        attributes manually. See `Laminar.set_span_attributes` for more
        information.

        Note that spans started with this method must be ended manually.
        In addition, they must be ended in LIFO order, e.g.
        span1 = Laminar.start_span("span1")
        span2 = Laminar.start_span("span2")
        span2.end()
        span1.end()
        Otherwise, the behavior is undefined.

        Usage example:
        ```python
        from src.lmnr import Laminar
        def foo(span):
            with Laminar.use_span(span):
                with Laminar.start_as_current_span("foo_inner"):
                    some_function()

        def bar():
            with Laminar.use_span(span):
                openai_client.chat.completions.create()

        span = Laminar.start_span("outer")
        foo(span)
        bar(span)
        # IMPORTANT: End the span manually
        span.end()

        # Results in:
        # | outer
        # |   | foo
        # |   |   | foo_inner
        # |   | bar
        # |   |   | openai.chat
        ```

        Args:
            name (str): name of the span
            input (Any, optional): input to the span. Will be sent as an\
                attribute, so must be json serializable. Defaults to None.
            span_type (Literal["DEFAULT", "LLM", "TOOL"], optional):\
                type of the span. If you use `"LLM"`, you should report usage\
                and response attributes manually. Defaults to "DEFAULT".
            context (Context | None, optional): raw OpenTelemetry context\
                to attach the span to. Defaults to None.
            parent_span_context (LaminarSpanContext | None, optional): parent\
                span context to use for the span. Useful for continuing traces\
                across services. If parent_span_context is a\
                raw OpenTelemetry span context, or if it is a dictionary or string\
                obtained from `Laminar.get_laminar_span_context_dict()` or\
                `Laminar.get_laminar_span_context_str()` respectively, it will be\
                converted to a `LaminarSpanContext` if possible. See also\
                `Laminar.get_span_context`, `Laminar.get_span_context_dict` and\
                `Laminar.get_span_context_str` for more information.
                Defaults to None.
            tags (list[str] | None, optional): tags to set for the span.
                Defaults to None.
            labels (dict[str, str] | None, optional): [DEPRECATED] Use tags\
                instead. Labels to set for the span. Defaults to None.
            user_id (str | None, optional): user id to set for the trace.
                Defaults to None.
            session_id (str | None, optional): session id to set for the trace.
                Defaults to None.
            metadata (dict[str, AttributeValue] | None, optional): metadata to\
                set for the trace. Defaults to None.
            attributes (dict[str, AttributeValue] | None, optional): attributes to\
                set for the span. This function may override attributes by laminar\
                internal values, such as tags or metadata. Defaults to None.
        """
        if not cls.is_initialized():
            trace_id = 0
            span_id = 0
            try:
                trace_id = RandomIdGenerator().generate_trace_id()
                span_id = RandomIdGenerator().generate_span_id()
            except Exception:
                pass
            return trace.NonRecordingSpan(
                trace.SpanContext(
                    trace_id=trace_id,
                    span_id=span_id,
                    is_remote=False,
                )
            )

        with get_tracer_with_context() as (tracer, isolated_context):
            ctx = context or isolated_context

            # Parse parent_span_context and extract all info
            parsed = _parse_parent_span_context(parent_span_context, cls.__logger)

            # Arm the debug runtime from a propagated debug block (first-wins,
            # idempotent, no-op when already armed or no block present). See the
            # matching call in start_as_current_span for the rationale.
            cls._arm_debug_runtime_from_context(parsed["debug"])

            # Re-read the isolated context after arming so the metadata merge
            # below reflects a newly-armed `rollout.session_id`. See the matching
            # comment in start_as_current_span — `ctx` was snapshot before arming
            # and context wins over global in the merge.
            if context is None:
                ctx = get_current_context()

            # Set parent span in context if present
            if parsed["otel_span_context"] is not None:
                ctx = trace.set_span_in_context(
                    trace.NonRecordingSpan(parsed["otel_span_context"]), ctx
                )

            # Get association props from context (fallback values)
            ctx_user_id = get_value(CONTEXT_USER_ID_KEY, ctx)
            ctx_session_id = get_value(CONTEXT_SESSION_ID_KEY, ctx)
            ctx_metadata = get_value(CONTEXT_METADATA_KEY, ctx)

            label_props = {}
            try:
                if labels:
                    warnings.warn(
                        "`Laminar.start_span` `labels` is deprecated. Use `tags` instead.",
                        DeprecationWarning,
                    )
                    label_props = {
                        f"{ASSOCIATION_PROPERTIES}.labels": json_dumps(labels)
                    }
            except Exception:
                cls.__logger.warning(
                    f"`start_span` Could not set labels: {labels}. They will be "
                    "propagated to the next span."
                )
            tag_props = {}
            if tags:
                if isinstance(tags, list) and all(isinstance(tag, str) for tag in tags):
                    tag_props = {f"{ASSOCIATION_PROPERTIES}.tags": tags}
                else:
                    cls.__logger.warning(
                        f"`start_span` Could not set tags: {tags}. Tags must be a list of strings. "
                        + "Tags will be ignored."
                    )

            # Determine trace_type with proper priority: explicit > parent > context
            trace_type = None
            if span_type in ["EVALUATION", "EXECUTOR", "EVALUATOR"]:
                trace_type = TraceType.EVALUATION
            elif parsed.get("trace_type") is not None:
                trace_type = parsed.get("trace_type")
            else:
                # Get trace_type from context if not set explicitly or from parent
                ctx_trace_type = get_value(CONTEXT_TRACE_TYPE_KEY, ctx)
                if ctx_trace_type:
                    try:
                        trace_type = TraceType(ctx_trace_type)
                    except (ValueError, TypeError):
                        pass

            # Merge with priority: global < context < parent < explicit
            merged_metadata = {
                **(cls.__global_metadata or {}),
                **(ctx_metadata or {}),
                **(parsed.get("metadata") or {}),
                **(metadata or {}),
            }

            # Merge user_id and session_id with priority: context < parent < explicit
            final_user_id = (
                user_id
                if user_id is not None
                else (
                    parsed["user_id"] if parsed["user_id"] is not None else ctx_user_id
                )
            )
            final_session_id = (
                session_id
                if session_id is not None
                else (
                    parsed["session_id"]
                    if parsed["session_id"] is not None
                    else ctx_session_id
                )
            )

            # Build association_props using merged values
            association_props = cls._get_association_prop_attributes(
                user_id=final_user_id,
                session_id=final_session_id,
                metadata=merged_metadata if merged_metadata else None,
                trace_type=trace_type,
            )

            span = tracer.start_span(
                name,
                context=ctx,
                attributes={
                    **(attributes or {}),
                    SPAN_TYPE: span_type,
                    PARENT_SPAN_PATH: parsed["path"],
                    PARENT_SPAN_IDS_PATH: parsed["span_ids_path"],
                    **(label_props),
                    **(tag_props),
                    **(association_props),
                },
            )

            if not isinstance(span, LaminarSpan):
                span = LaminarSpan(span)
            span.set_input(input)
            return span

    @classmethod
    @contextmanager
    def use_span(
        cls,
        span: Span,
        end_on_exit: bool = False,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> Generator[LaminarSpan | Span, None, None]:
        """Use a span as the current span. Useful for manual instrumentation.

        Fully copies the implementation of `use_span` from opentelemetry.trace
        and replaces the context API with Laminar's isolated context.

        Args:
            span: The span that should be activated in the current context.
            end_on_exit: Whether to end the span automatically when leaving the
                context manager scope.
            record_exception: Whether to record any exceptions raised within the
                context as error event on the span.
            set_status_on_exception: Only relevant if the returned span is used
                in a with/context manager. Defines whether the span status will
                be automatically set to ERROR when an uncaught exception is
                raised in the span with block. The span status won't be set by
                this mechanism if it was previously set manually.
        """
        if not cls.is_initialized():
            with use_span(
                span, end_on_exit, record_exception, set_status_on_exception
            ) as s:
                yield s
            return

        wrapper = TracerWrapper()

        try:
            # Set association props in context before push_span_context
            # so child spans inherit them
            assoc_props_token = set_association_props_in_context(span)
            if assoc_props_token and isinstance(span, LaminarSpan):
                span._lmnr_assoc_props_token = assoc_props_token

            context = wrapper.push_span_context(span)
            # Some auto-instrumentations are not under our control, so they
            # don't have access to our isolated context. We attach the context
            # to the OTEL global context, so that spans know their parent
            # span and trace_id.
            isolated_context_token = attach_context(context)
            context_token = context_api.attach(context)
            if isinstance(span, LaminarSpan):
                yield span
            else:
                yield LaminarSpan(span)

        # Record only exceptions that inherit Exception class but not BaseException, because
        # classes that directly inherit BaseException are not technically errors, e.g. GeneratorExit.
        # See https://github.com/open-telemetry/opentelemetry-python/issues/4484
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if isinstance(span, Span) and span.is_recording():
                # Record the exception as an event
                if record_exception:
                    span.record_exception(
                        exc, attributes=get_event_attributes_from_context()
                    )

                # Set status in case exception was raised
                if set_status_on_exception:
                    span.set_status(
                        Status(
                            status_code=StatusCode.ERROR,
                            description=f"{type(exc).__name__}: {exc}",
                        )
                    )

            # This causes parent spans to set their status to ERROR and to record
            # an exception as an event if a child span raises an exception even if
            # such child span was started with both record_exception and
            # set_status_on_exception attributes set to False.
            raise

        finally:
            try:
                context_api.detach(context_token)
                detach_context(isolated_context_token)
                wrapper.pop_span_context()
            finally:
                if end_on_exit:
                    span.end()

    @classmethod
    @contextmanager
    def use_span_context(
        cls,
        parent_span_context: LaminarSpanContext | dict | str,
    ) -> Generator[None, None, None]:
        """Activate a remote `LaminarSpanContext` as the parent for spans created
        inside this block, WITHOUT creating a span of its own.

        Everything started inside the `with` block — `@observe` spans, auto-
        instrumented LLM spans, etc. — parents off the provided context, so it
        nests under the upstream trace. This is the span-less counterpart to
        ``start_span(parent_span_context=...)``: use it when you want the caller's
        own spans to inherit the remote trace but don't want an extra wrapper span
        in between (e.g. a Temporal activity with `create_activity_span=False`).

        Like the span funnels, this also arms the downstream debug runtime from a
        nested ``debug`` block in the context and registers the parent path info
        so child span paths are correct.
        """
        if not cls.is_initialized():
            yield
            return

        parsed = _parse_parent_span_context(parent_span_context, cls.__logger)

        # Arm the debug runtime from a propagated debug block (first-wins,
        # idempotent), so a debug run flows through even when no span is created.
        cls._arm_debug_runtime_from_context(parsed["debug"])

        if parsed["otel_span_context"] is None:
            yield
            return

        # Re-read the isolated context AFTER arming so the merge below picks up a
        # freshly-stamped `rollout.session_id`, then set the remote parent on it.
        ctx = trace.set_span_in_context(
            trace.NonRecordingSpan(parsed["otel_span_context"]),
            get_current_context(),
        )

        # Merge association props the same way `start_span` does
        # (global < context < parent) instead of overwriting. A plain
        # `set_association_prop_context(metadata=parent_metadata)` would
        # `set_value` over `CONTEXT_METADATA_KEY` wholesale and drop the global /
        # ambient metadata — most importantly the just-armed `rollout.session_id`.
        ctx_user_id = get_value(CONTEXT_USER_ID_KEY, ctx)
        ctx_session_id = get_value(CONTEXT_SESSION_ID_KEY, ctx)
        ctx_metadata = get_value(CONTEXT_METADATA_KEY, ctx)

        merged_metadata = {
            **(cls.__global_metadata or {}),
            **(ctx_metadata or {}),
            **(parsed["metadata"] or {}),
        }
        final_user_id = (
            parsed["user_id"] if parsed["user_id"] is not None else ctx_user_id
        )
        final_session_id = (
            parsed["session_id"]
            if parsed["session_id"] is not None
            else ctx_session_id
        )

        ctx = set_association_prop_context(
            trace_type=parsed["trace_type"],
            user_id=final_user_id,
            session_id=final_session_id,
            metadata=merged_metadata or None,
            context=ctx,
            attach=False,
        )

        # Register the parent path so child spans build correct dotted paths,
        # mirroring the LMNR_SPAN_CONTEXT env-init path.
        processor = TracerWrapper.instance._span_processor
        if isinstance(processor, LaminarSpanProcessor):
            processor.set_parent_path_info(
                parsed["otel_span_context"].span_id,
                parsed["path"],
                parsed["span_ids_path"],
            )

        ctx_token = context_api.attach(ctx)
        isolated_context_token = attach_context(ctx)
        try:
            yield
        finally:
            detach_context(isolated_context_token)
            context_api.detach(ctx_token)

    @classmethod
    def start_active_span(
        cls,
        name: str,
        input: Any = None,
        span_type: Literal["DEFAULT", "LLM", "TOOL"] = "DEFAULT",
        context: Context | None = None,
        parent_span_context: LaminarSpanContext | None = None,
        tags: list[str] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, AttributeValue] | None = None,
        attributes: dict[str, AttributeValue] | None = None,
    ) -> LaminarSpan | Span:
        """Start a span and mark it as active within the current context.
        All spans started after this one will be children of this span.
        Useful for manual instrumentation. Must be ended manually.
        If `span_type` is set to `"LLM"`, you should report usage and response
        attributes manually. See `Laminar.set_span_attributes` for more
        information. Returns the span object.

        Note that ending the started span in a different async context yields
        unexpected results. When propagating spans across different async or
        threading contexts, it is recommended to either:
        - Make sure to start and end the span in the same async context or thread, or
        - Use `Laminar.start_span` + `Laminar.use_span` where possible.

        Note that spans started with this method must be ended manually.
        In addition, they must be ended in LIFO order, e.g.
        span1 = Laminar.start_active_span("span1")
        span2 = Laminar.start_active_span("span2")
        span2.end()
        span1.end()
        Otherwise, the behavior is undefined.

        Usage example:
        ```python
        from src.lmnr import Laminar, observe

        @observe()
        def foo():
            with Laminar.start_as_current_span("foo_inner"):
                some_function()

        @observe()
        def bar():
            openai_client.chat.completions.create()

        span = Laminar.start_active_span("outer")
        foo()
        bar()
        # IMPORTANT: End the span manually
        span.end()

        # Results in:
        # | outer
        # |   | foo
        # |   |   | foo_inner
        # |   | bar
        # |   |   | openai.chat
        ```

        Args:
            name (str): name of the span
            input (Any, optional): input to the span. Will be sent as an\
                attribute, so must be json serializable. Defaults to None.
            span_type (Literal["DEFAULT", "LLM", "TOOL"], optional):\
                type of the span. If you use `"LLM"`, you should report usage\
                and response attributes manually. Defaults to "DEFAULT".
            context (Context | None, optional): raw OpenTelemetry context\
                to attach the span to. Defaults to None.
            parent_span_context (LaminarSpanContext | None, optional): parent\
                span context to use for the span. Useful for continuing traces\
                across services. If parent_span_context is a\
                raw OpenTelemetry span context, or if it is a dictionary or string\
                obtained from `Laminar.get_laminar_span_context_dict()` or\
                `Laminar.get_laminar_span_context_str()` respectively, it will be\
                converted to a `LaminarSpanContext` if possible. See also\
                `Laminar.get_span_context`, `Laminar.get_span_context_dict` and\
                `Laminar.get_span_context_str` for more information.
                Defaults to None.
            tags (list[str] | None, optional): tags to set for the span.
                Defaults to None.
            user_id (str | None, optional): user id to set for the trace.
                Defaults to None.
            session_id (str | None, optional): session id to set for the trace.
                Defaults to None.
            metadata (dict[str, AttributeValue] | None, optional): metadata to\
                set for the trace. Defaults to None.
            attributes (dict[str, AttributeValue] | None, optional): attributes to\
                set for the span. This function may override attributes by laminar\
                internal values, such as tags or metadata. Defaults to None.
        """
        span = cls.start_span(
            name=name,
            input=input,
            span_type=span_type,
            context=context,
            parent_span_context=parent_span_context,
            tags=tags,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
            attributes=attributes,
        )
        if not cls.is_initialized():
            return span
        wrapper = TracerWrapper()

        # Set association props in context before push_span_context
        # so child spans inherit them
        assoc_props_token = set_association_props_in_context(span)
        if assoc_props_token and isinstance(span, LaminarSpan):
            span._lmnr_assoc_props_token = assoc_props_token

        context = wrapper.push_span_context(span, from_ctx=context)
        context_token = context_api.attach(context)
        isolated_context_token = attach_context(context)
        span._lmnr_ctx_token = context_token
        span._lmnr_isolated_ctx_token = isolated_context_token
        try:
            current_task = asyncio.current_task()
        except Exception:
            current_task = None
        span._lmnr_task_id = id(current_task)
        if isinstance(span, LaminarSpan):
            return span
        else:
            return LaminarSpan(span)

    @classmethod
    def set_span_output(cls, output: Any = None):
        """Set the output of the current span. Useful for manual
        instrumentation.

        Args:
            output (Any, optional): output of the span. Will be sent as an\
                attribute, so must be json serializable. Defaults to None.
        """
        span = cls.get_current_span()
        if span is None:
            return
        span.set_output(output)

    @classmethod
    def set_span_attributes(
        cls,
        attributes: dict[Attributes | str, Any],
    ):
        """Set attributes for the current span. Useful for manual
        instrumentation.
        Example:
        ```python
        with Laminar.start_as_current_span(
            name="my_span_name", input=input["messages"], span_type="LLM"
        ):
            response = await my_custom_call_to_openai(input)
            Laminar.set_span_output(response["choices"][0]["message"]["content"])
            Laminar.set_span_attributes({
                Attributes.PROVIDER: 'openai',
                Attributes.REQUEST_MODEL: input["model"],
                Attributes.RESPONSE_MODEL: response["model"],
                Attributes.INPUT_TOKEN_COUNT: response["usage"]["prompt_tokens"],
                Attributes.OUTPUT_TOKEN_COUNT: response["usage"]["completion_tokens"],
            })
            # ...
        ```

        Args:
            attributes (dict[Attributes | str, Any]): attributes to set for the span
        """
        span = cls.get_current_span()
        if span == trace.INVALID_SPAN or span is None:
            return

        for key, value in attributes.items():
            if isinstance(key, Attributes):
                key = key.value
            if not is_otel_attribute_value_type(value):
                span.set_attribute(key, json_dumps(value))
            else:
                span.set_attribute(key, value)

    @classmethod
    def get_laminar_span_context(
        cls, span: trace.Span | None = None
    ) -> LaminarSpanContext | None:
        """Get the laminar span context for a given span.
        If no span is provided, the current active span will be used.
        """
        if not cls.is_initialized():
            return None

        span = span or cls.get_current_span()
        if span == trace.INVALID_SPAN or span is None:
            return None
        if not isinstance(span, LaminarSpan):
            span = LaminarSpan(span)
        return span.get_laminar_span_context()

    @classmethod
    def get_laminar_span_context_dict(
        cls, span: trace.Span | None = None
    ) -> dict | None:
        span_context = cls.get_laminar_span_context(span)
        if span_context is None:
            return None
        return span_context.model_dump()

    @classmethod
    def serialize_span_context(cls, span: trace.Span | None = None) -> str | None:
        """Get the laminar span context for a given span as a string.
        If no span is provided, the current active span will be used.

        This is useful for continuing a trace across services.

        Example:
        ```python
        # service A:
        with Laminar.start_as_current_span("service_a"):
            span_context = Laminar.serialize_span_context()
            # send span_context to service B
            call_service_b(request, headers={"laminar-span-context": span_context})

        # service B:
        def call_service_b(request, headers):
            span_context = Laminar.deserialize_span_context(headers["laminar-span-context"])
            with Laminar.start_as_current_span("service_b", parent_span_context=span_context):
                # rest of the function
                pass
        ```

        This will result in a trace like:
        ```
        service_a
          service_b
        ```
        """
        span_context = cls.get_laminar_span_context(span)
        if span_context is None:
            return None
        return str(span_context)

    @classmethod
    def deserialize_span_context(cls, span_context: dict | str) -> LaminarSpanContext:
        return LaminarSpanContext.deserialize(span_context)

    @classmethod
    def get_current_span(cls, context: Context | None = None) -> LaminarSpan | None:
        """Get the current active span. If a context is provided, the span will
        be retrieved from that context.

        Args:
            context (Context | None, optional): The context to get the span\
                from. If not provided, the current context will be used.
                Defaults to None.

        Returns:
            LaminarSpan | None: The current active span, or None if there is no\
                active span.
        """
        context = context or get_current_context()
        span = trace.get_current_span(context=context)
        if span == trace.INVALID_SPAN:
            return None
        if isinstance(span, LaminarSpan):
            return span
        else:
            return LaminarSpan(span)

    @classmethod
    def connect_to_langfuse(cls) -> bool:
        """Bridge the Langfuse Python SDK into Laminar so spans emitted by
        ``@observe``, ``langfuse.openai``, ``langfuse.langchain``, etc. are
        dual-exported to Laminar in addition to Langfuse.

        The Langfuse bridge is opt-in: it is NOT installed merely because
        ``langfuse`` is present. Install it either by passing an explicit
        ``instruments`` set containing ``Instruments.LANGFUSE`` to
        ``Laminar.initialize``, or by calling this method after initialization
        (e.g. when initialization happens in code you don't control).

        Returns:
            bool: True if the bridge was installed, False otherwise
            (e.g. Laminar not initialized, ``langfuse < 3.0`` or not
            importable).
        """
        # `cls.__logger` is only set by `_initialize_logger()` during
        # `initialize()`, so we cannot rely on it here — this method is
        # reachable before initialization. Fall back to a module-level
        # logger instead.
        logger = logging.getLogger(__name__)
        if not cls.is_initialized():
            logger.warning(
                "Laminar is not initialized. Call Laminar.initialize() first."
            )
            return False
        from lmnr.opentelemetry_lib.tracing.instruments import (
            _langfuse_installed,
        )

        # `_langfuse_installed` gates on both presence AND version >= 3.0 —
        # the bridge is OTel-native and does nothing useful on langfuse 2.x.
        # Going through `instrument()` on 2.x would install a useless
        # translator, flip `_installed=True`, and permanently block a later
        # valid install.
        if not _langfuse_installed():
            logger.warning(
                "`langfuse >= 3.0` is required for the Laminar/Langfuse "
                "bridge. Install it with `pip install 'langfuse>=3.0'`."
            )
            return False
        from lmnr.opentelemetry_lib.opentelemetry.instrumentation.langfuse import (
            LangfuseInstrumentor,
            langfuse_sdk_importable,
        )

        # `_langfuse_installed()` only reads install metadata; it never imports
        # the SDK. On Python 3.14 langfuse's pydantic-v1 models fail to build,
        # so the package is "present" but `import
        # langfuse._client.resource_manager` raises. The bridge's
        # resource-manager attach/patch path swallows that and silently no-ops,
        # which would leave the bridge `_installed=True` but inert — SDK spans
        # would never reach Laminar. Refuse to report success in that case.
        if not langfuse_sdk_importable():
            logger.warning(
                "`langfuse` is installed but cannot be imported in this "
                "interpreter (a known pydantic v1 incompatibility on Python "
                "3.14). The Laminar/Langfuse bridge would be inert, so it was "
                "not installed."
            )
            return False

        wrapper = TracerWrapper.instance
        if wrapper._tracer_provider is None:
            return False
        # `LangfuseInstrumentor.instrument()` re-raises after rollback if the
        # attach-to-existing / resource-manager-patch phase fails (e.g.
        # `RuntimeError` from concurrent modification of
        # `LangfuseResourceManager._instances`). This public helper is
        # documented to return `bool`, so swallow the exception and surface
        # the failure as False — `uninstrument()` has already cleaned up
        # partial state by the time we get here.
        try:
            LangfuseInstrumentor().instrument(
                lmnr_tracer_provider=wrapper._tracer_provider,
                lmnr_span_processor=wrapper._span_processor,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(
                "Failed to install Laminar/Langfuse bridge: %s", exc
            )
            return False
        return LangfuseInstrumentor._installed

    @classmethod
    def flush(cls) -> bool:
        """Flush the internal tracer.

        Returns:
            bool: True if the tracer was flushed, False otherwise
            (e.g. no tracer or timeout).
        """
        if not cls.is_initialized():
            return False
        return TracerManager.flush()

    @classmethod
    def force_flush(cls):
        """Force flush the internal tracer. WARNING: Any active spans are
        removed from context; that is, spans started afterwards will start
        a new trace.

        Actually shuts down the span processor and re-initializes it as long
        as it is a LaminarSpanProcessor. This is not recommended in production
        workflows, but is useful at the end of Lambda functions, where a regular
        flush might be killed by the Lambda runtime, because the actual export
        inside it runs in a background thread.
        """
        if not cls.is_initialized():
            return
        TracerManager.force_reinit_processor()

    @classmethod
    def shutdown(cls):
        if cls.is_initialized():
            # Emit the debug run pointer before shutting down tracing so flows
            # that shut down without terminating the process still get
            # LMNR_DEBUG_RUN + .lmnr/last-run.json. Idempotent — the atexit hook
            # is a fallback.
            from lmnr.sdk.debug import get_runtime, reset_debug_runtime

            runtime = get_runtime()
            if runtime is not None:
                # Best-effort: emit_pointer prints to stdout, which can raise
                # OSError/BrokenPipeError (closed stdout in daemons/containers,
                # notebook kernel restarts). That must never skip the cleanup
                # below — leaving exporter threads alive and __initialized=True.
                try:
                    runtime.emit_pointer()
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    cls.__logger.debug("Failed to emit debug run pointer: %s", exc)
                # Close the cache clients retained for the run's lifetime (v2
                # keeps both open so provider wrappers can hit the cache
                # endpoint on every live call). Best-effort, each guarded
                # independently so one failing close can't leak the other.
                try:
                    runtime.client.close()
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    cls.__logger.debug("Failed to close debug cache client: %s", exc)
                cls._close_debug_async_client(runtime.async_client)
            TracerManager.shutdown()
            cls.__initialized = False
            # Clear the one-shot debug-runtime state so a subsequent
            # initialize() re-reads LMNR_DEBUG* instead of resurrecting the
            # previous run.
            reset_debug_runtime()
            # Reset the process-wide run-live latch so a fresh debug run in the
            # same process (init/shutdown loop) starts from a clean cache state.
            cls.set_debug_run_live(False)
            # The pointer was just emitted above, so the atexit hook would be a
            # redundant no-op; unregister it so init/shutdown loops don't pin
            # the retired runtime (and its replay cache) alive via atexit.
            if cls.__debug_exit_hook is not None:
                atexit.unregister(cls.__debug_exit_hook)
                cls.__debug_exit_hook = None

    @classmethod
    def set_span_tags(cls, tags: list[str]):
        """Set the tags for the current span.

        Args:
            tags (list[str]): Tags to set for the span.
        """
        if not cls.is_initialized():
            return

        span = cls.get_current_span()
        if span is None:
            return
        span.set_tags(tags)

    @classmethod
    def add_span_tags(cls, tags: list[str]):
        """Add tags to the current span."""
        span = cls.get_current_span()
        if span is None:
            return
        span.add_tags(tags)

    @classmethod
    def set_trace_session_id(cls, session_id: str | None = None):
        """Set the session id for the current trace.
        Overrides any existing session id.

        Args:
            session_id (str | None, optional): Custom session id. Defaults to None.
        """
        if not cls.is_initialized():
            return

        context = set_association_prop_context(session_id=session_id, attach=True)

        span = cls.get_current_span(context=context)
        if span is None:
            cls.__logger.warning("No active span to set session id on")
            return
        span.set_trace_session_id(session_id)

    @classmethod
    def set_trace_user_id(cls, user_id: str | None = None):
        """Set the user id for the current trace.
        Overrides any existing user id.

        Args:
            user_id (str | None, optional): Custom user id. Defaults to None.
        """
        if not cls.is_initialized():
            return

        context = set_association_prop_context(user_id=user_id, attach=True)

        span = cls.get_current_span(context=context)
        if span is None:
            cls.__logger.warning("No active span to set user id on")
            return
        span.set_trace_user_id(user_id)

    @classmethod
    def set_trace_metadata(cls, metadata: dict[str, AttributeValue]):
        """Set the metadata for the current trace.

        Args:
            metadata (dict[str, AttributeValue]): Metadata to set for the trace.
        """
        if not cls.is_initialized():
            return

        merged_metadata = {**cls.__global_metadata, **(metadata or {})}

        span = cls.get_current_span()
        if span is None:
            cls.__logger.warning("No active span to set metadata on")
            return
        span.set_trace_metadata(merged_metadata)

    @classmethod
    def get_base_http_url(cls):
        return cls.__base_http_url

    @classmethod
    def get_project_api_key(cls):
        return cls.__project_api_key

    @classmethod
    def get_trace_id(cls) -> uuid.UUID | None:
        """Get the trace id for the current active span represented as a UUID.
        Returns None if there is no active span.

        Returns:
            uuid.UUID | None: The trace id for the current span, or None if\
            there is no active span.
        """
        trace_id = (
            trace.get_current_span(context=get_current_context())
            .get_span_context()
            .trace_id
        )
        if trace_id == INVALID_TRACE_ID:
            return None
        return uuid.UUID(int=trace_id)

    @classmethod
    def _headers(cls):
        assert cls.__project_api_key is not None, "Project API key is not set"
        return {
            "Authorization": "Bearer " + cls.__project_api_key,
            "Content-Type": "application/json",
        }

    @classmethod
    def _set_trace_type(
        cls,
        trace_type: TraceType,
    ):
        """Set the trace_type for the current span and the context
        Args:
            trace_type (TraceType): Type of the trace
        """
        if not cls.is_initialized():
            return

        span = trace.get_current_span(context=get_current_context())
        if span == trace.INVALID_SPAN:
            cls.__logger.warning("No active span to set trace type on")
            return
        span.set_attribute(f"{ASSOCIATION_PROPERTIES}.{TRACE_TYPE}", trace_type.value)

    @classmethod
    def _get_association_prop_attributes(
        cls,
        user_id: str | None = None,
        session_id: str | None = None,
        trace_type: TraceType | None = None,
        metadata: dict[str, AttributeValue] | None = None,
    ) -> dict[str, AttributeValue]:
        association_properties = {}
        if user_id is not None:
            association_properties[f"{ASSOCIATION_PROPERTIES}.{USER_ID}"] = user_id
        if session_id is not None:
            association_properties[f"{ASSOCIATION_PROPERTIES}.{SESSION_ID}"] = (
                session_id
            )
        if trace_type is not None:
            trace_type_val = (
                trace_type.value if isinstance(trace_type, TraceType) else trace_type
            )
            association_properties[f"{ASSOCIATION_PROPERTIES}.{TRACE_TYPE}"] = (
                trace_type_val
            )

        merged_metadata = {**cls.__global_metadata, **(metadata or {})}
        association_properties.update(
            {
                f"{ASSOCIATION_PROPERTIES}.metadata.{k}": (
                    v if is_otel_attribute_value_type(v) else json_dumps(v)
                )
                for k, v in merged_metadata.items()
            }
        )
        return association_properties
