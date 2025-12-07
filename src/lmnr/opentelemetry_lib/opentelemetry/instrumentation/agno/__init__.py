"""OpenTelemetry Agno instrumentation"""

import logging
from typing import Collection
from importlib.metadata import version as package_version, PackageNotFoundError

from opentelemetry import context as context_api

from lmnr.sdk.laminar import Laminar
from ._tool_wrappers import (
    _FunctionCallExecuteWrapper,
    _FunctionCallAExecuteWrapper,
)
from .config import Config
from .streaming import AgnoAsyncStream, AgnoStream
from .utils import (
    dont_throw,
    set_span_attribute,
    should_send_prompts,
)
from .version import __version__
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.metrics import get_meter
from opentelemetry.semconv_ai import Meters
from opentelemetry.trace import get_tracer, SpanKind
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper

from lmnr.sdk.utils import json_dumps
from lmnr.opentelemetry_lib.tracing import get_current_context

logger = logging.getLogger(__name__)

_instruments = ("agno >= 2.0.0",)


def _get_agno_version():
    try:
        return package_version("agno")
    except PackageNotFoundError:
        return "unknown"


class AgnoInstrumentor(BaseInstrumentor):
    """An instrumentor for Agno framework."""

    def __init__(self, exception_logger=None, enrich_token_usage: bool = False):
        super().__init__()
        Config.exception_logger = exception_logger
        Config.enrich_token_usage = enrich_token_usage

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        meter_provider = kwargs.get("meter_provider")
        meter = get_meter(__name__, __version__, meter_provider)

        duration_histogram = meter.create_histogram(
            name=Meters.LLM_OPERATION_DURATION,
            unit="s",
            description="GenAI operation duration",
        )

        token_histogram = meter.create_histogram(
            name=Meters.LLM_TOKEN_USAGE,
            unit="token",
            description="Measures number of input and output tokens used",
        )

        # Wrap Agent methods
        wrap_function_wrapper(
            module="agno.agent",
            name="Agent.run",
            wrapper=_AgentRunWrapper(tracer, duration_histogram, token_histogram),
        )

        wrap_function_wrapper(
            module="agno.agent",
            name="Agent.arun",
            wrapper=_AgentARunWrapper(tracer, duration_histogram, token_histogram),
        )

        # Wrap Team methods if available
        try:
            wrap_function_wrapper(
                module="agno.team",
                name="Team.run",
                wrapper=_TeamRunWrapper(tracer, duration_histogram, token_histogram),
            )

            wrap_function_wrapper(
                module="agno.team",
                name="Team.arun",
                wrapper=_TeamARunWrapper(tracer, duration_histogram, token_histogram),
            )
        except Exception as e:
            logger.debug(f"Could not instrument Team: {e}")

        # Wrap FunctionCall methods for tool execution
        try:
            wrap_function_wrapper(
                module="agno.tools",
                name="FunctionCall.execute",
                wrapper=_FunctionCallExecuteWrapper(
                    tracer, duration_histogram, token_histogram
                ),
            )

            wrap_function_wrapper(
                module="agno.tools",
                name="FunctionCall.aexecute",
                wrapper=_FunctionCallAExecuteWrapper(
                    tracer, duration_histogram, token_histogram
                ),
            )
        except Exception as e:
            logger.debug(f"Could not instrument FunctionCall: {e}")

        # Wrap Workflow methods
        try:
            wrap_function_wrapper(
                module="agno.workflow",
                name="Workflow.run",
                wrapper=_WorkflowRunWrapper(
                    tracer, duration_histogram, token_histogram
                ),
            )

            wrap_function_wrapper(
                module="agno.workflow",
                name="Workflow.arun",
                wrapper=_WorkflowARunWrapper(
                    tracer, duration_histogram, token_histogram
                ),
            )
        except Exception as e:
            logger.debug(f"Could not instrument Workflow: {e}")

    def _uninstrument(self, **kwargs):
        unwrap("agno.agent", "Agent.run")
        unwrap("agno.agent", "Agent.arun")
        try:
            unwrap("agno.team", "Team.run")
            unwrap("agno.team", "Team.arun")
        except Exception:
            pass
        try:
            unwrap("agno.tools", "FunctionCall.execute")
            unwrap("agno.tools", "FunctionCall.aexecute")
        except Exception:
            pass
        try:
            unwrap("agno.workflow", "Workflow.run")
            unwrap("agno.workflow", "Workflow.arun")
        except Exception:
            pass


class _AgentRunWrapper:
    """Wrapper for Agent.run() method to capture synchronous agent execution."""

    def __init__(self, tracer, duration_histogram, token_histogram):
        """Initialize the wrapper with OpenTelemetry instrumentation objects."""
        self._tracer = tracer
        self._duration_histogram = duration_histogram
        self._token_histogram = token_histogram

    @dont_throw
    def __call__(self, wrapped, instance, args, kwargs):
        """Wrap the Agent.run() call with tracing instrumentation."""
        if context_api.get_value(
            context_api._SUPPRESS_INSTRUMENTATION_KEY
        ) or context_api.get_value("suppress_agno_instrumentation"):
            return wrapped(*args, **kwargs)

        is_streaming = kwargs.get("stream", False)

        if is_streaming:
            span_name = f"{getattr(instance, 'name', 'unknown')}.agent"

            span = self._tracer.start_span(
                span_name,
                kind=SpanKind.CLIENT,
                context=get_current_context(),
            )

            try:
                # span.set_attribute(GenAIAttributes.GEN_AI_SYSTEM, "agno")

                # if hasattr(instance, "name") and instance.name:
                #     span.set_attribute(GenAIAttributes.GEN_AI_AGENT_NAME, instance.name)

                # if hasattr(instance, "model") and instance.model:
                #     model_name = getattr(
                #         instance.model, "id", getattr(instance.model, "name", "unknown")
                #     )
                #     span.set_attribute(GenAIAttributes.GEN_AI_REQUEST_MODEL, model_name)

                if args and should_send_prompts():
                    input_message = json_dumps(args[0])
                    span.set_attribute("lmnr.span.input", input_message)

                import time

                start_time = time.time()

                response = wrapped(*args, **kwargs)

                return AgnoStream(
                    span,
                    response,
                    instance,
                    start_time,
                    self._duration_histogram,
                    self._token_histogram,
                )

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.end()
                raise
        else:
            span_name = f"{getattr(instance, 'name', 'unknown')}.agent"

            with Laminar.start_as_current_span(span_name) as span:
                try:
                    # span.set_attribute(GenAIAttributes.GEN_AI_SYSTEM, "agno")

                    # if hasattr(instance, "name") and instance.name:
                    #     span.set_attribute(
                    #         GenAIAttributes.GEN_AI_AGENT_NAME, instance.name
                    #     )

                    # if hasattr(instance, "model") and instance.model:
                    #     model_name = getattr(
                    #         instance.model,
                    #         "id",
                    #         getattr(instance.model, "name", "unknown"),
                    #     )
                    #     span.set_attribute(
                    #         GenAIAttributes.GEN_AI_REQUEST_MODEL, model_name
                    #     )

                    if args and should_send_prompts():
                        input_message = json_dumps(args[0])
                        span.set_attribute("lmnr.span.input", input_message)

                    import time

                    start_time = time.time()

                    result = wrapped(*args, **kwargs)

                    duration = time.time() - start_time

                    if hasattr(result, "content") and should_send_prompts():
                        set_span_attribute(
                            span, "lmnr.span.output", json_dumps(result.content)
                        )

                    if hasattr(result, "run_id") and result.run_id:
                        set_span_attribute(span, "agno.run.id", result.run_id)

                    # if hasattr(result, "metrics"):
                    #     metrics = result.metrics
                    #     if hasattr(metrics, "input_tokens"):
                    #         span.set_attribute(
                    #             GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
                    #             metrics.input_tokens,
                    #         )
                    #     if hasattr(metrics, "output_tokens"):
                    #         span.set_attribute(
                    #             GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
                    #             metrics.output_tokens,
                    #         )
                    #     if hasattr(metrics, "total_tokens"):
                    #         span.set_attribute(
                    #             SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
                    #             metrics.total_tokens,
                    #         )

                    span.set_status(Status(StatusCode.OK))

                    self._duration_histogram.record(
                        duration,
                        attributes={
                            "lmnr.span.type": "DEFAULT",
                        },
                    )

                    return result

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise


class _AgentARunWrapper:
    """Wrapper for Agent.arun() method to capture asynchronous agent execution."""

    def __init__(self, tracer, duration_histogram, token_histogram):
        """Initialize the wrapper with OpenTelemetry instrumentation objects."""
        self._tracer = tracer
        self._duration_histogram = duration_histogram
        self._token_histogram = token_histogram

    @dont_throw
    def __call__(self, wrapped, instance, args, kwargs):
        """Wrap the Agent.arun() call with tracing instrumentation."""
        if context_api.get_value(
            context_api._SUPPRESS_INSTRUMENTATION_KEY
        ) or context_api.get_value("suppress_agno_instrumentation"):
            return wrapped(*args, **kwargs)

        is_streaming = kwargs.get("stream", False)

        if is_streaming:
            span_name = f"{getattr(instance, 'name', 'unknown')}.agent"

            span = self._tracer.start_span(
                span_name,
                kind=SpanKind.CLIENT,
                context=get_current_context(),
            )

            try:
                # span.set_attribute(GenAIAttributes.GEN_AI_SYSTEM, "agno")

                # if hasattr(instance, "name") and instance.name:
                #     span.set_attribute(GenAIAttributes.GEN_AI_AGENT_NAME, instance.name)

                # if hasattr(instance, "model") and instance.model:
                #     model_name = getattr(
                #         instance.model, "id", getattr(instance.model, "name", "unknown")
                #     )
                #     span.set_attribute(GenAIAttributes.GEN_AI_REQUEST_MODEL, model_name)

                if args and should_send_prompts():
                    input_message = json_dumps(args[0])
                    span.set_attribute("lmnr.span.input", input_message)

                import time

                start_time = time.time()

                response = wrapped(*args, **kwargs)

                return AgnoAsyncStream(
                    span,
                    response,
                    instance,
                    start_time,
                    self._duration_histogram,
                    self._token_histogram,
                )

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.end()
                raise
        else:

            async def async_wrapper():
                span_name = f"{getattr(instance, 'name', 'unknown')}.agent"

                with Laminar.start_as_current_span(span_name) as span:
                    try:
                        # span.set_attribute(GenAIAttributes.GEN_AI_SYSTEM, "agno")

                        # if hasattr(instance, "name") and instance.name:
                        #     span.set_attribute(
                        #         GenAIAttributes.GEN_AI_AGENT_NAME, instance.name
                        #     )

                        # if hasattr(instance, "model") and instance.model:
                        #     model_name = getattr(
                        #         instance.model,
                        #         "id",
                        #         getattr(instance.model, "name", "unknown"),
                        #     )
                        #     span.set_attribute(
                        #         GenAIAttributes.GEN_AI_REQUEST_MODEL, model_name
                        #     )

                        if args and should_send_prompts():
                            input_message = json_dumps(args[0])
                            span.set_attribute("lmnr.span.input", input_message)

                        import time

                        start_time = time.time()

                        result = await wrapped(*args, **kwargs)

                        duration = time.time() - start_time

                        if hasattr(result, "content") and should_send_prompts():
                            span.set_attribute(
                                "lmnr.span.output",
                                json_dumps(result.content),
                            )

                        if hasattr(result, "run_id"):
                            span.set_attribute("agno.run.id", result.run_id)

                        # if hasattr(result, "metrics"):
                        #     metrics = result.metrics
                        #     if hasattr(metrics, "input_tokens"):
                        #         span.set_attribute(
                        #             GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
                        #             metrics.input_tokens,
                        #         )
                        #     if hasattr(metrics, "output_tokens"):
                        #         span.set_attribute(
                        #             GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
                        #             metrics.output_tokens,
                        #         )
                        #     if hasattr(metrics, "total_tokens"):
                        #         span.set_attribute(
                        #             SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
                        #             metrics.total_tokens,
                        #         )

                        span.set_status(Status(StatusCode.OK))

                        self._duration_histogram.record(
                            duration,
                            attributes={
                                "lmnr.span.type": "DEFAULT",
                            },
                        )

                        return result

                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise

            return async_wrapper()


class _TeamRunWrapper:
    """Wrapper for Team.run() method to capture synchronous team execution."""

    def __init__(self, tracer, duration_histogram, token_histogram):
        """Initialize the wrapper with OpenTelemetry instrumentation objects."""
        self._tracer = tracer
        self._duration_histogram = duration_histogram
        self._token_histogram = token_histogram

    @dont_throw
    def __call__(self, wrapped, instance, args, kwargs):
        """Wrap the Team.run() call with tracing instrumentation."""
        if context_api.get_value(
            context_api._SUPPRESS_INSTRUMENTATION_KEY
        ) or context_api.get_value("suppress_agno_instrumentation"):
            return wrapped(*args, **kwargs)

        span_name = f"{getattr(instance, 'name', 'unknown')}.team"

        with Laminar.start_as_current_span(span_name) as span:
            try:
                # span.set_attribute(GenAIAttributes.GEN_AI_SYSTEM, "agno")

                if hasattr(instance, "name") and instance.name:
                    span.set_attribute("agno.team.name", instance.name)

                if args and should_send_prompts():
                    input_message = json_dumps(args[0])
                    span.set_attribute("lmnr.span.input", input_message)

                import time

                start_time = time.time()

                result = wrapped(*args, **kwargs)

                duration = time.time() - start_time

                if hasattr(result, "content") and should_send_prompts():
                    span.set_attribute("lmnr.span.output", json_dumps(result.content))

                if hasattr(result, "run_id"):
                    span.set_attribute("agno.run.id", result.run_id)

                span.set_status(Status(StatusCode.OK))

                self._duration_histogram.record(
                    duration,
                    attributes={
                        "lmnr.span.type": "DEFAULT",
                    },
                )

                return result

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise


class _TeamARunWrapper:
    """Wrapper for Team.arun() method to capture asynchronous team execution."""

    def __init__(self, tracer, duration_histogram, token_histogram):
        """Initialize the wrapper with OpenTelemetry instrumentation objects."""
        self._tracer = tracer
        self._duration_histogram = duration_histogram
        self._token_histogram = token_histogram

    @dont_throw
    async def __call__(self, wrapped, instance, args, kwargs):
        """Wrap the Team.arun() call with tracing instrumentation."""
        if context_api.get_value(
            context_api._SUPPRESS_INSTRUMENTATION_KEY
        ) or context_api.get_value("suppress_agno_instrumentation"):
            return await wrapped(*args, **kwargs)

        span_name = f"{getattr(instance, 'name', 'unknown')}.team"

        with Laminar.start_as_current_span(span_name) as span:
            try:
                # span.set_attribute(GenAIAttributes.GEN_AI_SYSTEM, "agno")

                if hasattr(instance, "name") and instance.name:
                    span.set_attribute("agno.team.name", instance.name)

                if args and should_send_prompts():
                    input_message = json_dumps(args[0])
                    span.set_attribute("lmnr.span.input", input_message)

                import time

                start_time = time.time()

                result = await wrapped(*args, **kwargs)

                duration = time.time() - start_time

                if hasattr(result, "content") and should_send_prompts():
                    span.set_attribute("lmnr.span.output", json_dumps(result.content))

                if hasattr(result, "run_id") and result.run_id:
                    span.set_attribute("agno.run.id", result.run_id)

                span.set_status(Status(StatusCode.OK))

                self._duration_histogram.record(
                    duration,
                    attributes={
                        "lmnr.span.type": "DEFAULT",
                    },
                )

                return result

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise


class _WorkflowRunWrapper:
    """Wrapper for Workflow.run() method to capture synchronous workflow execution."""

    def __init__(self, tracer, duration_histogram, token_histogram):
        """Initialize the wrapper with OpenTelemetry instrumentation objects."""
        self._tracer = tracer
        self._duration_histogram = duration_histogram
        self._token_histogram = token_histogram

    @dont_throw
    def __call__(self, wrapped, instance, args, kwargs):
        """Wrap the Workflow.run() call with tracing instrumentation."""
        if context_api.get_value(
            context_api._SUPPRESS_INSTRUMENTATION_KEY
        ) or context_api.get_value("suppress_agno_instrumentation"):
            return wrapped(*args, **kwargs)

        span_name = f"{getattr(instance, 'name', 'unknown')}.workflow"

        with Laminar.start_as_current_span(span_name) as span:
            try:
                # span.set_attribute(GenAIAttributes.GEN_AI_SYSTEM, "agno")

                if hasattr(instance, "name") and instance.name:
                    span.set_attribute("agno.workflow.name", instance.name)

                if args and should_send_prompts():
                    input_message = json_dumps(args[0])
                    span.set_attribute("lmnr.span.input", input_message)

                import time

                start_time = time.time()

                result = wrapped(*args, **kwargs)

                duration = time.time() - start_time

                if hasattr(result, "content") and should_send_prompts():
                    span.set_attribute("lmnr.span.output", json_dumps(result.content))

                if hasattr(result, "run_id") and result.run_id:
                    span.set_attribute("agno.run.id", result.run_id)

                span.set_status(Status(StatusCode.OK))

                self._duration_histogram.record(
                    duration,
                    attributes={
                        "lmnr.span.type": "DEFAULT",
                    },
                )

                return result

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise


class _WorkflowARunWrapper:
    """Wrapper for Workflow.arun() method to capture asynchronous workflow execution."""

    def __init__(self, tracer, duration_histogram, token_histogram):
        """Initialize the wrapper with OpenTelemetry instrumentation objects."""
        self._tracer = tracer
        self._duration_histogram = duration_histogram
        self._token_histogram = token_histogram

    @dont_throw
    async def __call__(self, wrapped, instance, args, kwargs):
        """Wrap the Workflow.arun() call with tracing instrumentation."""
        if context_api.get_value(
            context_api._SUPPRESS_INSTRUMENTATION_KEY
        ) or context_api.get_value("suppress_agno_instrumentation"):
            return await wrapped(*args, **kwargs)

        span_name = f"{getattr(instance, 'name', 'unknown')}.workflow"

        with Laminar.start_as_current_span(span_name) as span:
            try:
                # span.set_attribute(GenAIAttributes.GEN_AI_SYSTEM, "agno")

                if hasattr(instance, "name") and instance.name:
                    span.set_attribute("agno.workflow.name", instance.name)

                if args and should_send_prompts():
                    input_message = json_dumps(args[0])
                    span.set_attribute("lmnr.span.input", input_message)

                import time

                start_time = time.time()

                result = await wrapped(*args, **kwargs)

                duration = time.time() - start_time

                if hasattr(result, "content") and should_send_prompts():
                    span.set_attribute("lmnr.span.output", json_dumps(result.content))

                if hasattr(result, "run_id") and result.run_id:
                    span.set_attribute("agno.run.id", result.run_id)

                span.set_status(Status(StatusCode.OK))

                self._duration_histogram.record(
                    duration,
                    attributes={
                        "lmnr.span.type": "DEFAULT",
                    },
                )

                return result

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
