"""OpenTelemetry Exa API instrumentation"""

import logging
from typing import Collection, Generator

from pydantic import BaseModel

from lmnr.opentelemetry_lib.decorators import json_dumps
from lmnr.opentelemetry_lib.tracing.context import (
    get_current_context,
    get_event_attributes_from_context,
)

from .utils import (
    dont_throw,
    with_tracer_wrapper,
)

from opentelemetry.trace import Tracer
from wrapt import wrap_function_wrapper

from opentelemetry import context as context_api
from opentelemetry.trace import get_tracer, SpanKind, Span, Status, StatusCode
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY, unwrap

from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
)

logger = logging.getLogger(__name__)

_instruments = ("exa-py >= 1.0.0",)

WRAPPED_METHODS = [
    # Basic search methods
    {
        "package": "exa_py",
        "object": "Exa",
        "method": "search",
        "span_name": "exa.search",
        "operation_type": "search",
        "is_streaming": False,
        "is_async": False,
    },
    {
        "package": "exa_py",
        "object": "Exa",
        "method": "search_and_contents",
        "span_name": "exa.search_and_contents",
        "operation_type": "search",
        "is_streaming": False,
        "is_async": False,
    },
    {
        "package": "exa_py",
        "object": "Exa",
        "method": "find_similar",
        "span_name": "exa.find_similar",
        "operation_type": "search",
        "is_streaming": False,
        "is_async": False,
    },
    {
        "package": "exa_py",
        "object": "Exa",
        "method": "find_similar_and_contents",
        "span_name": "exa.find_similar_and_contents",
        "operation_type": "search",
        "is_streaming": False,
        "is_async": False,
    },
    # Answer methods
    {
        "package": "exa_py",
        "object": "Exa",
        "method": "answer",
        "span_name": "exa.answer",
        "operation_type": "answer",
        "is_streaming": False,
        "is_async": False,
    },
    {
        "package": "exa_py",
        "object": "Exa",
        "method": "stream_answer",
        "span_name": "exa.stream_answer",
        "operation_type": "answer",
        "is_streaming": True,
        "is_async": False,
    }
]

@dont_throw
def _extract_service_metadata(operation_type: str, method_name: str, args: tuple, kwargs: dict) -> dict:
    """Extract Exa-specific metadata from request."""
    metadata = {}
    
    if operation_type == "search":
        metadata["num_results"] = kwargs.get("num_results", 10)
        metadata["search_type"] = kwargs.get("type", "auto")
        metadata["category"] = kwargs.get("category")
        metadata["include_domains"] = kwargs.get("include_domains")
        metadata["exclude_domains"] = kwargs.get("exclude_domains")
        metadata["include_text"] = kwargs.get("text", False)
        metadata["highlights"] = kwargs.get("highlights", False)
        metadata["summary"] = kwargs.get("summary", False)
        
        if "find_similar" in method_name:
            metadata["similar_url"] = args[0] if args else None
        else:
            metadata["query"] = args[0] if args else kwargs.get("query")
    
    elif operation_type == "answer":
        metadata["query"] = args[0] if args else kwargs.get("query")
        metadata["include_text"] = kwargs.get("text", False)
    
    elif operation_type == "research":
        metadata["instructions"] = args[0] if args else kwargs.get("instructions")
        metadata["model"] = kwargs.get("model", "exa-research")
        metadata["infer_schema"] = kwargs.get("infer_schema", False)
        metadata["has_output_schema"] = "output_schema" in kwargs
    
    # Remove None values
    return {k: v for k, v in metadata.items() if v is not None}

@dont_throw
def _extract_response_metadata(response) -> dict:
    """Extract metadata from Exa response."""
    metadata = {}
    
    try:
        if isinstance(response, str) or response is None:
            return metadata
        
        # Handle different response types safely
        response_dict = None
        if isinstance(response, dict):
            response_dict = response
        elif hasattr(response, 'model_dump') and callable(getattr(response, 'model_dump')):
            try:
                response_dict = response.model_dump()
            except Exception:
                response_dict = getattr(response, '__dict__', {})
        elif hasattr(response, '__dict__'):
            response_dict = response.__dict__
        
        if not isinstance(response_dict, dict):
            return metadata
        
        # Extract common response fields
        if "results" in response_dict:
            results = response_dict["results"]
            if isinstance(results, list):
                metadata["actual_results_count"] = len(results)
        
        if "citations" in response_dict:
            citations = response_dict["citations"]
            if isinstance(citations, list):
                metadata["citations_count"] = len(citations)
            elif isinstance(citations, dict):
                try:
                    total_citations = sum(
                        len(cits) if isinstance(cits, list) else 1 
                        for cits in citations.values() 
                        if cits is not None
                    )
                    metadata["citations_count"] = total_citations
                except (TypeError, ValueError):
                    metadata["citations_count"] = len(citations)
        
        if "requestId" in response_dict and response_dict["requestId"]:
            metadata["request_id"] = str(response_dict["requestId"])
        
        if "autopromptString" in response_dict:
            autoprompt = response_dict["autopromptString"]
            metadata["autoprompt_used"] = bool(autoprompt) if autoprompt is not None else False
        
        # Extract cost information from response
        if "costDollars" in response_dict:
            cost_data = response_dict["costDollars"]
            if isinstance(cost_data, dict):
                metadata["cost_dollars"] = cost_data
                
                # Extract total cost
                if "total" in cost_data and isinstance(cost_data["total"], (int, float)):
                    metadata["actual_cost_total"] = cost_data["total"]

    except Exception:
        pass
    
    return metadata


@dont_throw
def _update_cost_from_response(span: Span, response_metadata: dict):
    """Update cost based on actual response data."""
    try:
        # Use actual cost from Exa response if available
        actual_cost = response_metadata.get("actual_cost_total")
        if actual_cost is not None:
            span.set_attribute("service.cost.amount", actual_cost)
            span.set_attribute("service.cost.unit", "dollars")
            span.set_attribute("service.cost.unit_count", 1)
            span.set_attribute("service.cost.model", "actual")
        
        # Update unit count based on actual results
        actual_results = response_metadata.get("actual_results_count")
        if actual_results is not None and actual_cost is None:
            # Only update unit count if we don't have actual cost
            span.set_attribute("service.cost.unit_count", actual_results)
    
    except Exception:
        pass


@dont_throw
def _set_request_attributes(span: Span, to_wrap: dict, args: tuple, kwargs: dict):
    """Set request attributes on the span."""
    operation_type = to_wrap.get("operation_type", "search")
    method_name = to_wrap.get("method", "")
    
    # Set unified service attributes
    span.set_attribute("service.name", "exa")
    span.set_attribute("service.operation", operation_type)
    span.set_attribute("service.method", method_name)
    
    # Capture input using lmnr.span.input
    input_data = {
        "args": list(args),
        "kwargs": {k: v for k, v in kwargs.items() if k.lower() not in {"api_key", "token", "secret", "password"}}
    }
    try:
        span.set_attribute("lmnr.span.input", json_dumps(input_data))
    except Exception:
        span.set_attribute("lmnr.span.input", str(input_data))
    
    # Set service metadata as stringified JSON
    metadata = _extract_service_metadata(operation_type, method_name, args, kwargs)
    try:
        span.set_attribute("service.metadata", json_dumps(metadata))
    except Exception:
        span.set_attribute("service.metadata", str(metadata))
    
    # Set initial cost attributes (will be updated with actual cost from response)
    span.set_attribute("service.cost.amount", 0.0)
    span.set_attribute("service.cost.unit", "requests")
    span.set_attribute("service.cost.unit_count", 1)
    span.set_attribute("service.cost.model", "estimated")


@dont_throw
def _set_response_attributes(span: Span, to_wrap: dict, response):
    """Set response attributes on the span."""
    # Capture output using lmnr.span.output
    try:
        span.set_attribute("lmnr.span.output", json_dumps(response))
    except Exception:
        span.set_attribute("lmnr.span.output", str(response))
    
    # Set service response metrics
    try:
        if isinstance(response, str):
            span.set_attribute("service.response.size", len(response))
        else:
            response_json = json_dumps(response)
            span.set_attribute("service.response.size", len(response_json))
    except Exception:
        pass
    
    # Set response status
    span.set_attribute("service.response.status", "success")
    
    # Extract response metadata for cost adjustment
    response_metadata = _extract_response_metadata(response)
    if response_metadata:
        # Update cost based on actual usage (e.g., actual result count)
        _update_cost_from_response(span, response_metadata)


@dont_throw
def _build_from_streaming_response(
    span: Span, response: Generator, to_wrap: dict
) -> Generator:
    """Handle streaming responses for stream_answer method."""
    collected_chunks = []
    collected_content = []
    chunk_count = 0
    
    for chunk in response:
        chunk_count += 1
        collected_chunks.append(chunk)
        
        # Process chunk for final attributes
        if hasattr(chunk, 'content') and chunk.content:
            collected_content.append(chunk.content)
        
        yield chunk
    
    try:
        # Set final output using lmnr.span.output for streaming
        if collected_content:
            output_data = "".join(collected_content)
        else:
            output_data = ""
        
        try:
            span.set_attribute("lmnr.span.output", json_dumps(output_data))
        except Exception:
            span.set_attribute("lmnr.span.output", str(output_data))
        
        # Set service response metrics
        span.set_attribute("service.response.chunks_total", chunk_count)
        span.set_attribute("service.response.status", "success")
    
    finally:
        if span.is_recording():
            span.set_status(Status(StatusCode.OK))
            span.end()


@with_tracer_wrapper
def _wrap(tracer: Tracer, to_wrap, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return wrapped(*args, **kwargs)

    span = tracer.start_span(
        to_wrap.get("span_name"),
        kind=SpanKind.CLIENT,
        context=get_current_context(),
    )

    if span.is_recording():
        _set_request_attributes(span, to_wrap, args, kwargs)

    try:
        response = wrapped(*args, **kwargs)
        
        if to_wrap.get("is_streaming"):
            return _build_from_streaming_response(span, response, to_wrap)
        
        if span.is_recording():
            _set_response_attributes(span, to_wrap, response)
        
        span.set_status(Status(StatusCode.OK))
        span.end()
        return response
    
    except Exception as e:
        attributes = get_event_attributes_from_context()
        span.set_attribute(ERROR_TYPE, e.__class__.__name__)
        span.set_attribute("service.response.status", "error")
        span.record_exception(e, attributes=attributes)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.end()
        raise


class ExaInstrumentor(BaseInstrumentor):
    """An instrumentor for Exa's Python SDK."""

    def __init__(self):
        super().__init__()

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, "0.0.1a1", tracer_provider)

        for wrapped_method in WRAPPED_METHODS:
            try:
                wrap_function_wrapper(
                    wrapped_method.get("package"),
                    f"{wrapped_method.get('object')}.{wrapped_method.get('method')}",
                    _wrap(tracer, wrapped_method),
                )
            except Exception as e:
                logger.debug(f"Failed to instrument {wrapped_method.get('method')}: {e}")

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            try:
                unwrap(
                    f"{wrapped_method.get('package')}.{wrapped_method.get('object')}",
                    wrapped_method.get("method"),
                )
            except Exception as e:
                logger.debug(f"Failed to uninstrument {wrapped_method.get('method')}: {e}")
