import datetime
import functools
from typing import Any, Callable, Optional

from .context import LaminarSingleton
from .providers.fallback import FallbackProvider
from .utils import (
    PROVIDER_NAME_TO_OBJECT,
    get_input_from_func_args,
    is_async,
    is_method,
)


class LaminarDecorator:
    def observe(
        self,
        *,
        name: Optional[str] = None,
        span_type: Optional[str] = "DEFAULT",
        capture_input: bool = True,
        capture_output: bool = True,
        release: Optional[str] = None,
    ):
        context_manager = LaminarSingleton().get()

        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                span = context_manager.observe_start(
                    name=name or func.__name__,
                    span_type=span_type,
                    input=(
                        get_input_from_func_args(func, is_method(func), args, kwargs)
                        if capture_input
                        else None
                    ),
                    release=release,
                )
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    context_manager.observe_end(result=None, span=span, error=e)
                    raise e
                context_manager.observe_end(
                    result=result if capture_output else None, span=span
                )
                return result

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                span = context_manager.observe_start(
                    name=name or func.__name__,
                    span_type=span_type,
                    input=(
                        get_input_from_func_args(func, is_method(func), args, kwargs)
                        if capture_input
                        else None
                    ),
                    release=release,
                )
                try:
                    result = await func(*args, **kwargs)
                except Exception as e:
                    context_manager.observe_end(result=None, span=span, error=e)
                    raise e
                context_manager.observe_end(
                    result=result if capture_output else None, span=span
                )
                return result

            return async_wrapper if is_async(func) else wrapper

        return decorator

    def update_current_span(
        self,
        metadata: Optional[dict[str, Any]] = None,
        override: bool = False,
    ):
        laminar = LaminarSingleton().get()
        laminar.update_current_span(metadata=metadata, override=override)

    def update_current_trace(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        release: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        laminar = LaminarSingleton().get()
        laminar.update_current_trace(
            user_id=user_id, session_id=session_id, release=release, metadata=metadata
        )

    def event(self, name: str, timestamp: Optional[datetime.datetime] = None):
        laminar = LaminarSingleton().get()
        laminar.add_event(name)

    def check_span_event(self, name: str):
        laminar = LaminarSingleton().get()
        laminar.add_check_event_name(name)

    def run_pipeline(
        self,
        pipeline: str,
        inputs: dict[str, Any],
        env: dict[str, str] = None,
        metadata: dict[str, str] = None,
        stream: bool = False,
    ):
        laminar = LaminarSingleton().get()
        return laminar.run_pipeline(pipeline, inputs, env, metadata, stream)


def wrap_llm_call(func: Callable, name: str = None, provider: str = None) -> Callable:
    laminar = LaminarSingleton().get()
    # Simple heuristic to determine the package from where the LLM call is imported.
    # This works for major providers, but will likely make no sense for custom providers.
    provider_name = (
        provider.lower().strip() if provider else func.__module__.split(".")[0]
    )
    provider_module = PROVIDER_NAME_TO_OBJECT.get(provider_name, FallbackProvider())
    name = name or f"{provider_module.display_name()} completion"

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        inp = kwargs.get("messages")
        attributes = (
            provider_module.extract_llm_attributes_from_args(args, kwargs)
            if provider_module
            else {}
        )
        attributes["provider"] = provider_name
        span = laminar.observe_start(
            name=name, span_type="LLM", input=inp, attributes=attributes
        )
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            laminar.observe_end(
                result=None, span=span, error=e, provider_name=provider_name
            )
            raise e
        return laminar.observe_end(
            result=result, span=span, provider_name=provider_name
        )

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        inp = kwargs.get("messages")
        attributes = (
            provider_module.extract_llm_attributes_from_args(args, kwargs)
            if provider_module
            else {}
        )
        attributes["provider"] = provider_name
        span = laminar.observe_start(
            name=name, span_type="LLM", input=inp, attributes=attributes
        )
        try:
            result = await func(*args, **kwargs)
        except Exception as e:
            laminar.observe_end(
                result=None, span=span, error=e, provider_name=provider_name
            )
            raise e
        return laminar.observe_end(
            result=result, span=span, provider_name=provider_name
        )

    return async_wrapper if is_async(func) else wrapper


lmnr_context = LaminarDecorator()
observe = lmnr_context.observe
