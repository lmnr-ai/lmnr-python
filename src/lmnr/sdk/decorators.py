import datetime
import functools
from typing import Any, Callable, Literal, Optional, Union


from .context import LaminarSingleton
from .providers.fallback import FallbackProvider
from .types import NodeInput, PipelineRunResponse
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
        span_type: Optional[Literal["DEFAULT", "LLM"]] = "DEFAULT",
        capture_input: bool = True,
        capture_output: bool = True,
        release: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """The main decorator entrypoint for Laminar. This is used to wrap functions and methods to create spans.

        Args:
            name (Optional[str], optional): Name of the span. Function name is used if not specified. Defaults to None.
            span_type (Literal[&quot;DEFAULT&quot;, &quot;LLM&quot;], optional): Type of this span. Prefer `wrap_llm_call` instead of specifying
                                                                                 this as "LLM" . Defaults to "DEFAULT".
            capture_input (bool, optional): Whether to capture input parameters to the function. Defaults to True.
            capture_output (bool, optional): Whether to capture returned type from the function. Defaults to True.
            release (Optional[str], optional): Release version of your app. Useful for further grouping and analytics. Defaults to None.
            user_id (Optional[str], optional): Custom user_id of your user. Useful for grouping and further analytics. Defaults to None.
            session_id (Optional[str], optional): Custom session_id for your session. Random UUID is generated on Laminar side, if not specified.
                                                  Defaults to None.

        Raises:
            Exception: re-raises the exception if the wrapped function raises an exception

        Returns:
            Any: Returns the result of the wrapped function
        """
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
                    user_id=user_id,
                    session_id=session_id,
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
                    user_id=user_id,
                    session_id=session_id,
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
        """Update the current span with any optional metadata.

        Args:
            metadata (Optional[dict[str, Any]], optional): metadata to the span. Defaults to None.
            override (bool, optional): Whether to override the existing metadata. If False, metadata is merged with the existing metadata. Defaults to False.
        """
        laminar = LaminarSingleton().get()
        laminar.update_current_span(metadata=metadata, override=override)

    def update_current_trace(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        release: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """Update the current trace with any optional metadata.

        Args:
            user_id (Optional[str], optional): Custom user_id of your user. Useful for grouping and further analytics. Defaults to None.
            session_id (Optional[str], optional): Custom session_id for your session. Random UUID is generated on Laminar side, if not specified.
                                                  Defaults to None.
            release (Optional[str], optional): Release version of your app. Useful for further grouping and analytics. Defaults to None.
            metadata (Optional[dict[str, Any]], optional): metadata to the trace. Defaults to None.
        """
        laminar = LaminarSingleton().get()
        laminar.update_current_trace(
            user_id=user_id, session_id=session_id, release=release, metadata=metadata
        )

    def event(
        self,
        name: str,
        value: Optional[Union[str, int]] = None,
        timestamp: Optional[datetime.datetime] = None,
    ):
        """Associate an event with the current span

        Args:
            name (str): name of the event. Must be predefined in the Laminar events page.
            value (Optional[Union[str, int]], optional): value of the event. Must match range definition in Laminar events page. Defaults to None.
            timestamp (Optional[datetime.datetime], optional): If you need custom timestamp. If not specified, current time is used. Defaults to None.
        """
        laminar = LaminarSingleton().get()
        laminar.event(name, value=value, timestamp=timestamp)

    def evaluate_event(self, name: str, data: str):
        """Evaluate an event with the given name and data. The event value will be assessed by the Laminar evaluation engine.
        Data is passed as an input to the agent, so you need to specify which data you want to evaluate. Most of the times,
        this is an output of the LLM generation, but sometimes, you may want to evaluate the input or both. In the latter case,
        concatenate the input and output annotating with natural language.

        Args:
            name (str): Name of the event. Must be predefined in the Laminar events page.
            data (str): Data to be evaluated. Typically the output of the LLM generation.
        """
        laminar = LaminarSingleton().get()
        laminar.evaluate_event(name, data)

    def run_pipeline(
        self,
        pipeline: str,
        inputs: dict[str, NodeInput],
        env: dict[str, str] = None,
        metadata: dict[str, str] = None,
    ) -> PipelineRunResponse:
        """Run the laminar pipeline with the given inputs. Pipeline must be defined in the Laminar UI and have a target version.

        Args:
            pipeline (str): pipeline name
            inputs (dict[str, NodeInput]): Map from input node name to input value
            env (dict[str, str], optional): Environment variables for the pipeline executions. Typically contains API keys. Defaults to None.
            metadata (dict[str, str], optional): Any additional data to associate with the resulting span. Defaults to None.

        Returns:
            PipelineRunResponse: Response from the pipeline execution
        """
        laminar = LaminarSingleton().get()
        return laminar.run_pipeline(pipeline, inputs, env, metadata)


def wrap_llm_call(func: Callable, name: str = None, provider: str = None) -> Callable:
    """Wrap an LLM call with Laminar observability. This is a convenience function that does the same as `@observe()`, plus
    a few utilities around LLM-specific things, such as counting tokens and recording model params.

    Example usage:
    ```python
    wrap_llm_call(client.chat.completions.create)(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ],
        stream=True,
    )
    ```

    Args:
        func (Callable): The function to wrap
        name (str, optional): Name of the resulting span. Default "{provider name} completion" if not specified. Defaults to None.
        provider (str, optional): LLM model provider, e.g. openai, anthropic. This is needed to help us correctly parse
                                  things like token usage. If not specified, we infer it from the name of the package,
                                  where the function is imported from. Defaults to None.

    Raises:
        Exctption: re-raises the exception if the wrapped function raises an exception

    Returns:
        Callable: the wrapped function
    """
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
