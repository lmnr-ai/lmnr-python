import functools
from collections.abc import Callable
from typing import Any

import pytest
from openai import OpenAI
from typing_extensions import TypeVar

T = TypeVar("T")

def test_inner_exception_isnt_caught(
    openai_client: OpenAI,
):
    should_throw = True

    def throw_exception(f: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any):
            if should_throw:
                raise RuntimeError("Test exception")
            else:
                return f(*args, **kwargs)

        return wrapper

    with pytest.raises(RuntimeError):
        _ = throw_exception(openai_client.chat.completions.create)(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Tell me a joke about opentelemetry"}
            ],
        )

    should_throw = False


@pytest.mark.vcr
def test_exception_in_instrumentation_suppressed(
    openai_client: OpenAI,
):
    should_scramble = True

    def scramble_response(f: Callable[..., T]) -> Callable[..., Any]:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> T | dict[Any, Any]:
            if should_scramble:
                response = f(*args, **kwargs)
                response = {}
                return response
            else:
                return f(*args, **kwargs)

        return wrapper

    _ = scramble_response(openai_client.chat.completions.create)(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )
