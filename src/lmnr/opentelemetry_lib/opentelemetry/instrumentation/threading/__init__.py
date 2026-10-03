# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Instrument threading to propagate OpenTelemetry context.

Copied from opentelemetry-instrumentation-threading at commit:
ad2fe813abb2ab0b6e25bedeebef5041ca3189f7
https://github.com/open-telemetry/opentelemetry-python-contrib/blob/ad2fe813abb2ab0b6e25bedeebef5041ca3189f7/instrumentation/opentelemetry-instrumentation-threading/src/opentelemetry/instrumentation/threading/__init__.py

Modified to use the Laminar isolated context.

Usage
-----

.. code-block:: python

    from opentelemetry.instrumentation.threading import ThreadingInstrumentor

    ThreadingInstrumentor().instrument()

This library provides instrumentation for the `threading` module to ensure that
the OpenTelemetry context is propagated across threads. It is important to note
that this instrumentation does not produce any telemetry data on its own. It
merely ensures that the context is correctly propagated when threads are used.


When instrumented, new threads created using threading.Thread, threading.Timer,
or within futures.ThreadPoolExecutor will have the current OpenTelemetry
context attached, and this context will be re-activated in the thread's
run method or the executor's worker thread."
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Collection, Sequence
from concurrent import futures
from typing import TYPE_CHECKING, Any

from opentelemetry import context
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from typing_extensions import override
from wrapt import (
    wrap_function_wrapper,
)

from lmnr.opentelemetry_lib.tracing.context import (
    attach_context,
    detach_context,
    get_current_context,
)

_instruments = ()

if TYPE_CHECKING:
    from typing import Protocol, TypeVar

    R = TypeVar("R")

    class HasOtelContext(Protocol):
        _otel_context: context.Context
        _lmnr_otel_context: context.Context


class ThreadingInstrumentor(BaseInstrumentor):
    __WRAPPER_START_METHOD = "start"
    __WRAPPER_RUN_METHOD = "run"
    __WRAPPER_SUBMIT_METHOD = "submit"

    @override
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    @override
    def _instrument(self, **kwargs: Any):  # pyright: ignore[reportAny, reportExplicitAny]
        self._instrument_thread()
        self._instrument_timer()
        self._instrument_thread_pool()

    @override
    def _uninstrument(self, **kwargs: Any):  # pyright: ignore[reportAny, reportExplicitAny]
        self._uninstrument_thread()
        self._uninstrument_timer()
        self._uninstrument_thread_pool()

    @staticmethod
    def _instrument_thread():
        wrap_function_wrapper(
            threading.Thread,
            ThreadingInstrumentor.__WRAPPER_START_METHOD,
            ThreadingInstrumentor.__wrap_threading_start,
        )
        wrap_function_wrapper(
            threading.Thread,
            ThreadingInstrumentor.__WRAPPER_RUN_METHOD,
            ThreadingInstrumentor.__wrap_threading_run,
        )

    @staticmethod
    def _instrument_timer():
        wrap_function_wrapper(
            threading.Timer,
            ThreadingInstrumentor.__WRAPPER_START_METHOD,
            ThreadingInstrumentor.__wrap_threading_start,
        )
        wrap_function_wrapper(
            threading.Timer,
            ThreadingInstrumentor.__WRAPPER_RUN_METHOD,
            ThreadingInstrumentor.__wrap_threading_run,
        )

    @staticmethod
    def _instrument_thread_pool():
        wrap_function_wrapper(
            futures.ThreadPoolExecutor,
            ThreadingInstrumentor.__WRAPPER_SUBMIT_METHOD,
            ThreadingInstrumentor.__wrap_thread_pool_submit,
        )

    @staticmethod
    def _uninstrument_thread():
        unwrap(threading.Thread, ThreadingInstrumentor.__WRAPPER_START_METHOD)
        unwrap(threading.Thread, ThreadingInstrumentor.__WRAPPER_RUN_METHOD)

    @staticmethod
    def _uninstrument_timer():
        unwrap(threading.Timer, ThreadingInstrumentor.__WRAPPER_START_METHOD)
        unwrap(threading.Timer, ThreadingInstrumentor.__WRAPPER_RUN_METHOD)

    @staticmethod
    def _uninstrument_thread_pool():
        unwrap(
            futures.ThreadPoolExecutor,
            ThreadingInstrumentor.__WRAPPER_SUBMIT_METHOD,
        )

    @staticmethod
    def __wrap_threading_start(
        call_wrapped: Callable[[], None],
        instance: HasOtelContext,
        args: tuple[()],
        kwargs: dict[str, Any],  # pyright: ignore[reportExplicitAny]
    ) -> None:
        instance._lmnr_otel_context = get_current_context()
        return call_wrapped(*args, **kwargs)

    @staticmethod
    def __wrap_threading_run(
        call_wrapped: Callable[..., R],
        instance: HasOtelContext,
        args: Sequence[Any],  # pyright: ignore[reportExplicitAny]
        kwargs: dict[str, Any],  # pyright: ignore[reportExplicitAny]
    ) -> R:
        token = None
        try:
            # Genearally, this must be set in __wrap_threading_start, but it is
            # possible to Thread().run() without Thread().start(), so in that case,
            # we need to capture the context here.
            # We still want to capture the context in __wrap_threading_start,
            # in order to stay close to the original implementation.
            if not hasattr(instance, "_lmnr_otel_context"):
                instance._lmnr_otel_context = get_current_context()
            token = attach_context(instance._lmnr_otel_context)
            return call_wrapped(*args, **kwargs)
        finally:
            if token is not None:
                detach_context(token)

    @staticmethod
    def __wrap_thread_pool_submit(
        call_wrapped: Callable[..., R],
        _instance: futures.ThreadPoolExecutor,
        args: tuple[Callable[..., Any], ...],  # pyright: ignore[reportExplicitAny]
        kwargs: dict[str, Any],  # pyright: ignore[reportExplicitAny]
    ) -> R:
        # obtain the original function and wrapped kwargs
        original_func = args[0]
        otel_context = get_current_context()

        def wrapped_func(*func_args: Any, **func_kwargs: Any) -> R:  # pyright: ignore[reportAny, reportExplicitAny]
            token = None
            try:
                token = attach_context(otel_context)
                return original_func(*func_args, **func_kwargs)  # pyright: ignore[reportAny]
            finally:
                if token is not None:
                    detach_context(token)

        # replace the original function with the wrapped function
        new_args: tuple[Callable[..., Any], ...] = (wrapped_func,) + args[1:]  # pyright: ignore[reportExplicitAny]
        return call_wrapped(*new_args, **kwargs)
