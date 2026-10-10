import abc
from typing import Any

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor


class InstrumentorInitializer(abc.ABC):
    """Builds one instrumentor, or returns None if its package is missing.

    Lives in its own leaf module so `tracing/instruments.py` can type the
    initializer registry without importing the concrete initializers (which
    import every instrumentor, and through them `Laminar`).
    """

    @abc.abstractmethod
    def init_instrumentor(self, *args: Any, **kwargs: Any) -> BaseInstrumentor | None:
        pass
