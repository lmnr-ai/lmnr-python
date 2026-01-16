from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from .types import LaminarInstrumentorConfig


class BaseLaminarInstrumentor(BaseInstrumentor):
    instrumentor_config: LaminarInstrumentorConfig
