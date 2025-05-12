from lmnr.opentelemetry_lib.tracing.instruments import (
    Instruments,
    INSTRUMENTATION_INITIALIZERS,
)


def test_same_number_of_instrumentation_initializers():
    assert len(INSTRUMENTATION_INITIALIZERS) == len(Instruments)
