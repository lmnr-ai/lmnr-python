from lmnr.opentelemetry_lib.tracing.instruments import (
    INSTRUMENTATION_INITIALIZERS,
    Instruments,
)


def test_same_number_of_instrumentation_initializers():
    assert len(INSTRUMENTATION_INITIALIZERS) == len(Instruments)
