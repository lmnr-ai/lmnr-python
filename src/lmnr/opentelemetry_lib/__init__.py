"""Laminar's OpenTelemetry layer.

Tracing lifecycle lives in `lmnr.opentelemetry_lib.tracing` (`init_tracing`,
`get_tracer_wrapper`, `flush_tracing`, ...). Nothing is re-exported here on
purpose: importing this package must not drag in the whole tracing stack.
"""
