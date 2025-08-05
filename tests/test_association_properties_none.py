"""
Test that the observe decorator handles None association_properties correctly.

This test ensures that when association_properties is None, the decorator
doesn't throw AttributeError when trying to call .get() on None.
"""
import pytest
import sys
import os

# Make sure we import from the source code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lmnr import Laminar
from lmnr.opentelemetry_lib.decorators import observe_base, async_observe_base
from lmnr.opentelemetry_lib.tracing import TracerWrapper
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def test_observe_base_with_none_association_properties(span_exporter: InMemorySpanExporter):
    """Test that observe_base handles None association_properties correctly when initialized."""
    
    # The decorator should handle None association_properties gracefully
    @observe_base(
        name="test_func",
        ignore_input=False,
        ignore_output=False,
        span_type="DEFAULT",
        association_properties=None,  # This previously caused AttributeError
    )
    def test_function(x: int) -> int:
        return x * 2
    
    # This should not throw AttributeError
    result = test_function(5)
    assert result == 10
    
    # Verify span was created
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "test_func"


def test_async_observe_base_with_none_association_properties(span_exporter: InMemorySpanExporter):
    """Test that async_observe_base handles None association_properties correctly when initialized."""
    
    # The decorator should handle None association_properties gracefully
    @async_observe_base(
        name="async_test_func",
        ignore_input=False,
        ignore_output=False,
        span_type="DEFAULT",
        association_properties=None,  # This previously caused AttributeError
    )
    async def async_test_function(x: int) -> int:
        return x * 3
    
    # This should not throw AttributeError
    import asyncio
    result = asyncio.run(async_test_function(5))
    assert result == 15
    
    # Verify span was created
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "async_test_func"


def test_observe_base_with_empty_association_properties(span_exporter: InMemorySpanExporter):
    """Test that observe_base works with empty association_properties dict."""
    
    @observe_base(
        name="test_func_empty",
        ignore_input=False,
        ignore_output=False,
        span_type="DEFAULT",
        association_properties={},  # Empty dict should work fine
    )
    def test_function(x: int) -> int:
        return x * 4
    
    result = test_function(5)
    assert result == 20
    
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "test_func_empty"


def test_observe_base_with_association_properties(span_exporter: InMemorySpanExporter):
    """Test that observe_base works correctly with actual association_properties."""
    
    @observe_base(
        name="test_func_with_props",
        ignore_input=False,
        ignore_output=False,
        span_type="DEFAULT",
        association_properties={
            "session_id": "test-session-123",
            "user_id": "test-user-456",
            "custom_prop": "custom_value"
        },
    )
    def test_function(x: int) -> int:
        return x * 5
    
    result = test_function(5)
    assert result == 25
    
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "test_func_with_props"
    # Verify association properties were set
    attrs = spans[0].attributes
    assert attrs.get("lmnr.association.properties.custom_prop") == "custom_value"


def test_observe_without_initialization():
    """Test that decorators work when Laminar is not initialized."""
    
    # Clean up any existing instance
    if hasattr(TracerWrapper, 'instance'):
        delattr(TracerWrapper, 'instance')
    
    # Ensure TracerWrapper is not initialized
    assert not TracerWrapper.verify_initialized()
    
    @observe_base(
        name="uninit_test",
        ignore_input=False,
        ignore_output=False,
        span_type="DEFAULT",
        association_properties=None,
    )
    def test_function(x: int) -> int:
        return x * 6
    
    # Should execute the original function without any tracing
    result = test_function(5)
    assert result == 30
    
    # No initialization, so still not initialized
    assert not TracerWrapper.verify_initialized()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])