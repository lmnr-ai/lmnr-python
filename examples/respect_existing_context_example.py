"""
Example demonstrating the respect_existing_context feature.

This example shows how to configure Laminar to respect existing OpenTelemetry
context from other instrumented frameworks like Prefect, FastAPI, etc.
"""
from lmnr import Laminar, observe
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter


def example_isolated_context():
    """Example showing default behavior - Laminar uses isolated context."""
    print("=== Example 1: Isolated Context (Default) ===")
    
    # Initialize Laminar with default settings
    Laminar.initialize(
        project_api_key="your-api-key",
        respect_existing_context=False,  # This is the default
    )
    
    # Create a tracer for another framework (e.g., Prefect)
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    tracer = provider.get_tracer("prefect")
    
    # Create a span from the external framework
    with tracer.start_as_current_span("prefect_task") as prefect_span:
        print(f"Prefect trace ID: {prefect_span.get_span_context().trace_id:032x}")
        
        # Call a Laminar-instrumented function
        @observe(name="laminar_function")
        def process_data():
            current = trace.get_current_span()
            print(f"Laminar trace ID: {current.get_span_context().trace_id:032x}")
            return "processed"
        
        process_data()
    
    print("Notice: The trace IDs are different - spans are isolated\n")
    Laminar.shutdown()


def example_shared_context():
    """Example showing how to share context with external frameworks."""
    print("=== Example 2: Shared Context ===")
    
    # Initialize Laminar to respect existing context
    Laminar.initialize(
        project_api_key="your-api-key",
        respect_existing_context=True,  # Enable context sharing
    )
    
    # Create a tracer for another framework
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    tracer = provider.get_tracer("prefect")
    
    # Create a span from the external framework
    with tracer.start_as_current_span("prefect_flow") as flow_span:
        print(f"Prefect flow trace ID: {flow_span.get_span_context().trace_id:032x}")
        
        # Prefect task
        with tracer.start_as_current_span("prefect_task"):
            # Call a Laminar-instrumented function
            @observe(name="ai_processing")
            def call_llm():
                current = trace.get_current_span()
                print(f"Laminar span trace ID: {current.get_span_context().trace_id:032x}")
                # Your LLM call here
                return "ai_result"
            
            result = call_llm()
    
    print("Notice: All spans share the same trace ID - they're part of the same trace\n")
    Laminar.shutdown()


def example_mixed_usage():
    """Example showing mixed usage - some spans isolated, some shared."""
    print("=== Example 3: Mixed Usage ===")
    
    # Initialize with context sharing enabled
    Laminar.initialize(
        project_api_key="your-api-key",
        respect_existing_context=True,
    )
    
    # Laminar span without external context (creates new trace)
    @observe(name="standalone_operation")
    def standalone():
        current = trace.get_current_span()
        print(f"Standalone trace ID: {current.get_span_context().trace_id:032x}")
    
    standalone()
    
    # Now within external context
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    tracer = provider.get_tracer("my_app")
    
    with tracer.start_as_current_span("app_request") as app_span:
        print(f"App trace ID: {app_span.get_span_context().trace_id:032x}")
        
        @observe(name="shared_operation")
        def shared():
            current = trace.get_current_span()
            print(f"Shared trace ID: {current.get_span_context().trace_id:032x}")
        
        shared()
    
    print("Notice: Standalone creates its own trace, shared joins the app trace\n")
    Laminar.shutdown()


if __name__ == "__main__":
    # Run all examples
    example_isolated_context()
    example_shared_context()
    example_mixed_usage()
    
    print("""
Key Points:
1. Use respect_existing_context=True to integrate with other OTEL-instrumented frameworks
2. This allows you to see LLM calls in the context of your larger application flow
3. Particularly useful with orchestration frameworks like Prefect, Dagster, Airflow
4. Also works with web frameworks like FastAPI, Flask, Django that have OTEL support
""")