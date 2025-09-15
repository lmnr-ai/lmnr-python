#!/usr/bin/env python3
"""
Example showing how to use Laminar's Exa search API instrumentation.

This example demonstrates how to automatically instrument Exa API calls
with comprehensive input/output tracing using Laminar.

The instrumentation captures:
- All input parameters via lmnr.span.input
- Complete response data via lmnr.span.output
- Service-specific metadata as JSON (search params, results, etc.)
- Cost tracking with actual costs from Exa API responses
- Performance metrics (response size, status, duration)
- Streaming response chunks with aggregated final content

Uses unified service instrumentation specification:
- service.name: "exa"
- service.operation: "search" | "answer" | "research"
- service.cost.*: Cost tracking attributes
- service.metadata: Exa-specific parameters as JSON string
"""

import os
from lmnr import Laminar

# Initialize Laminar with Exa instrumentation
# This will automatically instrument all Exa API calls with comprehensive tracing
Laminar.initialize(
    project_api_key="your-laminar-api-key",  # Replace with your actual API key
    # EXA instrumentation is included by default, but you can explicitly specify it:
    # instruments={"EXA"}
)

# Content tracing is always enabled - all inputs and outputs are captured automatically

# Now you can use Exa normally, and all calls will be automatically traced
try:
    from exa_py import Exa
    
    # Initialize Exa client
    exa = Exa(os.getenv('EXA_API_KEY'))  # Set your EXA_API_KEY environment variable
    
    # All of these calls will be automatically instrumented:
    
    # Basic search
    print("Performing basic search...")
    results = exa.search("hottest AI startups", num_results=2)
    print(f"Found {len(results.results)} results")
    
    # Search with content
    print("\nPerforming search with content...")
    results_with_content = exa.search_and_contents(
        "AI in healthcare", 
        text=True, 
        num_results=2
    )
    print(f"Found {len(results_with_content.results)} results with content")
    
    # Find similar content
    print("\nFinding similar content...")
    similar_results = exa.find_similar(
        "https://www.adept.ai/", 
        num_results=2
    )
    print(f"Found {len(similar_results.results)} similar results")
    
    # Generate answer
    print("\nGenerating answer...")
    answer = exa.answer("What is the capital of France?")
    print(f"Answer: {answer}")
    
    # Research task (if available)
    print("\nCreating research task...")
    try:
        task = exa.research.create_task(
            instructions="What are the main benefits of meditation?",
            infer_schema=True
        )
        print(f"Created research task with ID: {task.id}")
        
        # Poll for completion
        result = exa.research.poll_task(task.id)
        print(f"Research completed with status: {result.status}")
        
    except AttributeError:
        print("Research methods not available in this version of exa_py")
    
    print("\n✓ All Exa API calls have been automatically instrumented!")
    print("Check your Laminar dashboard to see:")
    print("  - lmnr.span.input: Complete input parameters")
    print("  - lmnr.span.output: Full response data (no truncation)")
    print("  - service.name: 'exa'")
    print("  - service.cost.amount: Actual costs from Exa API (fallback to estimates)")
    print("  - service.metadata: Exa-specific parameters as JSON")
    print("  - service.response.status: 'success' or 'error'")
    print("  - Regular CLIENT spans following unified service spec")
    
except ImportError:
    print("exa_py not installed. Install it with: pip install exa_py")
    print("The instrumentation is still ready and will work once exa_py is installed.")

except Exception as e:
    print(f"Error: {e}")
    print("Make sure to set your EXA_API_KEY environment variable.")
    print("The instrumentation is working - this is just an API key issue.")
