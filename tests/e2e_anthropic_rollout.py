"""
E2E test for Anthropic rollout feature.

This test demonstrates:
1. Using the @observe decorator with rollout_entrypoint=True
2. Anthropic messages.create instrumented with Laminar
3. Rollout mode with session tracking

Usage:
    python tests/e2e_anthropic_rollout.py
"""

import os

from anthropic import Anthropic
from lmnr import Laminar, observe


Laminar.initialize(
    base_url="http://localhost",
    http_port=8000,
    grpc_port=8001,
    project_api_key=os.environ.get("LMNR_PROJECT_API_KEY_TEST"),
)

client = Anthropic()


@observe(name="anthropic_rollout_test", rollout_entrypoint=True)
def run_anthropic_test(query: str) -> str:
    """Test function that calls Anthropic API and returns the response."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        system="You are a helpful assistant. Keep your responses concise.",
        messages=[
            {"role": "user", "content": query}
        ],
    )
    return response.content[0].text


if __name__ == "__main__":
    result = run_anthropic_test("What is 2 + 2? Answer in one sentence.")
    print(f"Result: {result}")
    Laminar.shutdown()
    print("Done! Check the Laminar dashboard for the trace.")
