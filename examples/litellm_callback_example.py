#!/usr/bin/env python3
"""
Example showing how to use LaminarLiteLLMLogger with LiteLLM callbacks.

This demonstrates the manual setup approach where users add the callback themselves.
"""

import os
import asyncio

# Setup environment (replace with your actual API key)
os.environ["LMNR_PROJECT_API_KEY"] = "your-project-api-key-here"
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"  # Or other provider API key

# Import and initialize Laminar tracing
from lmnr import Laminar

Laminar.initialize(project_api_key=os.environ["LMNR_PROJECT_API_KEY"])

try:
    # Import LiteLLM and the Laminar callback
    import litellm
    from lmnr import (
        LaminarLiteLLMCallback,
    )

    # Create and register the callback
    laminar_callback = LaminarLiteLLMCallback()
    litellm.callbacks = [laminar_callback]

    print("✓ LiteLLM callback registered successfully!")

    def test_sync_completion():
        """Test synchronous completion with tracing"""
        print("\n🔄 Testing synchronous completion...")

        try:
            response = litellm.completion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello! How are you today?"}],
                temperature=0.7,
                max_tokens=50,
            )
            print(f"✓ Success: {response.choices[0].message.content}")
        except Exception as e:
            print(f"✗ Error: {e}")

    async def test_async_completion():
        """Test asynchronous completion with tracing"""
        print("\n🔄 Testing asynchronous completion...")

        try:
            response = await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "What's the weather like?"}],
                temperature=0.8,
                max_tokens=50,
            )
            print(f"✓ Success: {response.choices[0].message.content}")
        except Exception as e:
            print(f"✗ Error: {e}")

    def test_failure_case():
        """Test error handling with tracing"""
        print("\n🔄 Testing error case...")

        try:
            response = litellm.completion(
                model="invalid-model-name",
                messages=[{"role": "user", "content": "This should fail"}],
            )
        except Exception as e:
            print(f"✓ Expected error caught: {e}")

    if __name__ == "__main__":
        print("LiteLLM + Laminar Integration Example")
        print("=" * 40)

        # Run tests
        test_sync_completion()

        print("\n" + "-" * 20)
        asyncio.run(test_async_completion())

        print("\n" + "-" * 20)
        test_failure_case()

        print("\n" + "=" * 40)
        print("✓ All tests completed!")
        print("Check your Laminar dashboard to see the traced LiteLLM calls.")
        print("Spans should appear with timing, model info, and usage data.")

except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nTo use this example, install the required dependencies:")
    print("  pip install lmnr litellm")
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    raise
