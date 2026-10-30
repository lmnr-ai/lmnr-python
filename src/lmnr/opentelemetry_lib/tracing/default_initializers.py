"""The built-in `Instruments` -> initializer registry.

Deliberately NOT imported by `tracing/instruments.py`: the initializers import
every instrumentor, which import `Laminar`, which imports `instruments` — an
import cycle. Instead `lmnr/__init__.py` calls `register_default_initializers()`
once, from outside that loop, and `init_instrumentations` reads the registry.
"""

import lmnr.opentelemetry_lib.tracing._instrument_initializers as initializers
from lmnr.opentelemetry_lib.tracing.instruments import (
    Instruments,
    register_initializers,
)


def register_default_initializers() -> None:
    register_initializers(
        {
            Instruments.ANTHROPIC: initializers.AnthropicInstrumentorInitializer(),
            Instruments.BROWSER_USE: initializers.BrowserUseInstrumentorInitializer(),
            Instruments.BROWSER_USE_SESSION: initializers.BrowserUseSessionInstrumentorInitializer(),
            Instruments.BUBUS: initializers.BubusInstrumentorInitializer(),
            Instruments.CLAUDE_AGENT: initializers.ClaudeAgentInstrumentorInitializer(),
            Instruments.COHERE: initializers.CohereInstrumentorInitializer(),
            Instruments.CUA_AGENT: initializers.CuaAgentInstrumentorInitializer(),
            Instruments.CUA_COMPUTER: initializers.CuaComputerInstrumentorInitializer(),
            Instruments.DAYTONA_SDK: initializers.DaytonaSDKInstrumentorInitializer(),
            Instruments.DEEPAGENTS: initializers.DeepagentsInstrumentorInitializer(),
            Instruments.GOOGLE_ADK: initializers.GoogleADKInstrumentorInitializer(),
            Instruments.GOOGLE_AIPLATFORM: initializers.GoogleAIPlatformInstrumentorInitializer(),
            Instruments.GOOGLE_GENAI: initializers.GoogleGenAIInstrumentorInitializer(),
            Instruments.GROQ: initializers.GroqInstrumentorInitializer(),
            Instruments.KERNEL: initializers.KernelInstrumentorInitializer(),
            Instruments.LANGCHAIN: initializers.LangchainInstrumentorInitializer(),
            Instruments.LANGFUSE: initializers.LangfuseInstrumentorInitializer(),
            Instruments.LANGGRAPH: initializers.LanggraphInstrumentorInitializer(),
            Instruments.LITELLM: initializers.LitellmInstrumentorInitializer(),
            Instruments.MCP: initializers.MCPInstrumentorInitializer(),
            Instruments.MICROSOFT_AGENT_FRAMEWORK: (
                initializers.MicrosoftAgentFrameworkInstrumentorInitializer()
            ),
            Instruments.OPENAI: initializers.OpenAIInstrumentorInitializer(),
            Instruments.OPENAI_AGENTS: initializers.OpenAIAgentsInstrumentorInitializer(),
            Instruments.OPENROUTER: initializers.OpenRouterInstrumentorInitializer(),
            Instruments.OPENTELEMETRY: initializers.OpenTelemetryInstrumentorInitializer(),
            Instruments.PATCHRIGHT: initializers.PatchrightInstrumentorInitializer(),
            Instruments.PLAYWRIGHT: initializers.PlaywrightInstrumentorInitializer(),
            Instruments.PYDANTIC_AI: initializers.PydanticAIInstrumentorInitializer(),
            Instruments.SKYVERN: initializers.SkyvernInstrumentorInitializer(),
            Instruments.TEMPORAL: initializers.TemporalInstrumentorInitializer(),
            Instruments.TYPESAFE: initializers.TypeSafeInstrumentorInitializer(),
        }
    )
