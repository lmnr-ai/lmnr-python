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
    register_initializers({
        Instruments.ALEPHALPHA: initializers.AlephAlphaInstrumentorInitializer(),
        Instruments.ANTHROPIC: initializers.AnthropicInstrumentorInitializer(),
        Instruments.BEDROCK: initializers.BedrockInstrumentorInitializer(),
        Instruments.BROWSER_USE: initializers.BrowserUseInstrumentorInitializer(),
        Instruments.BROWSER_USE_SESSION: initializers.BrowserUseSessionInstrumentorInitializer(),
        Instruments.BUBUS: initializers.BubusInstrumentorInitializer(),
        Instruments.CHROMA: initializers.ChromaInstrumentorInitializer(),
        Instruments.CLAUDE_AGENT: initializers.ClaudeAgentInstrumentorInitializer(),
        Instruments.COHERE: initializers.CohereInstrumentorInitializer(),
        Instruments.CREWAI: initializers.CrewAIInstrumentorInitializer(),
        Instruments.CUA_AGENT: initializers.CuaAgentInstrumentorInitializer(),
        Instruments.CUA_COMPUTER: initializers.CuaComputerInstrumentorInitializer(),
        Instruments.DAYTONA_SDK: initializers.DaytonaSDKInstrumentorInitializer(),
        Instruments.DEEPAGENTS: initializers.DeepagentsInstrumentorInitializer(),
        Instruments.GOOGLE_ADK: initializers.GoogleADKInstrumentorInitializer(),
        Instruments.GOOGLE_GENAI: initializers.GoogleGenAIInstrumentorInitializer(),
        Instruments.GROQ: initializers.GroqInstrumentorInitializer(),
        Instruments.HAYSTACK: initializers.HaystackInstrumentorInitializer(),
        Instruments.KERNEL: initializers.KernelInstrumentorInitializer(),
        Instruments.LANCEDB: initializers.LanceDBInstrumentorInitializer(),
        Instruments.LANGCHAIN: initializers.LangchainInstrumentorInitializer(),
        Instruments.LANGFUSE: initializers.LangfuseInstrumentorInitializer(),
        Instruments.LANGGRAPH: initializers.LanggraphInstrumentorInitializer(),
        Instruments.LITELLM: initializers.LitellmInstrumentorInitializer(),
        Instruments.LLAMA_INDEX: initializers.LlamaIndexInstrumentorInitializer(),
        Instruments.MARQO: initializers.MarqoInstrumentorInitializer(),
        Instruments.MCP: initializers.MCPInstrumentorInitializer(),
        Instruments.MILVUS: initializers.MilvusInstrumentorInitializer(),
        Instruments.MISTRAL: initializers.MistralInstrumentorInitializer(),
        Instruments.OLLAMA: initializers.OllamaInstrumentorInitializer(),
        Instruments.OPENAI: initializers.OpenAIInstrumentorInitializer(),
        Instruments.OPENAI_AGENTS: initializers.OpenAIAgentsInstrumentorInitializer(),
        Instruments.OPENTELEMETRY: initializers.OpenTelemetryInstrumentorInitializer(),
        Instruments.PATCHRIGHT: initializers.PatchrightInstrumentorInitializer(),
        Instruments.PINECONE: initializers.PineconeInstrumentorInitializer(),
        Instruments.PLAYWRIGHT: initializers.PlaywrightInstrumentorInitializer(),
        Instruments.PYDANTIC_AI: initializers.PydanticAIInstrumentorInitializer(),
        Instruments.QDRANT: initializers.QdrantInstrumentorInitializer(),
        Instruments.REPLICATE: initializers.ReplicateInstrumentorInitializer(),
        Instruments.SAGEMAKER: initializers.SageMakerInstrumentorInitializer(),
        Instruments.SKYVERN: initializers.SkyvernInstrumentorInitializer(),
        Instruments.TEMPORAL: initializers.TemporalInstrumentorInitializer(),
        Instruments.TOGETHER: initializers.TogetherInstrumentorInitializer(),
        Instruments.TRANSFORMERS: initializers.TransformersInstrumentorInitializer(),
        Instruments.VERTEXAI: initializers.VertexAIInstrumentorInitializer(),
        Instruments.WATSONX: initializers.WatsonxInstrumentorInitializer(),
        Instruments.WEAVIATE: initializers.WeaviateInstrumentorInitializer(),
    })
