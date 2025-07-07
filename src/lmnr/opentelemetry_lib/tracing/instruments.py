import logging

from enum import Enum

from opentelemetry.trace import TracerProvider
import lmnr.opentelemetry_lib.tracing._instrument_initializers as initializers
from lmnr.sdk.client.synchronous.sync_client import LaminarClient
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient

module_logger = logging.getLogger(__name__)


class Instruments(Enum):
    # The list of libraries which will be autoinstrumented
    # if no specific instruments are provided to initialize()
    ALEPHALPHA = "alephalpha"
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    BROWSER_USE = "browser_use"
    CHROMA = "chroma"
    COHERE = "cohere"
    CREWAI = "crewai"
    GOOGLE_GENERATIVEAI = "google_generativeai"
    GOOGLE_GENAI = "google_genai"
    GROQ = "groq"
    HAYSTACK = "haystack"
    LANCEDB = "lancedb"
    LANGCHAIN = "langchain"
    LANGGRAPH = "langgraph"
    LLAMA_INDEX = "llama_index"
    MARQO = "marqo"
    MCP = "mcp"
    MILVUS = "milvus"
    MISTRAL = "mistral"
    OLLAMA = "ollama"
    OPENAI = "openai"
    # Patch OpenTelemetry to fix DataDog's broken Span context
    # See lmnr.opentelemetry_lib.opentelemetry.instrumentation.opentelemetry
    # for more details.
    OPENTELEMETRY = "opentelemetry"
    ###
    PATCHRIGHT = "patchright"
    PINECONE = "pinecone"
    PLAYWRIGHT = "playwright"
    QDRANT = "qdrant"
    REPLICATE = "replicate"
    SAGEMAKER = "sagemaker"
    SKYVERN = "skyvern"
    TOGETHER = "together"
    TRANSFORMERS = "transformers"
    VERTEXAI = "vertexai"
    WATSONX = "watsonx"
    WEAVIATE = "weaviate"


INSTRUMENTATION_INITIALIZERS: dict[
    Instruments, initializers.InstrumentorInitializer
] = {
    Instruments.ALEPHALPHA: initializers.AlephAlphaInstrumentorInitializer(),
    Instruments.ANTHROPIC: initializers.AnthropicInstrumentorInitializer(),
    Instruments.BEDROCK: initializers.BedrockInstrumentorInitializer(),
    Instruments.BROWSER_USE: initializers.BrowserUseInstrumentorInitializer(),
    Instruments.CHROMA: initializers.ChromaInstrumentorInitializer(),
    Instruments.COHERE: initializers.CohereInstrumentorInitializer(),
    Instruments.CREWAI: initializers.CrewAIInstrumentorInitializer(),
    Instruments.GOOGLE_GENERATIVEAI: initializers.GoogleGenerativeAIInstrumentorInitializer(),
    Instruments.GOOGLE_GENAI: initializers.GoogleGenAIInstrumentorInitializer(),
    Instruments.GROQ: initializers.GroqInstrumentorInitializer(),
    Instruments.HAYSTACK: initializers.HaystackInstrumentorInitializer(),
    Instruments.LANCEDB: initializers.LanceDBInstrumentorInitializer(),
    Instruments.LANGCHAIN: initializers.LangchainInstrumentorInitializer(),
    Instruments.LANGGRAPH: initializers.LanggraphInstrumentorInitializer(),
    Instruments.LLAMA_INDEX: initializers.LlamaIndexInstrumentorInitializer(),
    Instruments.MARQO: initializers.MarqoInstrumentorInitializer(),
    Instruments.MCP: initializers.MCPInstrumentorInitializer(),
    Instruments.MILVUS: initializers.MilvusInstrumentorInitializer(),
    Instruments.MISTRAL: initializers.MistralInstrumentorInitializer(),
    Instruments.OLLAMA: initializers.OllamaInstrumentorInitializer(),
    Instruments.OPENAI: initializers.OpenAIInstrumentorInitializer(),
    Instruments.OPENTELEMETRY: initializers.OpenTelemetryInstrumentorInitializer(),
    Instruments.PATCHRIGHT: initializers.PatchrightInstrumentorInitializer(),
    Instruments.PINECONE: initializers.PineconeInstrumentorInitializer(),
    Instruments.PLAYWRIGHT: initializers.PlaywrightInstrumentorInitializer(),
    Instruments.QDRANT: initializers.QdrantInstrumentorInitializer(),
    Instruments.REPLICATE: initializers.ReplicateInstrumentorInitializer(),
    Instruments.SAGEMAKER: initializers.SageMakerInstrumentorInitializer(),
    Instruments.SKYVERN: initializers.SkyvernInstrumentorInitializer(),
    Instruments.TOGETHER: initializers.TogetherInstrumentorInitializer(),
    Instruments.TRANSFORMERS: initializers.TransformersInstrumentorInitializer(),
    Instruments.VERTEXAI: initializers.VertexAIInstrumentorInitializer(),
    Instruments.WATSONX: initializers.WatsonxInstrumentorInitializer(),
    Instruments.WEAVIATE: initializers.WeaviateInstrumentorInitializer(),
}


def init_instrumentations(
    tracer_provider: TracerProvider,
    instruments: set[Instruments] | None = None,
    block_instruments: set[Instruments] | None = None,
    client: LaminarClient | None = None,
    async_client: AsyncLaminarClient | None = None,
):
    block_instruments = block_instruments or set()
    if instruments is None:
        instruments = set(Instruments)
    if not isinstance(instruments, set):
        instruments = set(instruments)

    # Remove any instruments that were explicitly blocked
    instruments = instruments - block_instruments

    for instrument in instruments:
        initializer = INSTRUMENTATION_INITIALIZERS.get(instrument)
        if initializer is None:
            module_logger.error(f"Invalid instrument: {instrument}")
            continue

        try:
            instrumentor = initializer.init_instrumentor(client, async_client)
            if instrumentor is None:
                continue
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument(tracer_provider=tracer_provider)
        except Exception as e:
            if "No module named 'langchain_community'" in str(e):
                # LangChain instrumentor does not require langchain_community,
                # but throws this error if it's not installed.
                continue
            module_logger.error(f"Error initializing instrumentor: {e}")
            continue
