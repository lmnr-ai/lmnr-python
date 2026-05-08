import logging

from enum import Enum
from typing import TYPE_CHECKING

from opentelemetry.trace import TracerProvider
import lmnr.opentelemetry_lib.tracing._instrument_initializers as initializers
from lmnr.opentelemetry_lib.utils.package_check import (
    get_package_version,
    is_package_installed,
)
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient

if TYPE_CHECKING:
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk.trace import SpanProcessor

module_logger = logging.getLogger(__name__)


class Instruments(Enum):
    # The list of libraries which will be autoinstrumented
    # if no specific instruments are provided to initialize()
    ALEPHALPHA = "alephalpha"
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    BROWSER_USE = "browser_use"
    BROWSER_USE_SESSION = "browser_use_session"
    BUBUS = "bubus"
    CHROMA = "chroma"
    CLAUDE_AGENT = "claude_agent"
    COHERE = "cohere"
    CREWAI = "crewai"
    CUA_AGENT = "cua_agent"
    CUA_COMPUTER = "cua_computer"
    DAYTONA_SDK = "daytona_sdk"
    DEEPAGENTS = "deepagents"
    GOOGLE_GENAI = "google_genai"
    GROQ = "groq"
    HAYSTACK = "haystack"
    KERNEL = "kernel"
    LANCEDB = "lancedb"
    LANGCHAIN = "langchain"
    # Auto-enabled by default when `langfuse` (>= 3.0) is installed. Langfuse's
    # own SDK is built on OpenTelemetry; this instrumentor dual-attaches
    # Laminar's span processor to every Langfuse `TracerProvider` so spans
    # emitted by `@observe`, `langfuse.openai`, and `langfuse.langchain` flow
    # into Laminar in addition to Langfuse. See
    # `_LANGFUSE_PROVIDER_CONFLICTS` for the raw-provider instrumentors that
    # are auto-removed to avoid duplicate spans.
    LANGFUSE = "langfuse"
    LANGGRAPH = "langgraph"
    LITELLM = "litellm"
    LLAMA_INDEX = "llama_index"
    MARQO = "marqo"
    MCP = "mcp"
    MILVUS = "milvus"
    MISTRAL = "mistral"
    OLLAMA = "ollama"
    OPENAI = "openai"
    OPENAI_AGENTS = "openai_agents"
    # Patch OpenTelemetry to fix DataDog's broken Span context
    # See lmnr.opentelemetry_lib.opentelemetry.instrumentation.opentelemetry
    # for more details.
    OPENTELEMETRY = "opentelemetry"
    ###
    PATCHRIGHT = "patchright"
    PINECONE = "pinecone"
    PLAYWRIGHT = "playwright"
    # Auto-enabled by default when `pydantic-ai-slim`/`pydantic-ai` is installed.
    # pydantic_ai emits its own OTel GenAI spans at the model abstraction layer,
    # so when it is auto-enabled we also auto-remove the overlapping raw provider
    # instrumentors (OpenAI, Anthropic, etc. — see _PYDANTIC_AI_PROVIDER_CONFLICTS)
    # from the default set to avoid duplicate spans. Users who need the raw
    # providers alongside pydantic_ai should pass an explicit `instruments`
    # set to `Laminar.initialize`.
    PYDANTIC_AI = "pydantic_ai"
    QDRANT = "qdrant"
    REPLICATE = "replicate"
    SAGEMAKER = "sagemaker"
    SKYVERN = "skyvern"
    TEMPORAL = "temporal"
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
}


#: Provider instrumentors that would produce spans overlapping with pydantic_ai's
#: own GenAI spans. When PYDANTIC_AI is auto-enabled (pydantic_ai is installed
#: and the caller didn't pass an explicit `instruments` set), these are removed
#: from the default set so the same model call isn't traced twice.
_PYDANTIC_AI_PROVIDER_CONFLICTS: set[Instruments] = {
    Instruments.ANTHROPIC,
    Instruments.BEDROCK,
    Instruments.COHERE,
    Instruments.GOOGLE_GENAI,
    Instruments.GROQ,
    Instruments.MISTRAL,
    Instruments.OPENAI,
}


#: When deepagents is installed, the Laminar `DeepagentsInstrumentor` (via
#: `LaminarMiddleware`) already emits the spans users actually care about —
#: one DEFAULT root span per agent invocation and one TOOL span per tool
#: call. The LangChain / LangGraph auto-instrumentors would otherwise add a
#: large number of langsmith-style internal-node spans on top of that, which
#: clutters the transcript without adding signal. Auto-remove them from the
#: default set; callers who want the raw spans can pass an explicit
#: `instruments` set.
_DEEPAGENTS_NOISE_CONFLICTS: set[Instruments] = {
    Instruments.LANGCHAIN,
    Instruments.LANGGRAPH,
}


#: Instrumentors that overlap with Langfuse's own wrappers. Langfuse emits
#: GenAI / tool spans through `langfuse.openai`, `langfuse.langchain`, and
#: its `@observe` decorator for everything the user routes through it. When
#: LANGFUSE is auto-enabled, running Laminar's raw-provider instrumentors on
#: top of that would produce duplicate spans for the same call.
_LANGFUSE_PROVIDER_CONFLICTS: set[Instruments] = {
    Instruments.ANTHROPIC,
    Instruments.BEDROCK,
    Instruments.COHERE,
    Instruments.GOOGLE_GENAI,
    Instruments.GROQ,
    Instruments.LANGCHAIN,
    Instruments.MISTRAL,
    Instruments.OPENAI,
}


def _pydantic_ai_installed() -> bool:
    return is_package_installed("pydantic-ai-slim") or is_package_installed(
        "pydantic-ai"
    )


def _deepagents_installed() -> bool:
    return is_package_installed("deepagents")


def _langfuse_installed() -> bool:
    """The bridge targets langfuse >= 3.0 (OTel-native). For langfuse 2.x we
    must report False so `_LANGFUSE_PROVIDER_CONFLICTS` does NOT strip the
    raw-provider instrumentors — the bridge initializer would then return None
    for 2.x and leave those providers silently uninstrumented."""
    if not is_package_installed("langfuse"):
        return False
    from packaging.version import InvalidVersion, parse

    version = get_package_version("langfuse")
    if version is None:
        # Present but unreadable version metadata — err on the side of not
        # stripping providers.
        return False
    try:
        return parse(version) >= parse("3.0.0")
    except InvalidVersion:
        return False


def init_instrumentations(
    tracer_provider: TracerProvider,
    logger_provider: "LoggerProvider | None" = None,
    instruments: set[Instruments] | None = None,
    block_instruments: set[Instruments] | None = None,
    async_client: AsyncLaminarClient | None = None,
    lmnr_span_processor: "SpanProcessor | None" = None,
):
    block_instruments = block_instruments or set()
    if instruments is None:
        instruments = set(Instruments)
        deepagents_active = (
            _deepagents_installed() and Instruments.DEEPAGENTS not in block_instruments
        )
        langfuse_active = (
            _langfuse_installed() and Instruments.LANGFUSE not in block_instruments
        )
        # Only auto-enable PYDANTIC_AI if the package is actually installed,
        # and only auto-remove overlapping provider instrumentors in that case.
        # If pydantic_ai isn't installed, PYDANTIC_AI stays in the set but its
        # initializer will short-circuit to None (see
        # PydanticAIInstrumentorInitializer).
        if (
            _pydantic_ai_installed()
            and Instruments.PYDANTIC_AI not in block_instruments
        ):
            # Deepagents wins over pydantic_ai when both are installed: the
            # deepagents instrumentation relies on the raw-provider
            # instrumentors (Anthropic / OpenAI / …) to emit LLM spans
            # underneath each tool call, so stripping them would leave the
            # `deep_agent` trace with only root + tool spans and no LLM
            # children. If the same app also uses pydantic_ai Agents,
            # provider calls on that path will be traced twice; callers who
            # want pydantic_ai's de-duplication can pass an explicit
            # `instruments` set to `Laminar.initialize`.
            if not deepagents_active:
                module_logger.warning(
                    "Not enabling default LLM instrumentations to avoid double "
                    + "instrumentation with Pydantic AI. To opt-in, pass the following "
                    + "to Laminar.initialize() instruments=[Instruments.ANTHROPIC, "
                    + "Instruments.BEDROCK, Instruments.COHERE, Instruments.GOOGLE_GENAI, "
                    + "Instruments.GROQ, Instruments.MISTRAL, Instruments.OPENAI]"
                )
                instruments = instruments - _PYDANTIC_AI_PROVIDER_CONFLICTS
        else:
            instruments = instruments - {Instruments.PYDANTIC_AI}
        # Auto-remove LangChain/LangGraph noise when deepagents is present;
        # LaminarMiddleware emits the relevant spans at the agent boundary.
        if deepagents_active:
            instruments = instruments - _DEEPAGENTS_NOISE_CONFLICTS
        else:
            instruments = instruments - {Instruments.DEEPAGENTS}
        # Auto-remove raw-provider instrumentors when Langfuse is present —
        # Langfuse wraps OpenAI, LangChain, etc. itself, and the bridge
        # dual-attaches Laminar's SpanProcessor to every Langfuse
        # TracerProvider so those langfuse-emitted spans also reach Laminar.
        # Langfuse wins over both deepagents and pydantic_ai: if the user
        # installed Langfuse they chose it as their trace surface, and the
        # goal of the bridge is to route spans to both Langfuse *and* Laminar
        # through Langfuse's own auto-patchers. Running Laminar's raw-provider
        # instrumentors alongside `langfuse.openai` / `@observe` / pydantic_ai
        # would double-cover the same call. Deepagents' DEFAULT + TOOL spans
        # still come from `DeepagentsInstrumentor` (Langfuse doesn't emit
        # those), and their LLM children come from Langfuse's auto-patchers
        # riding through the bridge. Callers who want the raw-provider
        # instrumentors alongside Langfuse can pass an explicit `instruments`
        # set to `Laminar.initialize`.
        if langfuse_active:
            instruments = instruments - _LANGFUSE_PROVIDER_CONFLICTS
            instruments = instruments - {Instruments.PYDANTIC_AI}
        else:
            instruments = instruments - {Instruments.LANGFUSE}
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
            instrumentor = initializer.init_instrumentor(async_client)
            if instrumentor is None:
                continue
            # LangfuseInstrumentor doesn't extend BaseInstrumentor; it needs
            # access to the Laminar SpanProcessor so it can dual-attach that
            # same processor onto Langfuse-owned TracerProviders.
            if instrument == Instruments.LANGFUSE:
                if lmnr_span_processor is None:
                    module_logger.debug(
                        "Skipping Langfuse bridge: no Laminar SpanProcessor provided."
                    )
                    continue
                instrumentor.instrument(
                    lmnr_tracer_provider=tracer_provider,
                    lmnr_span_processor=lmnr_span_processor,
                )
                continue
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument(
                    tracer_provider=tracer_provider,
                    logger_provider=logger_provider,
                )
        except Exception as e:
            if "No module named 'langchain_community'" in str(e):
                # LangChain instrumentor does not require langchain_community,
                # but throws this error if it's not installed.
                continue
            module_logger.error(f"Error initializing instrumentor: {e}")
            continue
