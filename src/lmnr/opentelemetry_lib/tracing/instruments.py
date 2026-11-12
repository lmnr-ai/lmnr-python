import logging
from collections.abc import Mapping
from enum import Enum
from typing import TYPE_CHECKING

from opentelemetry.trace import TracerProvider

from lmnr.opentelemetry_lib.tracing.instrumentor_initializer import (
    InstrumentorInitializer,
)
from lmnr.opentelemetry_lib.utils.package_check import (
    is_package_installed,
)
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient

if TYPE_CHECKING:
    from opentelemetry.sdk._logs import LoggerProvider

module_logger = logging.getLogger(__name__)


class Instruments(Enum):
    # The list of libraries which will be autoinstrumented
    # if no specific instruments are provided to initialize()
    ANTHROPIC = "anthropic"
    BROWSER_USE = "browser_use"
    BROWSER_USE_SESSION = "browser_use_session"
    BUBUS = "bubus"
    CLAUDE_AGENT = "claude_agent"
    COHERE = "cohere"
    CUA_AGENT = "cua_agent"
    CUA_COMPUTER = "cua_computer"
    DAYTONA_SDK = "daytona_sdk"
    DEEPAGENTS = "deepagents"
    # When google-adk is installed, GOOGLE_GENAI is auto-removed from the
    # default set (see _GOOGLE_ADK_GENAI_CONFLICTS): ADK self-instruments its
    # own `call_llm` span, which this instrumentor enriches directly from the
    # real request/response objects, so wrapping the genai SDK's own
    # `generate_content` on top would both double-cover the call and produce
    # a span whose parent depends on ADK's fragile generator-scoped context.
    GOOGLE_ADK = "google_adk"
    # Instruments the `vertexai` namespace of `google-cloud-aiplatform`.
    GOOGLE_AIPLATFORM = "google_aiplatform"
    GOOGLE_GENAI = "google_genai"
    GROQ = "groq"
    KERNEL = "kernel"
    LANGCHAIN = "langchain"
    LANGGRAPH = "langgraph"
    LITELLM = "litellm"
    MCP = "mcp"
    MICROSOFT_AGENT_FRAMEWORK = "microsoft_agent_framework"
    OPENAI = "openai"
    OPENAI_AGENTS = "openai_agents"
    OPENROUTER = "openrouter"
    # Patch OpenTelemetry to fix DataDog's broken Span context
    # See lmnr.opentelemetry_lib.opentelemetry.instrumentation.opentelemetry
    # for more details.
    OPENTELEMETRY = "opentelemetry"
    ###
    PATCHRIGHT = "patchright"
    PLAYWRIGHT = "playwright"
    # Auto-enabled by default when `pydantic-ai-slim`/`pydantic-ai` is installed.
    # pydantic_ai emits its own OTel GenAI spans at the model abstraction layer,
    # so when it is auto-enabled we also auto-remove the overlapping raw provider
    # instrumentors (OpenAI, Anthropic, etc. — see _PYDANTIC_AI_PROVIDER_CONFLICTS)
    # from the default set to avoid duplicate spans. Users who need the raw
    # providers alongside pydantic_ai should pass an explicit `instruments`
    # set to `Laminar.initialize`.
    PYDANTIC_AI = "pydantic_ai"
    SKYVERN = "skyvern"
    TEMPORAL = "temporal"
    TYPESAFE = "typesafe"
    # Legacy name for GOOGLE_AIPLATFORM, kept so existing `instruments=` /
    # `disabled_instruments=` sets keep working. Same value => enum alias.
    VERTEXAI = "google_aiplatform"  # noqa: PIE796


#: `Instruments` -> initializer. Populated by `register_initializers` (the
#: built-ins via `default_initializers.register_default_initializers`, called
#: from `lmnr/__init__.py`), so this module never imports the concrete
#: initializers and stays out of the import cycle through the instrumentors.
INSTRUMENTATION_INITIALIZERS: dict[Instruments, InstrumentorInitializer] = {}


def register_initializers(
    initializers: Mapping[Instruments, InstrumentorInitializer],
) -> None:
    INSTRUMENTATION_INITIALIZERS.update(initializers)


#: Provider instrumentors that would produce spans overlapping with pydantic_ai's
#: own GenAI spans. When PYDANTIC_AI is auto-enabled (pydantic_ai is installed
#: and the caller didn't pass an explicit `instruments` set), these are removed
#: from the default set so the same model call isn't traced twice.
_PYDANTIC_AI_PROVIDER_CONFLICTS: set[Instruments] = {
    Instruments.ANTHROPIC,
    Instruments.COHERE,
    Instruments.GOOGLE_GENAI,
    Instruments.GROQ,
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


#: When google-adk is installed, ADK self-instruments its own `call_llm`
#: span and the `GoogleAdkInstrumentor` enriches it directly from the real
#: request/response objects. Wrapping the genai SDK's own `generate_content`
#: on top of that would both double-cover the call and produce a span whose
#: parent depends on ADK's fragile generator-scoped context (its span can
#: outlive the `yield` that hands control back to the caller). Auto-remove
#: GOOGLE_GENAI from the default set in that case; callers who want the raw
#: genai spans alongside ADK can pass an explicit `instruments` set.
_GOOGLE_ADK_GENAI_CONFLICTS: set[Instruments] = {
    Instruments.GOOGLE_GENAI,
}


def _pydantic_ai_installed() -> bool:
    return is_package_installed("pydantic-ai-slim") or is_package_installed("pydantic-ai")


def _deepagents_installed() -> bool:
    return is_package_installed("deepagents")


def _google_adk_installed() -> bool:
    return is_package_installed("google-adk")


def init_instrumentations(
    tracer_provider: TracerProvider,
    logger_provider: "LoggerProvider | None" = None,
    instruments: set[Instruments] | None = None,
    block_instruments: set[Instruments] | None = None,
    async_client: AsyncLaminarClient | None = None,
):
    if not INSTRUMENTATION_INITIALIZERS:
        # Filled by `lmnr/__init__.py` (see `register_initializers`). Empty means
        # this ran while the package was still importing, before that call.
        module_logger.error(
            "No instrumentation initializers are registered, so nothing will be "
            + "auto-instrumented. Is Laminar being initialized while `lmnr` is "
            + "still being imported?"
        )
        return
    block_instruments = block_instruments or set()
    if instruments is None:
        instruments = set(Instruments)
        deepagents_active = (
            _deepagents_installed() and Instruments.DEEPAGENTS not in block_instruments
        )
        # Only auto-enable PYDANTIC_AI if the package is actually installed,
        # and only auto-remove overlapping provider instrumentors in that case.
        # If pydantic_ai isn't installed, PYDANTIC_AI stays in the set but its
        # initializer will short-circuit to None (see
        # PydanticAIInstrumentorInitializer).
        if _pydantic_ai_installed() and Instruments.PYDANTIC_AI not in block_instruments:
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
                # Only emitted when the providers are actually stripped —
                # with deepagents active they stay enabled, so warning here
                # would misstate what happened and misdirect debugging.
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
        # Only auto-remove GOOGLE_GENAI if google-adk is actually installed
        # (mirrors the pydantic_ai handling above).
        if _google_adk_installed() and Instruments.GOOGLE_ADK not in block_instruments:
            module_logger.warning(
                "Not enabling the default Google GenAI SDK instrumentation to "
                + "avoid double instrumentation with Google ADK. To opt-in, pass "
                + "an explicit instruments set to Laminar.initialize() that "
                + "includes both Instruments.GOOGLE_ADK and "
                + "Instruments.GOOGLE_GENAI."
            )
            instruments = instruments - _GOOGLE_ADK_GENAI_CONFLICTS
    if not isinstance(instruments, set):  # pyright: ignore[reportUnnecessaryIsInstance]
        instruments = set(instruments)  # pyright: ignore[reportUnreachable]

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
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument(
                    tracer_provider=tracer_provider,
                    logger_provider=logger_provider,
                )
        except Exception as e:
            module_logger.error(f"Error initializing instrumentor: {e}")
            continue
