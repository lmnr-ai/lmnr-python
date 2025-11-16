import abc

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

from lmnr.opentelemetry_lib.utils.package_check import (
    get_package_version,
    is_package_installed,
)


class InstrumentorInitializer(abc.ABC):
    @abc.abstractmethod
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        pass


class AlephAlphaInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("aleph_alpha_client"):
            return None
        if not is_package_installed("opentelemetry-instrumentation-alephalpha"):
            return None

        from opentelemetry.instrumentation.alephalpha import AlephAlphaInstrumentor

        return AlephAlphaInstrumentor()


class AnthropicInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("anthropic"):
            return None

        from ..opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

        return AnthropicInstrumentor(
            upload_base64_image=None,
        )


class BedrockInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("boto3"):
            return None
        if not is_package_installed("opentelemetry-instrumentation-bedrock"):
            return None

        from opentelemetry.instrumentation.bedrock import BedrockInstrumentor

        return BedrockInstrumentor()


class BrowserUseInstrumentorInitializer(InstrumentorInitializer):
    """Instruments for different versions of browser-use:

    - browser-use < 0.5: BrowserUseLegacyInstrumentor to track agent_step and
      other structure spans. Session instrumentation is controlled by
      Instruments.PLAYWRIGHT (or Instruments.PATCHRIGHT for several versions
      in 0.4.* that used patchright)
    - browser-use ~= 0.5: Structure spans live in browser_use package itself.
      Session instrumentation is controlled by Instruments.PLAYWRIGHT
    - browser-use >= 0.6.0rc1: BubusInstrumentor to keep spans structure.
      Session instrumentation is controlled by Instruments.BROWSER_USE_SESSION
    """

    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("browser-use"):
            return None

        version = get_package_version("browser-use")
        from packaging.version import parse

        if version and parse(version) < parse("0.5.0"):
            from lmnr.sdk.browser.browser_use_otel import BrowserUseLegacyInstrumentor

            return BrowserUseLegacyInstrumentor()

        return None


class BrowserUseSessionInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(
        self, async_client, *args, **kwargs
    ) -> BaseInstrumentor | None:
        if not is_package_installed("browser-use"):
            return None

        version = get_package_version("browser-use")
        from packaging.version import parse

        if version and parse(version) >= parse("0.6.0rc1"):
            from lmnr.sdk.browser.browser_use_cdp_otel import BrowserUseInstrumentor

            if async_client is None:
                return None

            return BrowserUseInstrumentor(async_client)

        return None


class BubusInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("bubus"):
            return None

        from lmnr.sdk.browser.bubus_otel import BubusInstrumentor

        return BubusInstrumentor()


class ChromaInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("chromadb"):
            return None
        if not is_package_installed("opentelemetry-instrumentation-chromadb"):
            return None

        from opentelemetry.instrumentation.chromadb import ChromaInstrumentor

        return ChromaInstrumentor()


class ClaudeAgentInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("claude-agent-sdk"):
            return None

        if not is_package_installed("lmnr-claude-code-proxy"):
            return None

        from ..opentelemetry.instrumentation.claude_agent import ClaudeAgentInstrumentor

        return ClaudeAgentInstrumentor()


class CohereInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("cohere"):
            return None
        if not is_package_installed("opentelemetry-instrumentation-cohere"):
            return None

        from opentelemetry.instrumentation.cohere import CohereInstrumentor

        return CohereInstrumentor()


class CrewAIInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("crewai"):
            return None
        if not is_package_installed("opentelemetry-instrumentation-crewai"):
            return None

        from opentelemetry.instrumentation.crewai import CrewAiInstrumentor

        return CrewAiInstrumentor()


class CuaAgentInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("cua-agent"):
            return None

        from ..opentelemetry.instrumentation.cua_agent import CuaAgentInstrumentor

        return CuaAgentInstrumentor()


class CuaComputerInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("cua-computer"):
            return None

        from ..opentelemetry.instrumentation.cua_computer import CuaComputerInstrumentor

        return CuaComputerInstrumentor()


class GoogleGenAIInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("google-genai"):
            return None

        from ..opentelemetry.instrumentation.google_genai import (
            GoogleGenAiSdkInstrumentor,
        )

        return GoogleGenAiSdkInstrumentor()


class GroqInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("groq"):
            return None

        from ..opentelemetry.instrumentation.groq import GroqInstrumentor

        return GroqInstrumentor()


class HaystackInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("haystack"):
            return None
        if not is_package_installed("opentelemetry-instrumentation-haystack"):
            return None

        from opentelemetry.instrumentation.haystack import HaystackInstrumentor

        return HaystackInstrumentor()


class KernelInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("kernel"):
            return None

        from ..opentelemetry.instrumentation.kernel import KernelInstrumentor

        return KernelInstrumentor()


class LanceDBInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("lancedb"):
            return None
        if not is_package_installed("opentelemetry-instrumentation-lancedb"):
            return None

        from opentelemetry.instrumentation.lancedb import LanceInstrumentor

        return LanceInstrumentor()


class LangchainInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("langchain"):
            return None
        if not is_package_installed("opentelemetry-instrumentation-langchain"):
            return None

        from opentelemetry.instrumentation.langchain import LangchainInstrumentor

        return LangchainInstrumentor()


class LanggraphInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("langgraph"):
            return None
        if not is_package_installed("langchain-core"):
            return None

        from ..opentelemetry.instrumentation.langgraph import LanggraphInstrumentor

        return LanggraphInstrumentor()


class LlamaIndexInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not (
            is_package_installed("llama-index") or is_package_installed("llama_index")
        ):
            return None
        if not is_package_installed("opentelemetry-instrumentation-llamaindex"):
            return None

        from opentelemetry.instrumentation.llamaindex import LlamaIndexInstrumentor

        return LlamaIndexInstrumentor()


class MarqoInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("marqo"):
            return None
        if not is_package_installed("opentelemetry-instrumentation-marqo"):
            return None

        from opentelemetry.instrumentation.marqo import MarqoInstrumentor

        return MarqoInstrumentor()


class MCPInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("mcp"):
            return None
        if not is_package_installed("opentelemetry-instrumentation-mcp"):
            return None

        from opentelemetry.instrumentation.mcp import McpInstrumentor

        return McpInstrumentor()


class MilvusInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("pymilvus"):
            return None
        if not is_package_installed("opentelemetry-instrumentation-milvus"):
            return None

        from opentelemetry.instrumentation.milvus import MilvusInstrumentor

        return MilvusInstrumentor()


class MistralInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("mistralai"):
            return None
        if not is_package_installed("opentelemetry-instrumentation-mistralai"):
            return None

        from opentelemetry.instrumentation.mistralai import MistralAiInstrumentor

        return MistralAiInstrumentor()


class OllamaInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("ollama"):
            return None
        if not is_package_installed("opentelemetry-instrumentation-ollama"):
            return None

        from opentelemetry.instrumentation.ollama import OllamaInstrumentor

        return OllamaInstrumentor()


class OpenAIInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("openai"):
            return None

        from ..opentelemetry.instrumentation.openai import OpenAIInstrumentor

        return OpenAIInstrumentor(
            # Default in the package provided is an empty function, which
            # results in dropping the image data if we don't explicitly
            # set it to None.
            upload_base64_image=None,
        )


class OpenHandsAIInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("openhands-ai"):
            return None
        from ..opentelemetry.instrumentation.openhands_ai import OpenHandsInstrumentor

        return OpenHandsInstrumentor()


class OpenTelemetryInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        from ..opentelemetry.instrumentation.opentelemetry import (
            OpentelemetryInstrumentor,
        )

        return OpentelemetryInstrumentor()


class PatchrightInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(
        self, async_client, *args, **kwargs
    ) -> BaseInstrumentor | None:
        if not is_package_installed("patchright"):
            return None

        from lmnr.sdk.browser.patchright_otel import PatchrightInstrumentor

        if async_client is None:
            return None

        return PatchrightInstrumentor(async_client)


class PineconeInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("pinecone"):
            return None
        if not is_package_installed("opentelemetry-instrumentation-pinecone"):
            return None

        from opentelemetry.instrumentation.pinecone import PineconeInstrumentor

        return PineconeInstrumentor()


class PlaywrightInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(
        self, async_client, *args, **kwargs
    ) -> BaseInstrumentor | None:
        if not is_package_installed("playwright"):
            return None

        from lmnr.sdk.browser.playwright_otel import PlaywrightInstrumentor

        if async_client is None:
            return None

        return PlaywrightInstrumentor(async_client)


class QdrantInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("qdrant_client"):
            return None
        if not is_package_installed("opentelemetry-instrumentation-qdrant"):
            return None

        from opentelemetry.instrumentation.qdrant import QdrantInstrumentor

        return QdrantInstrumentor()


class ReplicateInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("replicate"):
            return None
        if not is_package_installed("opentelemetry-instrumentation-replicate"):
            return None

        from opentelemetry.instrumentation.replicate import ReplicateInstrumentor

        return ReplicateInstrumentor()


class SageMakerInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("boto3"):
            return None
        if not is_package_installed("opentelemetry-instrumentation-sagemaker"):
            return None

        from opentelemetry.instrumentation.sagemaker import SageMakerInstrumentor

        return SageMakerInstrumentor()


class SkyvernInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("skyvern"):
            return None

        from ..opentelemetry.instrumentation.skyvern import SkyvernInstrumentor

        return SkyvernInstrumentor()


class TogetherInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("together"):
            return None
        if not is_package_installed("opentelemetry-instrumentation-together"):
            return None

        from opentelemetry.instrumentation.together import TogetherAiInstrumentor

        return TogetherAiInstrumentor()


class TransformersInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("transformers"):
            return None
        if not is_package_installed("opentelemetry-instrumentation-transformers"):
            return None

        from opentelemetry.instrumentation.transformers import TransformersInstrumentor

        return TransformersInstrumentor()


class VertexAIInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("vertexai"):
            return None
        if not is_package_installed("opentelemetry-instrumentation-vertexai"):
            return None

        from opentelemetry.instrumentation.vertexai import VertexAIInstrumentor

        return VertexAIInstrumentor()


class WatsonxInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not (
            is_package_installed("ibm-watsonx-ai")
            or is_package_installed("ibm-watson-machine-learning")
        ):
            return None
        if not is_package_installed("opentelemetry-instrumentation-watsonx"):
            return None

        from opentelemetry.instrumentation.watsonx import WatsonxInstrumentor

        return WatsonxInstrumentor()


class WeaviateInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("weaviate"):
            return None
        if not is_package_installed("opentelemetry-instrumentation-weaviate"):
            return None

        from opentelemetry.instrumentation.weaviate import WeaviateInstrumentor

        return WeaviateInstrumentor()
