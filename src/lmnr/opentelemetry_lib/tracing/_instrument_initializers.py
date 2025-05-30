import abc

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

from lmnr.opentelemetry_lib.utils.package_check import is_package_installed


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
        if not is_package_installed("opentelemetry-instrumentation-anthropic"):
            return None

        from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

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
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("browser-use"):
            return None

        from lmnr.sdk.browser.browser_use_otel import BrowserUseInstrumentor

        return BrowserUseInstrumentor()


class ChromaInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("chromadb"):
            return None
        if not is_package_installed("opentelemetry-instrumentation-chromadb"):
            return None

        from opentelemetry.instrumentation.chromadb import ChromaInstrumentor

        return ChromaInstrumentor()


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


class GoogleGenerativeAIInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("google-generativeai"):
            return None
        if not is_package_installed(
            "opentelemetry-instrumentation-google-generativeai"
        ):
            return None

        from opentelemetry.instrumentation.google_generativeai import (
            GoogleGenerativeAiInstrumentor,
        )

        return GoogleGenerativeAiInstrumentor()


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
        if not is_package_installed("opentelemetry-instrumentation-groq"):
            return None

        from opentelemetry.instrumentation.groq import GroqInstrumentor

        return GroqInstrumentor()


class HaystackInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(self, *args, **kwargs) -> BaseInstrumentor | None:
        if not is_package_installed("haystack"):
            return None
        if not is_package_installed("opentelemetry-instrumentation-haystack"):
            return None

        from opentelemetry.instrumentation.haystack import HaystackInstrumentor

        return HaystackInstrumentor()


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
        if not is_package_installed("opentelemetry-instrumentation-openai"):
            return None

        from opentelemetry.instrumentation.openai import OpenAIInstrumentor

        return OpenAIInstrumentor(
            # Default in the package provided is an empty function, which
            # results in dropping the image data if we don't explicitly
            # set it to None.
            upload_base64_image=None,
        )


class PatchrightInstrumentorInitializer(InstrumentorInitializer):
    def init_instrumentor(
        self, client, async_client, *args, **kwargs
    ) -> BaseInstrumentor | None:
        if not is_package_installed("patchright"):
            return None

        from lmnr.sdk.browser.patchright_otel import PatchrightInstrumentor

        return PatchrightInstrumentor(client, async_client)


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
        self, client, async_client, *args, **kwargs
    ) -> BaseInstrumentor | None:
        if not is_package_installed("playwright"):
            return None

        from lmnr.sdk.browser.playwright_otel import PlaywrightInstrumentor

        return PlaywrightInstrumentor(client, async_client)


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
