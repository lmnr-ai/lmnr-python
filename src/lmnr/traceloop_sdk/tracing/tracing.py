import atexit
import logging
import os


from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HTTPExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as GRPCExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, SpanProcessor
from opentelemetry.propagators.textmap import TextMapPropagator
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.trace.export import (
    SpanExporter,
    SimpleSpanProcessor,
    BatchSpanProcessor,
)
from opentelemetry.trace import get_tracer_provider, ProxyTracerProvider
from opentelemetry.context import get_value, attach, set_value
from opentelemetry.instrumentation.threading import ThreadingInstrumentor

# from lmnr.traceloop_sdk import Telemetry
from lmnr.traceloop_sdk.instruments import Instruments
from lmnr.traceloop_sdk.tracing.attributes import ASSOCIATION_PROPERTIES, SPAN_PATH
from lmnr.traceloop_sdk.tracing.content_allow_list import ContentAllowList
from lmnr.traceloop_sdk.utils import is_notebook
from lmnr.traceloop_sdk.utils.package_check import is_package_installed
from typing import Dict, Optional, Set


TRACER_NAME = "lmnr.tracer"
EXCLUDED_URLS = """
    iam.cloud.ibm.com,
    dataplatform.cloud.ibm.com,
    ml.cloud.ibm.com,
    api.openai.com,
    openai.azure.com,
    api.anthropic.com,
    api.cohere.ai,
    pinecone.io,
    api.lmnr.ai,
    posthog.com,
    sentry.io,
    bedrock-runtime,
    googleapis.com,
    githubusercontent.com,
    openaipublic.blob.core.windows.net"""


class TracerWrapper(object):
    resource_attributes: dict = {}
    enable_content_tracing: bool = True
    endpoint: str = None
    headers: Dict[str, str] = {}
    __tracer_provider: TracerProvider = None

    def __new__(
        cls,
        disable_batch=False,
        processor: Optional[SpanProcessor] = None,
        propagator: Optional[TextMapPropagator] = None,
        exporter: Optional[SpanExporter] = None,
        should_enrich_metrics: bool = True,
        instruments: Optional[Set[Instruments]] = None,
    ) -> "TracerWrapper":
        if not hasattr(cls, "instance"):
            obj = cls.instance = super(TracerWrapper, cls).__new__(cls)
            if not TracerWrapper.endpoint:
                return obj

            obj.__resource = Resource(attributes=TracerWrapper.resource_attributes)
            obj.__tracer_provider = init_tracer_provider(resource=obj.__resource)
            if processor:
                # Telemetry().capture("tracer:init", {"processor": "custom"})
                obj.__spans_processor: SpanProcessor = processor
                obj.__spans_processor_original_on_start = processor.on_start
            else:
                # if exporter:
                #     Telemetry().capture(
                #         "tracer:init",
                #         {
                #             "exporter": "custom",
                #             "processor": "simple" if disable_batch else "batch",
                #         },
                #     )
                # else:
                #     Telemetry().capture(
                #         "tracer:init",
                #         {
                #             "exporter": TracerWrapper.endpoint,
                #             "processor": "simple" if disable_batch else "batch",
                #         },
                #     )

                obj.__spans_exporter: SpanExporter = (
                    exporter
                    if exporter
                    else init_spans_exporter(
                        TracerWrapper.endpoint, TracerWrapper.headers
                    )
                )
                if disable_batch or is_notebook():
                    obj.__spans_processor: SpanProcessor = SimpleSpanProcessor(
                        obj.__spans_exporter
                    )
                else:
                    obj.__spans_processor: SpanProcessor = BatchSpanProcessor(
                        obj.__spans_exporter
                    )
                obj.__spans_processor_original_on_start = None

            obj.__spans_processor.on_start = obj._span_processor_on_start
            obj.__tracer_provider.add_span_processor(obj.__spans_processor)

            if propagator:
                set_global_textmap(propagator)

            # this makes sure otel context is propagated so we always want it
            ThreadingInstrumentor().instrument()

            if instruments is None:
                init_instrumentations(should_enrich_metrics)
            else:
                for instrument in instruments:
                    if instrument == Instruments.OPENAI:
                        if not init_openai_instrumentor(should_enrich_metrics):
                            print("Warning: OpenAI library does not exist.")
                    elif instrument == Instruments.ANTHROPIC:
                        if not init_anthropic_instrumentor(should_enrich_metrics):
                            print("Warning: Anthropic library does not exist.")
                    elif instrument == Instruments.COHERE:
                        if not init_cohere_instrumentor():
                            print("Warning: Cohere library does not exist.")
                    elif instrument == Instruments.PINECONE:
                        if not init_pinecone_instrumentor():
                            print("Warning: Pinecone library does not exist.")
                    elif instrument == Instruments.CHROMA:
                        if not init_chroma_instrumentor():
                            print("Warning: Chroma library does not exist.")
                    elif instrument == Instruments.GOOGLE_GENERATIVEAI:
                        if not init_google_generativeai_instrumentor():
                            print(
                                "Warning: Google Generative AI library does not exist."
                            )
                    elif instrument == Instruments.LANGCHAIN:
                        if not init_langchain_instrumentor():
                            print("Warning: LangChain library does not exist.")
                    elif instrument == Instruments.MISTRAL:
                        if not init_mistralai_instrumentor():
                            print("Warning: MistralAI library does not exist.")
                    elif instrument == Instruments.OLLAMA:
                        if not init_ollama_instrumentor():
                            print("Warning: Ollama library does not exist.")
                    elif instrument == Instruments.LLAMA_INDEX:
                        if not init_llama_index_instrumentor():
                            print("Warning: LlamaIndex library does not exist.")
                    elif instrument == Instruments.MILVUS:
                        if not init_milvus_instrumentor():
                            print("Warning: Milvus library does not exist.")
                    elif instrument == Instruments.TRANSFORMERS:
                        if not init_transformers_instrumentor():
                            print("Warning: Transformers library does not exist.")
                    elif instrument == Instruments.TOGETHER:
                        if not init_together_instrumentor():
                            print("Warning: TogetherAI library does not exist.")
                    elif instrument == Instruments.REQUESTS:
                        if not init_requests_instrumentor():
                            print("Warning: Requests library does not exist.")
                    elif instrument == Instruments.URLLIB3:
                        if not init_urllib3_instrumentor():
                            print("Warning: urllib3 library does not exist.")
                    elif instrument == Instruments.PYMYSQL:
                        if not init_pymysql_instrumentor():
                            print("Warning: PyMySQL library does not exist.")
                    elif instrument == Instruments.BEDROCK:
                        if not init_bedrock_instrumentor(should_enrich_metrics):
                            print("Warning: Bedrock library does not exist.")
                    elif instrument == Instruments.REPLICATE:
                        if not init_replicate_instrumentor():
                            print("Warning: Replicate library does not exist.")
                    elif instrument == Instruments.VERTEXAI:
                        if not init_vertexai_instrumentor():
                            print("Warning: Vertex AI library does not exist.")
                    elif instrument == Instruments.WATSONX:
                        if not init_watsonx_instrumentor():
                            print("Warning: Watsonx library does not exist.")
                    elif instrument == Instruments.WEAVIATE:
                        if not init_weaviate_instrumentor():
                            print("Warning: Weaviate library does not exist.")
                    elif instrument == Instruments.ALEPHALPHA:
                        if not init_alephalpha_instrumentor():
                            print("Warning: Aleph Alpha library does not exist.")
                    elif instrument == Instruments.MARQO:
                        if not init_marqo_instrumentor():
                            print("Warning: marqo library does not exist.")
                    elif instrument == Instruments.LANCEDB:
                        if not init_lancedb_instrumentor():
                            print("Warning: LanceDB library does not exist.")
                    elif instrument == Instruments.REDIS:
                        if not init_redis_instrumentor():
                            print("Warning: redis library does not exist.")

                    else:
                        print(
                            "Warning: "
                            + instrument
                            + " instrumentation does not exist."
                        )
                        print(
                            "Usage:\n"
                            + "from lmnr.traceloop_sdk.instruments import Instruments\n"
                            + 'Traceloop.init(app_name="...", instruments=set([Instruments.OPENAI]))'
                        )

            obj.__content_allow_list = ContentAllowList()

            # Force flushes for debug environments (e.g. local development)
            atexit.register(obj.exit_handler)

        return cls.instance

    def exit_handler(self):
        self.flush()

    def _span_processor_on_start(self, span, parent_context):
        span_path = get_value("span_path")
        if span_path is not None:
            # This is done redundantly here for most decorated functions
            # However, need to do this for auto-instrumented libraries.
            # Then, for auto-instrumented ones, they'll attach
            # the final part of the name to the span on the backend.
            span.set_attribute(SPAN_PATH, span_path)

        association_properties = get_value("association_properties")
        if association_properties is not None:
            _set_association_properties_attributes(span, association_properties)

            if not self.enable_content_tracing:
                if self.__content_allow_list.is_allowed(association_properties):
                    attach(set_value("override_enable_content_tracing", True))
                else:
                    attach(set_value("override_enable_content_tracing", False))

        # Call original on_start method if it exists in custom processor
        if self.__spans_processor_original_on_start:
            self.__spans_processor_original_on_start(span, parent_context)

    @staticmethod
    def set_static_params(
        resource_attributes: dict,
        enable_content_tracing: bool,
        endpoint: str,
        headers: Dict[str, str],
    ) -> None:
        TracerWrapper.resource_attributes = resource_attributes
        TracerWrapper.enable_content_tracing = enable_content_tracing
        TracerWrapper.endpoint = endpoint
        TracerWrapper.headers = headers

    @classmethod
    def verify_initialized(cls) -> bool:
        if hasattr(cls, "instance"):
            return True

        if (os.getenv("TRACELOOP_SUPPRESS_WARNINGS") or "false").lower() == "true":
            return False

        print("Warning: Laminar not initialized, make sure to initialize")
        return False

    def flush(self):
        self.__spans_processor.force_flush()

    def get_tracer(self):
        return self.__tracer_provider.get_tracer(TRACER_NAME)


def set_association_properties(properties: dict) -> None:
    attach(set_value("association_properties", properties))

    # TODO: When called inside observe decorator, this actually sets the properties on the parent span, not the current one
    # Then, processor's on_start will assign this to current span
    span = trace.get_current_span()
    _set_association_properties_attributes(span, properties)


def update_association_properties(properties: dict) -> None:
    """Only adds or updates properties that are not already present"""
    association_properties = get_value("association_properties") or {}
    association_properties.update(properties)

    attach(set_value("association_properties", association_properties))

    # TODO: When called inside observe decorator, this actually sets the properties on the parent span, not the current one
    # Then, processor's on_start will assign this to current span
    span = trace.get_current_span()
    _set_association_properties_attributes(span, properties)


def _set_association_properties_attributes(span, properties: dict) -> None:
    for key, value in properties.items():
        span.set_attribute(f"{ASSOCIATION_PROPERTIES}.{key}", value)


def get_span_path(span_name: str) -> str:
    current_span_path = get_value("span_path")
    span_path = f"{current_span_path}.{span_name}" if current_span_path else span_name
    return span_path


def set_managed_prompt_tracing_context(
    key: str,
    version: int,
    version_name: str,
    version_hash: str,
    template_variables: dict,
) -> None:
    attach(set_value("managed_prompt", True))
    attach(set_value("prompt_key", key))
    attach(set_value("prompt_version", version))
    attach(set_value("prompt_version_name", version_name))
    attach(set_value("prompt_version_hash", version_hash))
    attach(set_value("prompt_template_variables", template_variables))


def set_external_prompt_tracing_context(
    template: str, variables: dict, version: int
) -> None:
    attach(set_value("managed_prompt", False))
    attach(set_value("prompt_version", version))
    attach(set_value("prompt_template", template))
    attach(set_value("prompt_template_variables", variables))


def init_spans_exporter(api_endpoint: str, headers: Dict[str, str]) -> SpanExporter:
    if "http" in api_endpoint.lower() or "https" in api_endpoint.lower():
        return HTTPExporter(endpoint=f"{api_endpoint}/v1/traces", headers=headers)
    else:
        return GRPCExporter(endpoint=f"{api_endpoint}", headers=headers)


def init_tracer_provider(resource: Resource) -> TracerProvider:
    provider: TracerProvider = None
    default_provider: TracerProvider = get_tracer_provider()

    if isinstance(default_provider, ProxyTracerProvider):
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)
    elif not hasattr(default_provider, "add_span_processor"):
        logging.error(
            "Cannot add span processor to the default provider since it doesn't support it"
        )
        return
    else:
        provider = default_provider

    return provider


def init_instrumentations(should_enrich_metrics: bool):
    init_openai_instrumentor(should_enrich_metrics)
    init_anthropic_instrumentor(should_enrich_metrics)
    init_cohere_instrumentor()
    init_pinecone_instrumentor()
    init_qdrant_instrumentor()
    init_chroma_instrumentor()
    init_google_generativeai_instrumentor()
    init_haystack_instrumentor()
    init_langchain_instrumentor()
    init_mistralai_instrumentor()
    init_ollama_instrumentor()
    init_llama_index_instrumentor()
    init_milvus_instrumentor()
    init_transformers_instrumentor()
    init_together_instrumentor()
    init_bedrock_instrumentor(should_enrich_metrics)
    init_replicate_instrumentor()
    init_vertexai_instrumentor()
    init_watsonx_instrumentor()
    init_weaviate_instrumentor()
    init_alephalpha_instrumentor()
    init_marqo_instrumentor()
    init_lancedb_instrumentor()
    init_groq_instrumentor()

    # These libraries are not instrumented by default, but if the user wants, he can manually specify them
    # init_redis_instrumentor()
    # init_requests_instrumentor()
    # init_urllib3_instrumentor()
    # init_pymysql_instrumentor()


def init_openai_instrumentor(should_enrich_metrics: bool):
    try:
        if is_package_installed("openai"):
            # Telemetry().capture("instrumentation:openai:init")
            from opentelemetry.instrumentation.openai import OpenAIInstrumentor

            instrumentor = OpenAIInstrumentor(
                # exception_logger=lambda e: Telemetry().log_exception(e),
                enrich_assistant=should_enrich_metrics,
                enrich_token_usage=should_enrich_metrics,
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True

    except Exception as e:
        logging.error(f"Error initializing OpenAI instrumentor: {e}")
        # Telemetry().log_exception(e)
        return False


def init_anthropic_instrumentor(should_enrich_metrics: bool):
    try:
        if is_package_installed("anthropic"):
            # Telemetry().capture("instrumentation:anthropic:init")
            from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

            instrumentor = AnthropicInstrumentor(
                # exception_logger=lambda e: Telemetry().log_exception(e),
                enrich_token_usage=should_enrich_metrics,
                upload_base64_image=None,
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.error(f"Error initializing Anthropic instrumentor: {e}")
        # Telemetry().log_exception(e)
        return False


def init_cohere_instrumentor():
    try:
        if is_package_installed("cohere"):
            # Telemetry().capture("instrumentation:cohere:init")
            from opentelemetry.instrumentation.cohere import CohereInstrumentor

            instrumentor = CohereInstrumentor(
                # exception_logger=lambda e: Telemetry().log_exception(e),
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.error(f"Error initializing Cohere instrumentor: {e}")
        # Telemetry().log_exception(e)
        return False


def init_pinecone_instrumentor():
    try:
        if is_package_installed("pinecone"):
            # Telemetry().capture("instrumentation:pinecone:init")
            from opentelemetry.instrumentation.pinecone import PineconeInstrumentor

            instrumentor = PineconeInstrumentor(
                # exception_logger=lambda e: Telemetry().log_exception(e),
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.error(f"Error initializing Pinecone instrumentor: {e}")
        # Telemetry().log_exception(e)
        return False


def init_qdrant_instrumentor():
    try:
        if is_package_installed("qdrant_client"):
            # Telemetry().capture("instrumentation:qdrant:init")
            from opentelemetry.instrumentation.qdrant import QdrantInstrumentor

            instrumentor = QdrantInstrumentor(
                # exception_logger=lambda e: Telemetry().log_exception(e),
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
    except Exception as e:
        logging.error(f"Error initializing Qdrant instrumentor: {e}")
        # Telemetry().log_exception(e)
        return False


def init_chroma_instrumentor():
    try:
        if is_package_installed("chromadb"):
            # Telemetry().capture("instrumentation:chromadb:init")
            from opentelemetry.instrumentation.chromadb import ChromaInstrumentor

            instrumentor = ChromaInstrumentor(
                # exception_logger=lambda e: Telemetry().log_exception(e),
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.error(f"Error initializing Chroma instrumentor: {e}")
        # Telemetry().log_exception(e)
        return False


def init_google_generativeai_instrumentor():
    try:
        if is_package_installed("google.generativeai"):
            # Telemetry().capture("instrumentation:gemini:init")
            from opentelemetry.instrumentation.google_generativeai import (
                GoogleGenerativeAiInstrumentor,
            )

            instrumentor = GoogleGenerativeAiInstrumentor(
                # exception_logger=lambda e: Telemetry().log_exception(e),
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.error(f"Error initializing Gemini instrumentor: {e}")
        # Telemetry().log_exception(e)
        return False


def init_haystack_instrumentor():
    try:
        if is_package_installed("haystack"):
            # Telemetry().capture("instrumentation:haystack:init")
            from opentelemetry.instrumentation.haystack import HaystackInstrumentor

            instrumentor = HaystackInstrumentor(
                # exception_logger=lambda e: Telemetry().log_exception(e),
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.error(f"Error initializing Haystack instrumentor: {e}")
        # Telemetry().log_exception(e)
        return False


def init_langchain_instrumentor():
    try:
        if is_package_installed("langchain"):
            # Telemetry().capture("instrumentation:langchain:init")
            from opentelemetry.instrumentation.langchain import LangchainInstrumentor

            instrumentor = LangchainInstrumentor(
                # exception_logger=lambda e: Telemetry().log_exception(e),
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.error(f"Error initializing LangChain instrumentor: {e}")
        # Telemetry().log_exception(e)
        return False


def init_mistralai_instrumentor():
    try:
        if is_package_installed("mistralai"):
            # Telemetry().capture("instrumentation:mistralai:init")
            from opentelemetry.instrumentation.mistralai import MistralAiInstrumentor

            instrumentor = MistralAiInstrumentor(
                # exception_logger=lambda e: Telemetry().log_exception(e),
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.error(f"Error initializing MistralAI instrumentor: {e}")
        # Telemetry().log_exception(e)
        return False


def init_ollama_instrumentor():
    try:
        if is_package_installed("ollama"):
            # Telemetry().capture("instrumentation:ollama:init")
            from opentelemetry.instrumentation.ollama import OllamaInstrumentor

            instrumentor = OllamaInstrumentor(
                # exception_logger=lambda e: Telemetry().log_exception(e),
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.error(f"Error initializing Ollama instrumentor: {e}")
        # Telemetry().log_exception(e)
        return False


def init_transformers_instrumentor():
    try:
        if is_package_installed("transformers"):
            # Telemetry().capture("instrumentation:transformers:init")
            from opentelemetry.instrumentation.transformers import (
                TransformersInstrumentor,
            )

            instrumentor = TransformersInstrumentor(
                # exception_logger=lambda e: Telemetry().log_exception(e),
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.error(f"Error initializing Transformers instrumentor: {e}")
        # Telemetry().log_exception(e)
        return False


def init_together_instrumentor():
    try:
        if is_package_installed("together"):
            # Telemetry().capture("instrumentation:together:init")
            from opentelemetry.instrumentation.together import TogetherAiInstrumentor

            instrumentor = TogetherAiInstrumentor(
                # exception_logger=lambda e: Telemetry().log_exception(e),
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.error(f"Error initializing TogetherAI instrumentor: {e}")
        # Telemetry().log_exception(e)
        return False


def init_llama_index_instrumentor():
    try:
        if is_package_installed("llama-index") or is_package_installed("llama_index"):
            # Telemetry().capture("instrumentation:llamaindex:init")
            from opentelemetry.instrumentation.llamaindex import LlamaIndexInstrumentor

            instrumentor = LlamaIndexInstrumentor(
                # exception_logger=lambda e: Telemetry().log_exception(e),
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.error(f"Error initializing LlamaIndex instrumentor: {e}")
        # Telemetry().log_exception(e)
        return False


def init_milvus_instrumentor():
    try:
        if is_package_installed("pymilvus"):
            # Telemetry().capture("instrumentation:milvus:init")
            from opentelemetry.instrumentation.milvus import MilvusInstrumentor

            instrumentor = MilvusInstrumentor(
                # exception_logger=lambda e: Telemetry().log_exception(e),
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.error(f"Error initializing Milvus instrumentor: {e}")
        # Telemetry().log_exception(e)
        return False


def init_requests_instrumentor():
    try:
        if is_package_installed("requests"):
            from opentelemetry.instrumentation.requests import RequestsInstrumentor

            instrumentor = RequestsInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument(excluded_urls=EXCLUDED_URLS)
        return True
    except Exception as e:
        logging.error(f"Error initializing Requests instrumentor: {e}")
        # Telemetry().log_exception(e)
        return False


def init_urllib3_instrumentor():
    try:
        if is_package_installed("urllib3"):
            from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor

            instrumentor = URLLib3Instrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument(excluded_urls=EXCLUDED_URLS)
        return True
    except Exception as e:
        logging.error(f"Error initializing urllib3 instrumentor: {e}")
        # Telemetry().log_exception(e)
        return False


def init_pymysql_instrumentor():
    try:
        if is_package_installed("sqlalchemy"):
            from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

            instrumentor = SQLAlchemyInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.error(f"Error initializing SQLAlchemy instrumentor: {e}")
        # Telemetry().log_exception(e)
        return False


def init_bedrock_instrumentor(should_enrich_metrics: bool):
    try:
        if is_package_installed("boto3"):
            from opentelemetry.instrumentation.bedrock import BedrockInstrumentor

            instrumentor = BedrockInstrumentor(
                # exception_logger=lambda e: Telemetry().log_exception(e),
                enrich_token_usage=should_enrich_metrics,
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.error(f"Error initializing Bedrock instrumentor: {e}")
        # Telemetry().log_exception(e)
        return False


def init_replicate_instrumentor():
    try:
        if is_package_installed("replicate"):
            # Telemetry().capture("instrumentation:replicate:init")
            from opentelemetry.instrumentation.replicate import ReplicateInstrumentor

            instrumentor = ReplicateInstrumentor(
                # exception_logger=lambda e: Telemetry().log_exception(e),
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.error(f"Error initializing Replicate instrumentor: {e}")
        # Telemetry().log_exception(e)
        return False


def init_vertexai_instrumentor():
    try:
        if is_package_installed("vertexai"):
            # Telemetry().capture("instrumentation:vertexai:init")
            from opentelemetry.instrumentation.vertexai import VertexAIInstrumentor

            instrumentor = VertexAIInstrumentor(
                # exception_logger=lambda e: Telemetry().log_exception(e),
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.warning(f"Error initializing Vertex AI instrumentor: {e}")
        # Telemetry().log_exception(e)
        return False


def init_watsonx_instrumentor():
    try:
        if is_package_installed("ibm-watsonx-ai") or is_package_installed(
            "ibm-watson-machine-learning"
        ):
            # Telemetry().capture("instrumentation:watsonx:init")
            from opentelemetry.instrumentation.watsonx import WatsonxInstrumentor

            instrumentor = WatsonxInstrumentor(
                # exception_logger=lambda e: Telemetry().log_exception(e),
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.warning(f"Error initializing Watsonx instrumentor: {e}")
        # Telemetry().log_exception(e)
        return False


def init_weaviate_instrumentor():
    try:
        if is_package_installed("weaviate"):
            # Telemetry().capture("instrumentation:weaviate:init")
            from opentelemetry.instrumentation.weaviate import WeaviateInstrumentor

            instrumentor = WeaviateInstrumentor(
                # exception_logger=lambda e: Telemetry().log_exception(e),
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.warning(f"Error initializing Weaviate instrumentor: {e}")
        # Telemetry().log_exception(e)
        return False


def init_alephalpha_instrumentor():
    try:
        if is_package_installed("aleph_alpha_client"):
            # Telemetry().capture("instrumentation:alephalpha:init")
            from opentelemetry.instrumentation.alephalpha import AlephAlphaInstrumentor

            instrumentor = AlephAlphaInstrumentor(
                # exception_logger=lambda e: Telemetry().log_exception(e),
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.error(f"Error initializing Aleph Alpha instrumentor: {e}")
        # Telemetry().log_exception(e)
        return False


def init_marqo_instrumentor():
    try:
        if is_package_installed("marqo"):
            # Telemetry().capture("instrumentation:marqo:init")
            from opentelemetry.instrumentation.marqo import MarqoInstrumentor

            instrumentor = MarqoInstrumentor(
                # exception_logger=lambda e: Telemetry().log_exception(e),
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.error(f"Error initializing marqo instrumentor: {e}")
        # Telemetry().log_exception(e)
        return False


def init_lancedb_instrumentor():
    try:
        if is_package_installed("lancedb"):
            # Telemetry().capture("instrumentation:lancedb:init")
            from opentelemetry.instrumentation.lancedb import LanceInstrumentor

            instrumentor = LanceInstrumentor(
                # exception_logger=lambda e: Telemetry().log_exception(e),
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.error(f"Error initializing LanceDB instrumentor: {e}")


def init_redis_instrumentor():
    try:
        if is_package_installed("redis"):
            from opentelemetry.instrumentation.redis import RedisInstrumentor

            instrumentor = RedisInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument(excluded_urls=EXCLUDED_URLS)
        return True
    except Exception as e:
        logging.error(f"Error initializing redis instrumentor: {e}")
        # Telemetry().log_exception(e)
        return False


def init_groq_instrumentor():
    try:
        if is_package_installed("groq"):
            # Telemetry().capture("instrumentation:groq:init")
            from opentelemetry.instrumentation.groq import GroqInstrumentor

            instrumentor = GroqInstrumentor(
                # exception_logger=lambda e: Telemetry().log_exception(e),
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.error(f"Error initializing Groq instrumentor: {e}")
        # Telemetry().log_exception(e)
        return False
