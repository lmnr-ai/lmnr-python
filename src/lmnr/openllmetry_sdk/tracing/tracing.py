import atexit
import copy
import logging


from contextvars import Context
from lmnr.sdk.log import VerboseColorfulFormatter
from lmnr.openllmetry_sdk.instruments import Instruments
from lmnr.sdk.browser import init_browser_tracing
from lmnr.openllmetry_sdk.tracing.attributes import (
    ASSOCIATION_PROPERTIES,
    SPAN_INSTRUMENTATION_SOURCE,
    SPAN_PATH,
    TRACING_LEVEL,
)
from lmnr.openllmetry_sdk.tracing.content_allow_list import ContentAllowList
from lmnr.openllmetry_sdk.utils import is_notebook
from lmnr.openllmetry_sdk.utils.package_check import is_package_installed
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HTTPExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as GRPCExporter,
)
from opentelemetry.instrumentation.threading import ThreadingInstrumentor
from opentelemetry.context import get_value, attach, get_current, set_value, Context
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.textmap import TextMapPropagator
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, SpanProcessor, Span
from opentelemetry.sdk.trace.export import (
    SpanExporter,
    SimpleSpanProcessor,
    BatchSpanProcessor,
)
from opentelemetry.trace import get_tracer_provider, ProxyTracerProvider

from typing import Dict, Optional, Set

module_logger = logging.getLogger(__name__)
console_log_handler = logging.StreamHandler()
console_log_handler.setFormatter(VerboseColorfulFormatter())
module_logger.addHandler(console_log_handler)


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
    sagemaker-runtime,
    googleapis.com,
    githubusercontent.com,
    openaipublic.blob.core.windows.net"""


class TracerWrapper(object):
    resource_attributes: dict = {}
    enable_content_tracing: bool = True
    endpoint: str = None
    headers: Dict[str, str] = {}
    __tracer_provider: TracerProvider = None
    __logger: logging.Logger = None
    __span_id_to_path: dict[int, list[str]] = {}

    def __new__(
        cls,
        disable_batch=False,
        processor: Optional[SpanProcessor] = None,
        propagator: Optional[TextMapPropagator] = None,
        exporter: Optional[SpanExporter] = None,
        should_enrich_metrics: bool = False,
        instruments: Optional[Set[Instruments]] = None,
        base_http_url: Optional[str] = None,
        project_api_key: Optional[str] = None,
        max_export_batch_size: Optional[int] = None,
    ) -> "TracerWrapper":
        cls._initialize_logger(cls)
        if not hasattr(cls, "instance"):
            obj = cls.instance = super(TracerWrapper, cls).__new__(cls)
            if not TracerWrapper.endpoint:
                return obj

            obj.__resource = Resource(attributes=TracerWrapper.resource_attributes)
            obj.__tracer_provider = init_tracer_provider(resource=obj.__resource)
            if processor:
                obj.__spans_processor: SpanProcessor = processor
                obj.__spans_processor_original_on_start = processor.on_start
            else:
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
                        obj.__spans_exporter,
                        max_export_batch_size=max_export_batch_size,
                    )
                obj.__spans_processor_original_on_start = None

            obj.__spans_processor.on_start = obj._span_processor_on_start
            obj.__tracer_provider.add_span_processor(obj.__spans_processor)

            if propagator:
                set_global_textmap(propagator)

            # this makes sure otel context is propagated so we always want it
            ThreadingInstrumentor().instrument()

            instrument_set = init_instrumentations(
                should_enrich_metrics,
                instruments,
                base_http_url=base_http_url,
                project_api_key=project_api_key,
            )

            if not instrument_set:
                cls.__logger.warning(
                    "No valid instruments set. Remove 'instrument' "
                    "argument to use all instruments, or set a valid instrument."
                )

            obj.__content_allow_list = ContentAllowList()

            # Force flushes for debug environments (e.g. local development)
            atexit.register(obj.exit_handler)

        return cls.instance

    def exit_handler(self):
        self.__span_id_to_path = {}
        self.flush()

    def _initialize_logger(self):
        self.__logger = logging.getLogger(__name__)
        console_log_handler = logging.StreamHandler()
        console_log_handler.setFormatter(VerboseColorfulFormatter())
        self.__logger.addHandler(console_log_handler)

    def _span_processor_on_start(
        self, span: Span, parent_context: Optional[Context] = None
    ):
        span_path_in_context = get_value("span_path", parent_context or get_current())
        span_path_in_context = None
        parent_span_path = span_path_in_context or (
            self.__span_id_to_path.get(span.parent.span_id) if span.parent else None
        )
        span_path = parent_span_path + [span.name] if parent_span_path else [span.name]
        span.set_attribute(SPAN_PATH, span_path)
        set_value("span_path", span_path, get_current())
        self.__span_id_to_path[span.get_span_context().span_id] = span_path

        span.set_attribute(SPAN_INSTRUMENTATION_SOURCE, "python")

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
        return hasattr(cls, "instance")

    @classmethod
    def clear(cls):
        # Any state cleanup. Now used in between tests
        cls.__span_id_to_path = {}

    def flush(self):
        self.__spans_processor.force_flush()

    def get_tracer(self):
        return self.__tracer_provider.get_tracer(TRACER_NAME)


def set_association_properties(properties: dict) -> None:
    attach(set_value("association_properties", properties))

    span = trace.get_current_span()
    _set_association_properties_attributes(span, properties)


def update_association_properties(
    properties: dict,
    set_on_current_span: bool = True,
    context: Optional[Context] = None,
) -> None:
    """Only adds or updates properties that are not already present"""
    association_properties = get_value("association_properties", context) or {}
    association_properties.update(properties)

    attach(set_value("association_properties", association_properties, context))

    if set_on_current_span:
        span = trace.get_current_span()
        _set_association_properties_attributes(span, properties)


def remove_association_properties(properties: dict) -> None:
    props: dict = copy.copy(get_value("association_properties") or {})
    for k in properties.keys():
        props.pop(k, None)
    set_association_properties(props)


def _set_association_properties_attributes(span, properties: dict) -> None:
    for key, value in properties.items():
        if key == TRACING_LEVEL:
            span.set_attribute(f"lmnr.internal.{TRACING_LEVEL}", value)
            continue
        span.set_attribute(f"{ASSOCIATION_PROPERTIES}.{key}", value)


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
        module_logger.error(
            "Cannot add span processor to the default provider since it doesn't support it"
        )
        return
    else:
        provider = default_provider

    return provider


def init_instrumentations(
    should_enrich_metrics: bool,
    instruments: Optional[Set[Instruments]] = None,
    block_instruments: Optional[Set[Instruments]] = None,
    base_http_url: Optional[str] = None,
    project_api_key: Optional[str] = None,
):
    block_instruments = block_instruments or set()
    # These libraries are not instrumented by default,
    # but if the user wants, they can manually specify them
    default_off_instruments = set(
        [
            Instruments.REQUESTS,
            Instruments.URLLIB3,
            Instruments.REDIS,
            Instruments.PYMYSQL,
        ]
    )

    instruments = (
        instruments
        if instruments is not None
        else (set(Instruments) - default_off_instruments)
    )

    # Remove any instruments that were explicitly blocked
    instruments = instruments - block_instruments

    instrument_set = False
    for instrument in instruments:
        if instrument == Instruments.ALEPHALPHA:
            if init_alephalpha_instrumentor():
                instrument_set = True
        elif instrument == Instruments.ANTHROPIC:
            if init_anthropic_instrumentor(should_enrich_metrics):
                instrument_set = True
        elif instrument == Instruments.BEDROCK:
            if init_bedrock_instrumentor(should_enrich_metrics):
                instrument_set = True
        elif instrument == Instruments.CHROMA:
            if init_chroma_instrumentor():
                instrument_set = True
        elif instrument == Instruments.COHERE:
            if init_cohere_instrumentor():
                instrument_set = True
        elif instrument == Instruments.GOOGLE_GENERATIVEAI:
            if init_google_generativeai_instrumentor():
                instrument_set = True
        elif instrument == Instruments.GROQ:
            if init_groq_instrumentor():
                instrument_set = True
        elif instrument == Instruments.HAYSTACK:
            if init_haystack_instrumentor():
                instrument_set = True
        elif instrument == Instruments.LANCEDB:
            if init_lancedb_instrumentor():
                instrument_set = True
        elif instrument == Instruments.LANGCHAIN:
            if init_langchain_instrumentor():
                instrument_set = True
        elif instrument == Instruments.LLAMA_INDEX:
            if init_llama_index_instrumentor():
                instrument_set = True
        elif instrument == Instruments.MARQO:
            if init_marqo_instrumentor():
                instrument_set = True
        elif instrument == Instruments.MILVUS:
            if init_milvus_instrumentor():
                instrument_set = True
        elif instrument == Instruments.MISTRAL:
            if init_mistralai_instrumentor():
                instrument_set = True
        elif instrument == Instruments.OLLAMA:
            if init_ollama_instrumentor():
                instrument_set = True
        elif instrument == Instruments.OPENAI:
            if init_openai_instrumentor(should_enrich_metrics):
                instrument_set = True
        elif instrument == Instruments.PINECONE:
            if init_pinecone_instrumentor():
                instrument_set = True
        elif instrument == Instruments.PYMYSQL:
            if init_pymysql_instrumentor():
                instrument_set = True
        elif instrument == Instruments.QDRANT:
            if init_qdrant_instrumentor():
                instrument_set = True
        elif instrument == Instruments.REDIS:
            if init_redis_instrumentor():
                instrument_set = True
        elif instrument == Instruments.REPLICATE:
            if init_replicate_instrumentor():
                instrument_set = True
        elif instrument == Instruments.REQUESTS:
            if init_requests_instrumentor():
                instrument_set = True
        elif instrument == Instruments.SAGEMAKER:
            if init_sagemaker_instrumentor(should_enrich_metrics):
                instrument_set = True
        elif instrument == Instruments.TOGETHER:
            if init_together_instrumentor():
                instrument_set = True
        elif instrument == Instruments.TRANSFORMERS:
            if init_transformers_instrumentor():
                instrument_set = True
        elif instrument == Instruments.URLLIB3:
            if init_urllib3_instrumentor():
                instrument_set = True
        elif instrument == Instruments.VERTEXAI:
            if init_vertexai_instrumentor():
                instrument_set = True
        elif instrument == Instruments.WATSONX:
            if init_watsonx_instrumentor():
                instrument_set = True
        elif instrument == Instruments.WEAVIATE:
            if init_weaviate_instrumentor():
                instrument_set = True
        elif instrument == Instruments.PLAYWRIGHT:
            if init_browser_tracing(base_http_url, project_api_key):
                instrument_set = True
        else:
            module_logger.warning(
                f"Warning: {instrument} instrumentation does not exist."
            )
            module_logger.warning(
                "Usage:\n"
                "from lmnr import Laminar, Instruments\n"
                "Laminar.init(instruments=set([Instruments.OPENAI]))"
            )

    return instrument_set


def init_openai_instrumentor(should_enrich_metrics: bool):
    try:
        if is_package_installed("openai") and is_package_installed(
            "opentelemetry-instrumentation-openai"
        ):
            from opentelemetry.instrumentation.openai import OpenAIInstrumentor

            instrumentor = OpenAIInstrumentor(
                enrich_assistant=should_enrich_metrics,
                enrich_token_usage=should_enrich_metrics,
                # Default in the package provided is an empty function, which
                # results in dropping the image data if we don't explicitly
                # set it to None.
                upload_base64_image=None,
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True

    except Exception as e:
        module_logger.error(f"Error initializing OpenAI instrumentor: {e}")
        return False


def init_anthropic_instrumentor(should_enrich_metrics: bool):
    try:
        if is_package_installed("anthropic") and is_package_installed(
            "opentelemetry-instrumentation-anthropic"
        ):
            from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

            instrumentor = AnthropicInstrumentor(
                enrich_token_usage=should_enrich_metrics,
                upload_base64_image=None,
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        module_logger.error(f"Error initializing Anthropic instrumentor: {e}")
        return False


def init_cohere_instrumentor():
    try:
        if is_package_installed("cohere") and is_package_installed(
            "opentelemetry-instrumentation-cohere"
        ):
            from opentelemetry.instrumentation.cohere import CohereInstrumentor

            instrumentor = CohereInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        module_logger.error(f"Error initializing Cohere instrumentor: {e}")
        return False


def init_pinecone_instrumentor():
    try:
        if is_package_installed("pinecone") and is_package_installed(
            "opentelemetry-instrumentation-pinecone"
        ):
            from opentelemetry.instrumentation.pinecone import PineconeInstrumentor

            instrumentor = PineconeInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        module_logger.error(f"Error initializing Pinecone instrumentor: {e}")
        return False


def init_qdrant_instrumentor():
    try:
        if is_package_installed("qdrant_client") and is_package_installed(
            "opentelemetry-instrumentation-qdrant"
        ):
            from opentelemetry.instrumentation.qdrant import QdrantInstrumentor

            instrumentor = QdrantInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
    except Exception as e:
        module_logger.error(f"Error initializing Qdrant instrumentor: {e}")
        return False


def init_chroma_instrumentor():
    try:
        if is_package_installed("chromadb") and is_package_installed(
            "opentelemetry-instrumentation-chromadb"
        ):
            from opentelemetry.instrumentation.chromadb import ChromaInstrumentor

            instrumentor = ChromaInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        module_logger.error(f"Error initializing Chroma instrumentor: {e}")
        return False


def init_google_generativeai_instrumentor():
    try:
        if is_package_installed("google-generativeai") and is_package_installed(
            "opentelemetry-instrumentation-google-generativeai"
        ):
            from opentelemetry.instrumentation.google_generativeai import (
                GoogleGenerativeAiInstrumentor,
            )

            instrumentor = GoogleGenerativeAiInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        module_logger.error(f"Error initializing Gemini instrumentor: {e}")
        return False


def init_haystack_instrumentor():
    try:
        if is_package_installed("haystack") and is_package_installed(
            "opentelemetry-instrumentation-haystack"
        ):
            from opentelemetry.instrumentation.haystack import HaystackInstrumentor

            instrumentor = HaystackInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        module_logger.error(f"Error initializing Haystack instrumentor: {e}")
        return False


def init_langchain_instrumentor():
    try:
        if is_package_installed("langchain") and is_package_installed(
            "opentelemetry-instrumentation-langchain"
        ):
            from opentelemetry.instrumentation.langchain import LangchainInstrumentor

            instrumentor = LangchainInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        # FIXME: silencing this error temporarily, it appears to not be critical
        if str(e) != "No module named 'langchain_community'":
            module_logger.error(f"Error initializing LangChain instrumentor: {e}")
        return False


def init_mistralai_instrumentor():
    try:
        if is_package_installed("mistralai") and is_package_installed(
            "opentelemetry-instrumentation-mistralai"
        ):
            from opentelemetry.instrumentation.mistralai import MistralAiInstrumentor

            instrumentor = MistralAiInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        module_logger.error(f"Error initializing MistralAI instrumentor: {e}")
        return False


def init_ollama_instrumentor():
    try:
        if is_package_installed("ollama") and is_package_installed(
            "opentelemetry-instrumentation-ollama"
        ):
            from opentelemetry.instrumentation.ollama import OllamaInstrumentor

            instrumentor = OllamaInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        module_logger.error(f"Error initializing Ollama instrumentor: {e}")
        return False


def init_transformers_instrumentor():
    try:
        if is_package_installed("transformers") and is_package_installed(
            "opentelemetry-instrumentation-transformers"
        ):
            from opentelemetry.instrumentation.transformers import (
                TransformersInstrumentor,
            )

            instrumentor = TransformersInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        module_logger.error(f"Error initializing Transformers instrumentor: {e}")
        return False


def init_together_instrumentor():
    try:
        if is_package_installed("together") and is_package_installed(
            "opentelemetry-instrumentation-together"
        ):
            from opentelemetry.instrumentation.together import TogetherAiInstrumentor

            instrumentor = TogetherAiInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        module_logger.error(f"Error initializing TogetherAI instrumentor: {e}")
        return False


def init_llama_index_instrumentor():
    try:
        if (
            is_package_installed("llama-index") or is_package_installed("llama_index")
        ) and is_package_installed("opentelemetry-instrumentation-llamaindex"):
            from opentelemetry.instrumentation.llamaindex import LlamaIndexInstrumentor

            instrumentor = LlamaIndexInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        module_logger.error(f"Error initializing LlamaIndex instrumentor: {e}")
        return False


def init_milvus_instrumentor():
    try:
        if is_package_installed("pymilvus") and is_package_installed(
            "opentelemetry-instrumentation-milvus"
        ):
            from opentelemetry.instrumentation.milvus import MilvusInstrumentor

            instrumentor = MilvusInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        module_logger.error(f"Error initializing Milvus instrumentor: {e}")
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
        module_logger.error(f"Error initializing Requests instrumentor: {e}")
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
        module_logger.error(f"Error initializing urllib3 instrumentor: {e}")
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
        module_logger.error(f"Error initializing SQLAlchemy instrumentor: {e}")
        return False


def init_bedrock_instrumentor(should_enrich_metrics: bool):
    try:
        if is_package_installed("boto3") and is_package_installed(
            "opentelemetry-instrumentation-bedrock"
        ):
            from opentelemetry.instrumentation.bedrock import BedrockInstrumentor

            instrumentor = BedrockInstrumentor(
                enrich_token_usage=should_enrich_metrics,
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        module_logger.error(f"Error initializing Bedrock instrumentor: {e}")
        return False


def init_replicate_instrumentor():
    try:
        if is_package_installed("replicate") and is_package_installed(
            "opentelemetry-instrumentation-replicate"
        ):
            from opentelemetry.instrumentation.replicate import ReplicateInstrumentor

            instrumentor = ReplicateInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        module_logger.error(f"Error initializing Replicate instrumentor: {e}")
        return False


def init_vertexai_instrumentor():
    try:
        if is_package_installed("vertexai") and is_package_installed(
            "opentelemetry-instrumentation-vertexai"
        ):
            from opentelemetry.instrumentation.vertexai import VertexAIInstrumentor

            instrumentor = VertexAIInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        module_logger.warning(f"Error initializing Vertex AI instrumentor: {e}")
        return False


def init_watsonx_instrumentor():
    try:
        if (
            is_package_installed("ibm-watsonx-ai")
            or is_package_installed("ibm-watson-machine-learning")
        ) and is_package_installed("opentelemetry-instrumentation-watsonx"):
            from opentelemetry.instrumentation.watsonx import WatsonxInstrumentor

            instrumentor = WatsonxInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        module_logger.warning(f"Error initializing Watsonx instrumentor: {e}")
        return False


def init_weaviate_instrumentor():
    try:
        if is_package_installed("weaviate") and is_package_installed(
            "opentelemetry-instrumentation-weaviate"
        ):
            from opentelemetry.instrumentation.weaviate import WeaviateInstrumentor

            instrumentor = WeaviateInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        module_logger.warning(f"Error initializing Weaviate instrumentor: {e}")
        return False


def init_alephalpha_instrumentor():
    try:
        if is_package_installed("aleph_alpha_client") and is_package_installed(
            "opentelemetry-instrumentation-alephalpha"
        ):
            from opentelemetry.instrumentation.alephalpha import AlephAlphaInstrumentor

            instrumentor = AlephAlphaInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        module_logger.error(f"Error initializing Aleph Alpha instrumentor: {e}")
        return False


def init_marqo_instrumentor():
    try:
        if is_package_installed("marqo") and is_package_installed(
            "opentelemetry-instrumentation-marqo"
        ):
            from opentelemetry.instrumentation.marqo import MarqoInstrumentor

            instrumentor = MarqoInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        module_logger.error(f"Error initializing marqo instrumentor: {e}")
        return False


def init_lancedb_instrumentor():
    try:
        if is_package_installed("lancedb") and is_package_installed(
            "opentelemetry-instrumentation-lancedb"
        ):
            from opentelemetry.instrumentation.lancedb import LanceInstrumentor

            instrumentor = LanceInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        module_logger.error(f"Error initializing LanceDB instrumentor: {e}")


def init_redis_instrumentor():
    try:
        if is_package_installed("redis") and is_package_installed(
            "opentelemetry-instrumentation-redis"
        ):
            from opentelemetry.instrumentation.redis import RedisInstrumentor

            instrumentor = RedisInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument(excluded_urls=EXCLUDED_URLS)
        return True
    except Exception as e:
        module_logger.error(f"Error initializing redis instrumentor: {e}")
        return False


def init_groq_instrumentor():
    try:
        if is_package_installed("groq") and is_package_installed(
            "opentelemetry-instrumentation-groq"
        ):
            from opentelemetry.instrumentation.groq import GroqInstrumentor

            instrumentor = GroqInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        module_logger.error(f"Error initializing Groq instrumentor: {e}")
        return False


def init_sagemaker_instrumentor(should_enrich_metrics: bool):
    try:
        if is_package_installed("boto3") and is_package_installed(
            "opentelemetry-instrumentation-sagemaker"
        ):
            from opentelemetry.instrumentation.sagemaker import SageMakerInstrumentor

            instrumentor = SageMakerInstrumentor(
                enrich_token_usage=should_enrich_metrics,
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        module_logger.error(f"Error initializing SageMaker instrumentor: {e}")
        return False
