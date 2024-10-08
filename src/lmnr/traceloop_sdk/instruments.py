from enum import Enum


class Instruments(Enum):
    # The list of libraries which will be autoinstrumented
    # if no specific instruments are provided to initialize()
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    PINECONE = "pinecone"
    CHROMA = "chroma"
    GOOGLE_GENERATIVEAI = "google_generativeai"
    LANGCHAIN = "langchain"
    MISTRAL = "mistral"
    OLLAMA = "ollama"
    LLAMA_INDEX = "llama_index"
    MILVUS = "milvus"
    TRANSFORMERS = "transformers"
    TOGETHER = "together"
    BEDROCK = "bedrock"
    REPLICATE = "replicate"
    VERTEXAI = "vertexai"
    WATSONX = "watsonx"
    WEAVIATE = "weaviate"
    ALEPHALPHA = "alephalpha"
    MARQO = "marqo"
    LANCEDB = "lancedb"

    # The following libraries will not be autoinstrumented unless
    # specified explicitly in the initialize() call.
    REDIS = "redis"
    REQUESTS = "requests"
    URLLIB3 = "urllib3"
    PYMYSQL = "pymysql"
