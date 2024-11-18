from enum import Enum


class Instruments(Enum):
    # The list of libraries which will be autoinstrumented
    # if no specific instruments are provided to initialize()
    ALEPHALPHA = "alephalpha"
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    CHROMA = "chroma"
    COHERE = "cohere"
    GOOGLE_GENERATIVEAI = "google_generativeai"
    GROQ = "groq"
    HAYSTACK = "haystack"
    LANCEDB = "lancedb"
    LANGCHAIN = "langchain"
    LLAMA_INDEX = "llama_index"
    MARQO = "marqo"
    MILVUS = "milvus"
    MISTRAL = "mistral"
    OLLAMA = "ollama"
    OPENAI = "openai"
    PINECONE = "pinecone"
    QDRANT = "qdrant"
    REPLICATE = "replicate"
    SAGEMAKER = "sagemaker"
    TOGETHER = "together"
    TRANSFORMERS = "transformers"
    VERTEXAI = "vertexai"
    WATSONX = "watsonx"
    WEAVIATE = "weaviate"

    # The following libraries will not be autoinstrumented unless
    # specified explicitly in the initialize() call.
    REDIS = "redis"
    REQUESTS = "requests"
    URLLIB3 = "urllib3"
    PYMYSQL = "pymysql"
