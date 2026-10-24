"""Map a LangChain model class name onto the `gen_ai.system` vendor string."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_VENDOR = "langchain"


@dataclass(frozen=True)
class _VendorRule:
    exact_matches: frozenset[str]
    patterns: tuple[str, ...]
    vendor_name: str

    def matches(self, class_name: str) -> bool:
        if class_name in self.exact_matches:
            return True
        class_lower = class_name.lower()
        return any(pattern in class_lower for pattern in self.patterns)


# Ordered most specific first: "AzureChatOpenAI" must resolve to Azure before
# the generic "openai" pattern gets a chance to match it.
_VENDOR_RULES: tuple[_VendorRule, ...] = (
    _VendorRule(
        frozenset({"AzureChatOpenAI", "AzureOpenAI", "AzureOpenAIEmbeddings"}),
        ("azure",),
        "azure.ai.openai",
    ),
    _VendorRule(
        frozenset({"ChatOpenAI", "OpenAI", "OpenAIEmbeddings"}),
        ("openai",),
        "openai",
    ),
    _VendorRule(
        frozenset({"ChatBedrock", "ChatBedrockConverse", "BedrockEmbeddings", "Bedrock"}),
        ("bedrock", "aws"),
        "aws.bedrock",
    ),
    _VendorRule(
        frozenset({"ChatAnthropic", "AnthropicLLM"}),
        ("anthropic",),
        "anthropic",
    ),
    _VendorRule(
        frozenset({"ChatVertexAI", "VertexAI", "VertexAIEmbeddings"}),
        ("vertex",),
        "gcp.vertex_ai",
    ),
    _VendorRule(
        frozenset({"ChatGoogleGenerativeAI", "GoogleGenerativeAI"}),
        ("google", "gemini"),
        "gcp.gen_ai",
    ),
    _VendorRule(
        frozenset({"ChatCohere", "CohereEmbeddings", "Cohere"}),
        ("cohere",),
        "cohere",
    ),
    _VendorRule(
        frozenset({"HuggingFacePipeline", "HuggingFaceEndpoint", "ChatHuggingFace"}),
        ("huggingface",),
        "hugging_face",
    ),
    _VendorRule(
        frozenset({"ChatOllama", "OllamaEmbeddings", "OllamaLLM"}),
        ("ollama",),
        "ollama",
    ),
    _VendorRule(
        frozenset({"Together", "ChatTogether"}),
        ("together",),
        "together_ai",
    ),
    _VendorRule(
        frozenset({"Replicate", "ChatReplicate"}),
        ("replicate",),
        "replicate",
    ),
    _VendorRule(
        frozenset({"ChatFireworks", "Fireworks"}),
        ("fireworks",),
        "fireworks",
    ),
    _VendorRule(frozenset({"ChatGroq"}), ("groq",), "groq"),
    _VendorRule(
        frozenset({"ChatMistralAI", "MistralAI"}),
        ("mistral",),
        "mistral_ai",
    ),
)


def detect_vendor_from_class(class_name: str | None) -> str:
    """Return the vendor for a LangChain model class name, or `"langchain"`."""
    if not class_name:
        return DEFAULT_VENDOR
    for rule in _VENDOR_RULES:
        if rule.matches(class_name):
            return rule.vendor_name
    return DEFAULT_VENDOR
