"""Langfuse -> Laminar bridge.

Langfuse's Python SDK (3.x and later) is built on OpenTelemetry: every
observation is a span on a `TracerProvider` that a `LangfuseSpanProcessor`
exports to Langfuse. The bridge adds a `LangfuseBridgeSpanProcessor` to those
providers, which hands Laminar a translated copy of each span Langfuse exports.
Langfuse's own spans and export are left exactly as they were.

- `attributes.py`: attribute-name constants.
- `translate.py`: pure shape conversions.
- `processor.py`: the bridge span processor and attribute translation.
- `provider_attachment.py`: attaching to Langfuse providers, now and later.
- `litellm_bridge.py`: LiteLLM's `langfuse_otel` callback, whose providers
  Langfuse's resource manager never sees.
"""

from __future__ import annotations

from packaging.version import InvalidVersion, parse
from typing_extensions import override

from lmnr.opentelemetry_lib.bridges.base import Bridge, BridgeContext, LaminarBridge
from lmnr.opentelemetry_lib.bridges.langfuse.litellm_bridge import LiteLLMLangfuseBridge
from lmnr.opentelemetry_lib.bridges.langfuse.provider_attachment import ProviderAttachment
from lmnr.opentelemetry_lib.tracing.instruments import Instruments
from lmnr.opentelemetry_lib.utils.package_check import get_package_version, is_package_installed
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

MIN_LANGFUSE_MAJOR = 3


def _langfuse_version() -> str | None:
    if not is_package_installed("langfuse"):
        return None
    return get_package_version("langfuse")


def _major(version: str | None) -> int | None:
    if version is None:
        return None
    try:
        return parse(version).major
    except InvalidVersion:
        return None


class LangfuseBridge(LaminarBridge):
    bridge: Bridge = Bridge.LANGFUSE
    requirement: str = f"langfuse >= {MIN_LANGFUSE_MAJOR}.0 (`pip install -U langfuse`)"
    # Everything that records LLM calls or agent-framework steps, which reach
    # Laminar through Langfuse (its wrappers, or OTel instrumentations on its
    # provider). Browser / sandbox instrumentations and the Claude Agent SDK,
    # which Langfuse doesn't cover as well, stay on.
    conflicting_instruments: frozenset[Instruments] = frozenset(
        {
            Instruments.ANTHROPIC,
            Instruments.COHERE,
            Instruments.DEEPAGENTS,
            Instruments.GOOGLE_ADK,
            Instruments.GOOGLE_AIPLATFORM,
            Instruments.GOOGLE_GENAI,
            Instruments.GROQ,
            Instruments.LANGCHAIN,
            Instruments.LANGGRAPH,
            Instruments.LITELLM,
            Instruments.MCP,
            Instruments.MICROSOFT_AGENT_FRAMEWORK,
            Instruments.OPENAI,
            Instruments.OPENAI_AGENTS,
            Instruments.OPENROUTER,
            Instruments.PYDANTIC_AI,
        }
    )
    # On a shared global provider Laminar and Langfuse would see (and stamp)
    # each other's spans, and Langfuse would inherit Laminar's sampling and
    # resource. Kept apart, Langfuse builds its provider exactly as it would
    # without Laminar.
    requires_private_tracer_provider: bool = True
    inherits_global_context: bool = True

    def __init__(self) -> None:
        self._attachment: ProviderAttachment | None = None
        self._litellm: LiteLLMLangfuseBridge | None = None

    @override
    def is_available(self) -> bool:
        major = _major(_langfuse_version())
        return major is not None and major >= MIN_LANGFUSE_MAJOR

    @property
    def installed(self) -> bool:
        return self._attachment is not None

    @override
    def install(self, context: BridgeContext) -> bool:
        self.uninstall()
        version = _langfuse_version()
        major = _major(version)
        if not self.is_available() or major is None:
            logger.warning(
                "The Langfuse bridge needs langfuse >= %s.0 (found %s); Langfuse spans "
                + "will not be sent to Laminar. Upgrade with `pip install -U langfuse`.",
                MIN_LANGFUSE_MAJOR,
                version or "none",
            )
            return False
        attachment = ProviderAttachment(
            context.span_processor,
            # Never attach to Laminar's own provider.
            context.tracer_provider,
            bridge_version=version,
            # langfuse 5 JSON-encodes metadata values.
            json_encoded_metadata=major >= 5,
        )
        litellm = LiteLLMLangfuseBridge(attachment)
        self._attachment = attachment
        self._litellm = litellm
        try:
            attachment.attach_to_existing_langfuse_providers()
            attachment.patch_resource_manager()
            litellm.attach_to_existing_loggers()
            litellm.patch_logger_factory()
        except Exception:
            logger.warning("Failed to install the Langfuse bridge", exc_info=True)
            self.uninstall()
            return False
        return True

    @override
    def uninstall(self) -> None:
        try:
            if self._attachment is not None:
                self._attachment.unpatch_resource_manager()
                self._attachment.detach_all()
            if self._litellm is not None:
                self._litellm.unpatch_logger_factory()
                self._litellm.unwrap_loggers()
        except Exception:
            logger.debug("Failed to uninstall the Langfuse bridge", exc_info=True)
        finally:
            self._attachment = None
            self._litellm = None


__all__ = ["LangfuseBridge"]
