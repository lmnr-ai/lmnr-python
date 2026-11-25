"""LangSmith -> Laminar bridge.

LangSmith's SDK isn't built on OpenTelemetry: it sends runs to LangSmith's
REST API. The bridge taps the runs in process, at the point where LangSmith's
client has sampled and redacted them and is about to queue them for sending,
and records each one as a Laminar span. What LangSmith sends is unchanged.

- `tap.py`: where and how the LangSmith SDK is hooked.
- `runs.py`: merging a run's create and update halves into one span.
- `translate.py`: pure run -> Laminar attribute conversions.
"""

from __future__ import annotations

from packaging.version import InvalidVersion, Version
from typing_extensions import override

from lmnr.opentelemetry_lib.bridges.base import Bridge, BridgeContext, LaminarBridge
from lmnr.opentelemetry_lib.bridges.langsmith.runs import LangSmithRunRecorder
from lmnr.opentelemetry_lib.bridges.langsmith.tap import LangSmithTap
from lmnr.opentelemetry_lib.tracing.instruments import Instruments
from lmnr.opentelemetry_lib.utils.package_check import get_package_version, is_package_installed
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

# The first release whose `Client._create_run` / `_update_run` see every run
# LangSmith sends; before it, batched runs were queued in `create_run`.
MIN_LANGSMITH_VERSION = "0.4.15"


def _langsmith_version() -> str | None:
    if not is_package_installed("langsmith"):
        return None
    return get_package_version("langsmith")


def _supported(version: str | None) -> bool:
    if version is None:
        return False
    try:
        return Version(version) >= Version(MIN_LANGSMITH_VERSION)
    except InvalidVersion:
        return False


class LangSmithBridge(LaminarBridge):
    bridge: Bridge = Bridge.LANGSMITH
    requirement: str = f"langsmith >= {MIN_LANGSMITH_VERSION} (`pip install -U langsmith`)"
    # What LangSmith's wrappers, integrations and LangChain tracer record:
    # provider SDK calls (directly or under LangChain chat models), LangChain
    # and LangGraph runs, and the agent SDKs LangSmith integrates with. The
    # langgraph instrumentation records no spans (it publishes the graph's
    # topology, which the bridge puts on the graph's run), so it stays on.
    conflicting_instruments: frozenset[Instruments] = frozenset(
        {
            Instruments.ANTHROPIC,
            Instruments.CLAUDE_AGENT,
            Instruments.COHERE,
            Instruments.DEEPAGENTS,
            Instruments.GOOGLE_ADK,
            Instruments.GOOGLE_AIPLATFORM,
            Instruments.GOOGLE_GENAI,
            Instruments.GROQ,
            Instruments.LANGCHAIN,
            Instruments.LITELLM,
            Instruments.MCP,
            Instruments.OPENAI,
            Instruments.OPENAI_AGENTS,
            Instruments.OPENROUTER,
        }
    )
    # LangSmith's `otel` / `hybrid` tracing modes send through the global
    # tracer provider when one is set. If it were Laminar's, those spans would
    # go to Laminar instead of LangSmith.
    requires_private_tracer_provider: bool = True

    def __init__(self) -> None:
        self._tap: LangSmithTap | None = None

    @override
    def is_available(self) -> bool:
        return _supported(_langsmith_version())

    @property
    def installed(self) -> bool:
        return self._tap is not None

    @override
    def install(self, context: BridgeContext) -> bool:
        self.uninstall()
        version = _langsmith_version()
        if not _supported(version):
            logger.warning(
                "The LangSmith bridge needs langsmith >= %s (found %s); LangSmith runs "
                + "will not be sent to Laminar. Upgrade with `pip install -U langsmith`.",
                MIN_LANGSMITH_VERSION,
                version or "none",
            )
            return False
        recorder = LangSmithRunRecorder(
            context.span_processor, context.tracer_provider, bridge_version=version
        )
        tap = LangSmithTap(recorder)
        try:
            if not tap.install():
                tap.uninstall()
                return False
        except Exception:
            logger.warning("Failed to install the LangSmith bridge", exc_info=True)
            tap.uninstall()
            return False
        self._tap = tap
        return True

    @override
    def uninstall(self) -> None:
        try:
            if self._tap is not None:
                self._tap.uninstall()
        except Exception:
            logger.debug("Failed to uninstall the LangSmith bridge", exc_info=True)
        finally:
            self._tap = None


__all__ = ["LangSmithBridge"]
