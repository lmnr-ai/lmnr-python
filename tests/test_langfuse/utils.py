"""Shared helpers for the Langfuse bridge tests."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.util.types import AttributeValue

from lmnr.opentelemetry_lib.bridges.langfuse.processor import translate_span


def litellm_langfuse_otel_importable() -> bool:
    try:
        import litellm.integrations.langfuse.langfuse_otel  # noqa: F401 # pyright: ignore[reportUnusedImport]
    except Exception:
        return False
    return True


litellm_required = pytest.mark.skipif(
    not litellm_langfuse_otel_importable(),
    reason="litellm (with langfuse_otel) not importable on this interpreter",
)


class FakeSpan:
    """Minimal stand-in for a `ReadableSpan` handed to the translator."""

    def __init__(
        self,
        attributes: dict[str, AttributeValue | dict[str, int]],
        scope_name: str = "langfuse-sdk",
    ):
        self.attributes: dict[str, AttributeValue] = cast(Any, dict(attributes))
        scope = MagicMock()
        scope.name = scope_name
        self.instrumentation_scope: MagicMock | None = scope


def translate(span: FakeSpan, json_encoded_metadata: bool = False) -> dict[str, AttributeValue]:
    """`span`'s attributes with the bridge's translation applied on top."""
    return {
        **span.attributes,
        **translate_span(cast(ReadableSpan, cast(object, span)), json_encoded_metadata),
    }
