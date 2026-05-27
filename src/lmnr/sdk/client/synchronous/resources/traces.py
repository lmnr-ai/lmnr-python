"""Resource for operating on existing traces."""

import uuid
from typing import Any

from lmnr.sdk.client.synchronous.resources.base import BaseResource
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.utils import format_id

logger = get_default_logger(__name__)


class Traces(BaseResource):
    """Resource for post-factum operations on existing traces."""

    def push_metadata(
        self,
        trace_id: str | int | uuid.UUID,
        metadata: dict[str, Any],
    ) -> None:
        """Push a metadata patch to an existing trace.

        The patch is shallow-merged server-side into the trace's existing
        metadata (`existing || patch`, last-write-wins per top-level key).
        Useful for attaching post-factum signals — quality scores, human
        edits, triage labels — to a trace that has already finished. The
        patch does NOT extend ``end_time`` or change tokens / cost / top
        span / tags / span names. ``num_spans`` is incremented by 1 (paid
        by the virtual span that carried the patch through the ingestion
        queue) so the new ClickHouse row beats the prior version on
        ``ReplacingMergeTree(num_spans)``. No row is added to the
        ``spans`` table.

        Compared to ``Laminar.set_trace_metadata`` (which sets metadata on
        the currently in-flight trace via OpenTelemetry attributes), this
        method operates on a finished trace by trace id, so it must be
        called after the trace has been flushed.

        Args:
            trace_id: The trace id to push metadata to. Accepts a UUID, an
                int (interpreted as a 128-bit OTel trace id), or a UUID
                string.
            metadata: The metadata patch. Top-level keys are merged into
                the trace's existing metadata. Must be non-empty (the
                server rejects empty patches with 400).

        Raises:
            ValueError: If the metadata is empty or the server returns a
                non-success status other than 404. A 404 is logged as a
                warning (the trace may not have been flushed yet) and the
                call returns without raising.

        Example:
            ```python
            from lmnr import Laminar, LaminarClient, observe

            Laminar.initialize()
            client = LaminarClient()

            @observe()
            def generate():
                return "draft response"

            generate()
            trace_id = Laminar.get_trace_id()
            Laminar.flush()

            client.traces.push_metadata(
                trace_id,
                {"score": 0.85, "reviewer": "alice", "needs_review": False},
            )
            ```
        """
        if not metadata:
            raise ValueError("metadata must be a non-empty dict")

        formatted_trace_id = format_id(trace_id)

        response = self._client.post(
            self._base_url + "/v1/traces/metadata",
            json={"traceId": formatted_trace_id, "metadata": metadata},
            headers=self._headers(),
        )

        if response.status_code == 404:
            logger.warning(
                f"Trace {formatted_trace_id} not found. The trace may not have "
                "been flushed yet — call Laminar.flush() and retry."
            )
            return

        if response.status_code != 200:
            raise ValueError(
                f"Failed to push trace metadata: [{response.status_code}] {response.text}"
            )
