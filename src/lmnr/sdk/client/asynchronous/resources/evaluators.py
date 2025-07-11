"""Evaluators resource for creating evaluator scores."""

import uuid
from typing import Any

from lmnr.sdk.client.asynchronous.resources.base import BaseAsyncResource
from lmnr.sdk.utils import format_id


class AsyncEvaluators(BaseAsyncResource):
    """Resource for creating evaluator scores."""

    async def score(
        self,
        *,
        name: str,
        trace_id: str | int | uuid.UUID | None = None,
        span_id: str | int | uuid.UUID | None = None,
        metadata: dict[str, Any] | None = None,
        score: float,
    ) -> None:
        """Create a score for a span.

        Args:
            name (str): Name of the score
            trace_id (str | int | uuid.UUID | None, optional): The trace ID to score (will be attached to root span)
            span_id (str | int | uuid.UUID | None, optional): The span ID to score
            metadata (dict[str, Any] | None, optional): Additional metadata. Defaults to None.
            score (float): The score value (float)

        Raises:
            ValueError: If there's an error creating the score.

        Example:
            Score by trace ID (will attach to root span):

            >>> await laminar_client.evaluators.score(
            ...     name="quality",
            ...     trace_id="trace-id-here",
            ...     score=0.95,
            ...     metadata={"model": "gpt-4"}
            ... )

            Score by span ID:

            >>> await laminar_client.evaluators.score(
            ...     name="relevance",
            ...     span_id="span-id-here",
            ...     score=0.87
            ... )
        """
        if trace_id is not None and span_id is not None:
            raise ValueError("Cannot provide both trace_id and span_id. Please provide only one.")
        if trace_id is None and span_id is None:
            raise ValueError("Either 'trace_id' or 'span_id' must be provided.")

        if trace_id is not None:
            formatted_trace_id = format_id(trace_id)
            payload = {
                "name": name,
                "traceId": formatted_trace_id,
                "metadata": metadata,
                "score": score,
                "source": "Code",
            }
        else:
            formatted_span_id = format_id(span_id)
            payload = {
                "name": name,
                "spanId": formatted_span_id,
                "metadata": metadata,
                "score": score,
                "source": "Code",
            }

        response = await self._client.post(
            self._base_url + "/v1/evaluators/score",
            json=payload,
            headers=self._headers(),
        )

        if response.status_code != 200:
            if response.status_code == 401:
                raise ValueError("Unauthorized. Please check your project API key.")
            raise ValueError(f"Error creating evaluator score: {response.text}")