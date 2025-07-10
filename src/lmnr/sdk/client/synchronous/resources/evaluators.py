"""Evaluators resource for creating evaluator scores."""

import uuid
from typing import Any

from lmnr.sdk.client.synchronous.resources.base import BaseResource


class Evaluators(BaseResource):
    """Resource for creating evaluator scores."""

    def score(
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

            >>> evaluators.score(
            ...     name="quality",
            ...     trace_id="trace-id-here",
            ...     score=0.95,
            ...     metadata={"model": "gpt-4"}
            ... )

            Score by span ID:

            >>> evaluators.score(
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
            formatted_trace_id = self._format_id(trace_id)
            payload = {
                "name": name,
                "traceId": formatted_trace_id,
                "metadata": metadata,
                "score": score,
                "source": "SDK",
            }
        else:  # span_id is not None
            formatted_span_id = self._format_id(span_id)
            payload = {
                "name": name,
                "spanId": formatted_span_id,
                "metadata": metadata,
                "score": score,
                "source": "SDK",
            }

        response = self._client.post(
            self._base_url + "/v1/evaluators/score",
            json=payload,
            headers=self._headers(),
        )

        if response.status_code != 200:
            if response.status_code == 401:
                raise ValueError("Unauthorized. Please check your project API key.")
            raise ValueError(f"Error creating evaluator score: {response.text}")

    def _format_id(self, id_value: str | int | uuid.UUID) -> str:
        """Format trace/span ID to proper UUID string format.
        
        Args:
            id_value: The trace/span ID in various formats
            
        Returns:
            str: UUID string representation
            
        Raises:
            ValueError: If id_value cannot be converted to a valid UUID
        """
        if isinstance(id_value, uuid.UUID):
            return str(id_value)
        elif isinstance(id_value, int):
            return str(uuid.UUID(int=id_value))
        elif isinstance(id_value, str):
            try:
                uuid.UUID(id_value)
                return id_value
            except ValueError:
                try:
                    return str(uuid.UUID(int=int(id_value)))
                except ValueError:
                    try:
                        return str(uuid.UUID(int=int(id_value, 16)))
                    except ValueError:
                        raise ValueError(f"Invalid ID format: {id_value}")
        else:
            raise ValueError(f"Invalid ID type: {type(id_value)}")