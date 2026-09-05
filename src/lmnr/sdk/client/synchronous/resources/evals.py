"""Evals resource for interacting with Laminar evaluations API."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, cast

from lmnr.sdk.client.synchronous.resources.base import BaseResource
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.utils import describe_response, json_dumps, serialize

# `lmnr.sdk.evaluations` (a package) transitively imports this client package
# (via `lmnr.sdk.datasets` -> `LaminarClient`), so importing
# `lmnr.sdk.evaluations.models` at module level here would be circular.
# Annotation-only uses are deferred via `TYPE_CHECKING` (safe under
# `from __future__ import annotations`); actual constructors/functions are
# imported lazily inside the methods that call them.
if TYPE_CHECKING:
    from lmnr.sdk.evaluations.models import (
        EvaluationResultDatapoint,
        InitEvaluationResponse,
        PartialEvaluationDatapoint,
    )

INITIAL_EVALUATION_DATAPOINT_MAX_DATA_LENGTH = 16_000_000  # 16MB
logger = get_default_logger(__name__)


class Evals(BaseResource):
    """Resource for interacting with Laminar evaluations API."""

    def init(
        self,
        name: str | None = None,
        group_name: str | None = None,
        metadata: dict[str, Any] | None = None,  # pyright:ignore[reportExplicitAny],
    ) -> InitEvaluationResponse:
        """Initialize a new evaluation.

        Args:
            name (str | None, optional): Name of the evaluation. Defaults to None.
            group_name (str | None, optional): Group name for the evaluation. Defaults to None.
            metadata (dict[str, Any] | None, optional): Metadata to associate with. Defaults to None.

        Returns:
            InitEvaluationResponse: The response from the initialization request.
        """
        from lmnr.sdk.evaluations.models import parse_init_evaluation_response

        response = self._client.post(
            self._base_url + "/v1/evals",
            json={
                "name": name,
                "groupName": group_name,
                "metadata": metadata,
            },
            headers=self._headers(),
        )
        if response.status_code != 200:
            if response.status_code == 401:
                raise ValueError("Unauthorized. Please check your project API key.")
            raise ValueError(
                f"Error initializing evaluation: {describe_response(response)}"
            )
        resp_json = cast(dict[str, str], response.json())
        return parse_init_evaluation_response(resp_json)

    def create_evaluation(
        self,
        name: str | None = None,
        group_name: str | None = None,
        metadata: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny],
    ) -> uuid.UUID:
        """
        Create a new evaluation and return its ID.

        Parameters:
            name (str | None, optional): Optional name of the evaluation.
            group_name (str | None, optional): An identifier to group evaluations.
            metadata (dict[str, Any] | None, optional): Metadata to associate with. Defaults to None.

        Returns:
            uuid.UUID: The evaluation ID.
        """
        evaluation = self.init(name=name, group_name=group_name, metadata=metadata)
        return evaluation["id"]

    def update_evaluation(
        self,
        eval_id: uuid.UUID,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny],
    ) -> InitEvaluationResponse:
        """Update an evaluation's name and/or metadata. The group ID is
        immutable. Fields left as None are kept unchanged.

        Args:
            eval_id (uuid.UUID): The evaluation ID.
            name (str | None, optional): New name of the evaluation.
                Defaults to None.
            metadata (dict[str, Any] | None, optional): New metadata for
                the evaluation. Defaults to None.

        Returns:
            InitEvaluationResponse: The updated evaluation.
        """
        from lmnr.sdk.evaluations.models import parse_init_evaluation_response

        response = self._client.post(
            self._base_url + f"/v1/evals/{eval_id}",
            json={
                "name": name,
                "metadata": metadata,
            },
            headers=self._headers(),
        )
        if response.status_code != 200:
            if response.status_code == 401:
                raise ValueError(
                    "Unauthorized. Please check your project API key."
                )
            raise ValueError(
                f"Error updating evaluation: {describe_response(response)}"
            )
        return parse_init_evaluation_response(cast(dict[str, str], response.json()))

    def create_datapoint(
        self,
        eval_id: uuid.UUID,
        data: Any,  # pyright: ignore[reportExplicitAny, reportAny],
        target: Any = None,  # pyright: ignore[reportExplicitAny, reportAny],
        metadata: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny],
        index: int | None = None,
        trace_id: uuid.UUID | None = None,
    ) -> uuid.UUID:
        """
        Create a datapoint for an evaluation.

        Parameters:
            eval_id (uuid.UUID): The evaluation ID.
            data: The input data for the executor.
            target: The target/expected output for evaluators.
            metadata (dict[str, Any] | None, optional): Optional metadata.
            index (int | None, optional): Optional index of the datapoint.
            trace_id (uuid.UUID | None, optional): Optional trace ID.

        Returns:
            uuid.UUID: The datapoint ID.
        """
        from lmnr.sdk.evaluations.models import PartialEvaluationDatapoint

        datapoint_id = uuid.uuid4()

        # Create a minimal datapoint first
        partial_datapoint = PartialEvaluationDatapoint(
            id=datapoint_id,
            data=data,
            target=target,  # pyright: ignore[reportAny]
            index=index or 0,
            trace_id=trace_id or uuid.uuid4(),
            executor_span_id=uuid.uuid4(),  # Will be updated when executor runs
            metadata=metadata,
        )

        self.save_datapoints(eval_id, [partial_datapoint])
        return datapoint_id

    def save_datapoints(
        self,
        eval_id: uuid.UUID,
        datapoints: list[EvaluationResultDatapoint | PartialEvaluationDatapoint],
        group_name: str | None = None,
    ):
        """Save evaluation datapoints.

        Args:
            eval_id (uuid.UUID): The evaluation ID.
            datapoints (list[EvaluationResultDatapoint | PartialEvaluationDatapoint]): The datapoints to save.
            group_name (str | None, optional): Group name for the datapoints. Defaults to None.

        Raises:
            ValueError: If there's an error saving the datapoints.
        """
        length = INITIAL_EVALUATION_DATAPOINT_MAX_DATA_LENGTH
        points = [datapoint.to_dict(max_data_length=length) for datapoint in datapoints]
        response = self._client.post(
            self._base_url + f"/v1/evals/{eval_id}/datapoints",
            json={
                "points": points,
                "groupName": group_name,
            },
            headers=self._headers(),
        )
        if response.status_code == 413:
            self._retry_save_datapoints(eval_id, datapoints, group_name)
            return

        if response.status_code != 200:
            raise ValueError(
                f"Error saving evaluation datapoints: {describe_response(response)}"
            )

    def update_datapoint(
        self,
        eval_id: uuid.UUID,
        datapoint_id: uuid.UUID,
        scores: dict[str, float | int],
        executor_output: Any | None = None,  # pyright: ignore[reportExplicitAny]
        trace_id: uuid.UUID | None = None,
    ) -> None:
        """Update a datapoint with evaluation results.

        Args:
            eval_id (uuid.UUID): The evaluation ID.
            datapoint_id (uuid.UUID): The datapoint ID.
            scores (dict[str, float | int]): The scores.
            executor_output (Any | None, optional): The executor output. Defaults to None.
            trace_id (uuid.UUID | None, optional): If provided, updates the trace ID associated with the datapoint. Defaults to None.
        """

        response = self._client.post(
            self._base_url + f"/v1/evals/{eval_id}/datapoints/{datapoint_id}",
            json={
                "executorOutput": (
                    json_dumps(serialize(executor_output))[
                        :INITIAL_EVALUATION_DATAPOINT_MAX_DATA_LENGTH
                    ]
                    if executor_output is not None
                    else None
                ),
                "scores": scores,
                "traceId": str(trace_id) if trace_id is not None else None,
            },
            headers=self._headers(),
        )

        if response.status_code != 200:
            raise ValueError(
                f"Error updating evaluation datapoint: {describe_response(response)}"
            )


    def _retry_save_datapoints(
        self,
        eval_id: uuid.UUID,
        datapoints: list[EvaluationResultDatapoint | PartialEvaluationDatapoint],
        group_name: str | None = None,
        initial_length: int = INITIAL_EVALUATION_DATAPOINT_MAX_DATA_LENGTH,
        max_retries: int = 20,
    ):
        retry = 0
        length = initial_length
        response = None
        while retry < max_retries:
            retry += 1
            length = length // 2
            logger.debug(
                f"Retrying save datapoints: {retry} of {max_retries}, length: {length}"
            )
            if length == 0:
                raise ValueError(
                    "Error saving evaluation datapoints: the server rejected the payload as too " +
                    "large even after truncating datapoint data to nothing. " +
                    f"Last server response: {describe_response(response)}"
                )
            points = [
                datapoint.to_dict(max_data_length=length) for datapoint in datapoints
            ]
            response = self._client.post(
                self._base_url + f"/v1/evals/{eval_id}/datapoints",
                json={
                    "points": points,
                    "groupName": group_name,
                },
                headers=self._headers(),
            )
            if response.status_code != 413:
                break
        if response is None or response.status_code != 200:
            raise ValueError(
                f"Error saving evaluation datapoints: {describe_response(response)}"
            )
