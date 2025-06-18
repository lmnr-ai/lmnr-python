"""
Laminar HTTP client. Used to send data to/from the Laminar API.
"""

import httpx
import re
import uuid
from typing import Any, TypeVar
from types import TracebackType

from lmnr.sdk.client.synchronous.resources import (
    Agent,
    BrowserEvents,
    Evals,
    Tags,
)
from lmnr.sdk.utils import from_env
from lmnr.sdk.types import PartialEvaluationDatapoint, EvaluationResultDatapoint

_T = TypeVar("_T", bound="LaminarClient")


class LaminarClient:
    __base_url: str
    __project_api_key: str
    __client: httpx.Client = None

    # Resource properties
    __agent: Agent | None = None
    __evals: Evals | None = None
    __tags: Tags | None = None

    def __init__(
        self,
        base_url: str | None = None,
        project_api_key: str | None = None,
        port: int | None = None,
        timeout: int = 3600,
    ):
        """Initializer for the Laminar HTTP client.

        Args:
            base_url (str | None): base URL of the Laminar API. If not\
                provided, the LMNR_BASE_URL environment variable is used or we\
                default to "https://api.lmnr.ai".
            project_api_key (str | None): Laminar project API key. If not\
                provided, the LMNR_PROJECT_API_KEY environment variable is used.
            port (int | None, optional): port of the Laminar API HTTP server.\
                Overrides any port in the base URL.
                Defaults to None. If none is provided, the default port (443) will
                be used.
            timeout (int, optional): global timeout seconds for the HTTP client.\
                Applied to all httpx operations, i.e. connect, read, get_from_pool, etc.
                Defaults to 3600.
        """
        # If port is already in the base URL, use it as is
        base_url = base_url or from_env("LMNR_BASE_URL") or "https://api.lmnr.ai"
        if match := re.search(r":(\d{1,5})$", base_url):
            base_url = base_url[: -len(match.group(0))]
            if port is None:
                port = int(match.group(1))

        base_url = base_url.rstrip("/")
        self.__base_url = f"{base_url}:{port or 443}"
        self.__project_api_key = project_api_key or from_env("LMNR_PROJECT_API_KEY")
        if not self.__project_api_key:
            raise ValueError(
                "Project API key is not set. Please set the LMNR_PROJECT_API_KEY environment "
                "variable or pass project_api_key to the initializer."
            )
        self.__client = httpx.Client(
            headers=self._headers(),
            timeout=timeout,
        )

        # Initialize resource objects
        self.__agent = Agent(self.__client, self.__base_url, self.__project_api_key)
        self.__evals = Evals(self.__client, self.__base_url, self.__project_api_key)
        self.__browser_events = BrowserEvents(
            self.__client, self.__base_url, self.__project_api_key
        )
        self.__tags = Tags(self.__client, self.__base_url, self.__project_api_key)

    @property
    def agent(self) -> Agent:
        """Get the Agent resource.

        Returns:
            Agent: The Agent resource instance.
        """
        return self.__agent

    @property
    def _evals(self) -> Evals:
        """Get the Evals resource.

        Returns:
            Evals: The Evals resource instance.
        """
        return self.__evals

    @property
    def _browser_events(self) -> BrowserEvents:
        """Get the BrowserEvents resource.

        Returns:
            BrowserEvents: The BrowserEvents resource instance.
        """
        return self.__browser_events

    @property
    def tags(self) -> Tags:
        """Get the Tags resource.

        Returns:
            Tags: The Tags resource instance.
        """
        return self.__tags

    def shutdown(self):
        """Shutdown the client by closing underlying connections."""
        self.__client.close()

    def is_closed(self) -> bool:
        """Check if the client is closed.

        Returns:
            bool: True if the client is closed, False otherwise.
        """
        return self.__client.is_closed

    def close(self) -> None:
        """Close the underlying HTTPX client.

        The client will *not* be usable after this.
        """
        # If an error is thrown while constructing a client, self._client
        # may not be present
        if hasattr(self, "__client"):
            self.__client.close()

    def __enter__(self: _T) -> _T:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def _headers(self) -> dict[str, str]:
        assert self.__project_api_key is not None, "Project API key is not set"
        return {
            "Authorization": "Bearer " + self.__project_api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    # Evaluation methods for customers who manage their own iteration logic
    
    def create_evaluation(
        self,
        name: str | None = None,
        group_name: str | None = None,
    ) -> uuid.UUID:
        """
        Create a new evaluation and return its ID.
        
        Parameters:
            name (str | None, optional): Optional name of the evaluation.
            group_name (str | None, optional): An identifier to group evaluations.
        
        Returns:
            uuid.UUID: The evaluation ID.
        """
        evaluation = self._evals.init(name=name, group_name=group_name)
        return evaluation.id

    def create_datapoint(
        self,
        eval_id: uuid.UUID,
        data: Any,
        target: Any = None,
        metadata: dict[str, Any] | None = None,
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
        
        datapoint_id = uuid.uuid4()
        
        # Create a minimal datapoint first
        partial_datapoint = PartialEvaluationDatapoint(
            id=datapoint_id,
            data=data,
            target=target,
            index=index or 0,
            trace_id=trace_id or uuid.uuid4(),
            executor_span_id=uuid.uuid4(),  # Will be updated when executor runs
            metadata=metadata,
        )
        
        self._evals.save_datapoints(eval_id, [partial_datapoint])
        return datapoint_id

    def update_datapoint(
        self,
        eval_id: uuid.UUID,
        datapoint_id: uuid.UUID,
        executor_output: Any | None = None,
        scores: dict[str, float | int] | None = None,
    ) -> None:
        """
        Update a datapoint with evaluation results.
        
        Parameters:
            eval_id (uuid.UUID): The evaluation ID.
            datapoint_id (uuid.UUID): The datapoint ID.
            scores (dict[str, float | int]): The scores.
            executor_output (Any | None, optional): The executor output.

        Returns:
            None
        """
        
        self._evals.update_datapoint(eval_id, datapoint_id, scores, executor_output)
