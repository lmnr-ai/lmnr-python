"""
Laminar HTTP client. Used to send data to/from the Laminar API.
"""

import httpx
from typing import Optional, TypeVar
from types import TracebackType

from lmnr.sdk.client.asynchronous.resources.agent import Agent
from lmnr.sdk.client.asynchronous.resources.browser_events import BrowserEvents
from lmnr.sdk.client.asynchronous.resources.evals import Evals
from lmnr.sdk.client.asynchronous.resources.pipeline import Pipeline
from lmnr.sdk.client.asynchronous.resources.semantic_search import SemanticSearch

_T = TypeVar("_T", bound="AsyncLaminarClient")


class AsyncLaminarClient:
    __base_url: str
    __project_api_key: str
    __client: httpx.AsyncClient = None

    def __init__(self, base_url: str, project_api_key: str):
        self.__base_url = base_url
        self.__project_api_key = project_api_key
        self.__client = httpx.AsyncClient(
            headers=self._headers(),
            timeout=350,
        )

        # Initialize resource objects
        self.__pipeline = Pipeline(
            self.__client, self.__base_url, self.__project_api_key
        )
        self.__semantic_search = SemanticSearch(
            self.__client, self.__base_url, self.__project_api_key
        )
        self.__agent = Agent(self.__client, self.__base_url, self.__project_api_key)
        self.__evals = Evals(self.__client, self.__base_url, self.__project_api_key)
        self.__browser_events = BrowserEvents(
            self.__client, self.__base_url, self.__project_api_key
        )

    @property
    def pipeline(self) -> Pipeline:
        """Get the Pipeline resource.

        Returns:
            Pipeline: The Pipeline resource instance.
        """
        return self.__pipeline

    @property
    def semantic_search(self) -> SemanticSearch:
        """Get the SemanticSearch resource.

        Returns:
            SemanticSearch: The SemanticSearch resource instance.
        """
        return self.__semantic_search

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

    def is_closed(self) -> bool:
        return self.__client.is_closed

    async def close(self) -> None:
        """Close the underlying HTTPX client.

        The client will *not* be usable after this.
        """
        await self.__client.aclose()

    async def __aenter__(self: _T) -> _T:
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        await self.close()

    def _headers(self) -> dict[str, str]:
        assert self.__project_api_key is not None, "Project API key is not set"
        return {
            "Authorization": "Bearer " + self.__project_api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
