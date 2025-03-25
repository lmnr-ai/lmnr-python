"""
Laminar HTTP client. Used to send data to/from the Laminar API.
"""

import httpx
import re
from typing import Optional, TypeVar
from types import TracebackType

from lmnr.sdk.client.asynchronous.resources import (
    AsyncAgent,
    AsyncBrowserEvents,
    AsyncEvals,
    AsyncPipeline,
    AsyncSemanticSearch,
)
from lmnr.sdk.utils import from_env

_T = TypeVar("_T", bound="AsyncLaminarClient")


class AsyncLaminarClient:
    __base_url: str
    __project_api_key: str
    __client: httpx.AsyncClient = None

    def __init__(
        self,
        base_url: Optional[str] = None,
        project_api_key: Optional[str] = None,
        port: Optional[int] = None,
        timeout: int = 3600,
    ):
        """Initializer for the Laminar HTTP client.

        Args:
            base_url (str): base URL of the Laminar API.
            project_api_key (str): Laminar project API key
            port (Optional[int], optional): port of the Laminar API HTTP server.\
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

        self.__client = httpx.AsyncClient(
            headers=self._headers(),
            timeout=timeout,
        )

        # Initialize resource objects
        self.__pipeline = AsyncPipeline(
            self.__client, self.__base_url, self.__project_api_key
        )
        self.__semantic_search = AsyncSemanticSearch(
            self.__client, self.__base_url, self.__project_api_key
        )
        self.__agent = AsyncAgent(
            self.__client, self.__base_url, self.__project_api_key
        )
        self.__evals = AsyncEvals(
            self.__client, self.__base_url, self.__project_api_key
        )
        self.__browser_events = AsyncBrowserEvents(
            self.__client, self.__base_url, self.__project_api_key
        )

    @property
    def pipeline(self) -> AsyncPipeline:
        """Get the Pipeline resource.

        Returns:
            Pipeline: The Pipeline resource instance.
        """
        return self.__pipeline

    @property
    def semantic_search(self) -> AsyncSemanticSearch:
        """Get the SemanticSearch resource.

        Returns:
            SemanticSearch: The SemanticSearch resource instance.
        """
        return self.__semantic_search

    @property
    def agent(self) -> AsyncAgent:
        """Get the Agent resource.

        Returns:
            Agent: The Agent resource instance.
        """
        return self.__agent

    @property
    def _evals(self) -> AsyncEvals:
        """Get the Evals resource.

        Returns:
            Evals: The Evals resource instance.
        """
        return self.__evals

    @property
    def _browser_events(self) -> AsyncBrowserEvents:
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
