"""
Laminar HTTP client. Used to send data to/from the Laminar API.
"""

import re
import httpx
from typing import Optional, TypeVar
from types import TracebackType

from lmnr.sdk.client.synchronous.resources import (
    Agent,
    BrowserEvents,
    Evals,
    Pipeline,
    SemanticSearch,
)

_T = TypeVar("_T", bound="LaminarClient")


class LaminarClient:
    __base_url: str
    __project_api_key: str
    __client: httpx.Client = None

    # Resource properties
    __pipeline: Optional[Pipeline] = None
    __semantic_search: Optional[SemanticSearch] = None
    __agent: Optional[Agent] = None
    __evals: Optional[Evals] = None

    def __init__(
        self,
        base_url: str,
        project_api_key: str,
        port: Optional[int] = None,
    ):
        """Initializer for the Laminar HTTP client.

        Args:
            base_url (str): base URL of the Laminar API. If you include a port,
                the `port` argument will be ignored.
            project_api_key (str): Laminar project API key
            port (Optional[int], optional): port of the Laminar API HTTP server.
                Defaults to None. If none is provided, the default port (443) will
                be used.
        """
        # If port is already in the base URL, use it as is
        if re.search(r":\d{1,5}$", base_url):
            self.__base_url = base_url
        else:
            self.__base_url = f"{base_url}:{port or 443}"
        # Remove trailing slash from base URL
        self.__base_url = re.sub(r"/$", "", self.__base_url)
        self.__project_api_key = project_api_key
        self.__client = httpx.Client(
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
        if hasattr(self, "_client"):
            self.__client.close()

    def __enter__(self: _T) -> _T:
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()

    def _headers(self) -> dict[str, str]:
        assert self.__project_api_key is not None, "Project API key is not set"
        return {
            "Authorization": "Bearer " + self.__project_api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
