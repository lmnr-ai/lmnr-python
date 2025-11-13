"""
Laminar HTTP client. Used to send data to/from the Laminar API.
"""

import httpx
import re
from typing import TypeVar
from types import TracebackType

from lmnr.sdk.client.asynchronous.resources import (
    AsyncAgent,
    AsyncBrowserEvents,
    AsyncEvals,
    AsyncTags,
    AsyncEvaluators,
)
from lmnr.sdk.client.asynchronous.resources.datasets import AsyncDatasets
from lmnr.sdk.utils import from_env

_T = TypeVar("_T", bound="AsyncLaminarClient")


class AsyncLaminarClient:
    __base_url: str
    __project_api_key: str
    __client: httpx.AsyncClient = None

    def __init__(
        self,
        base_url: str | None = None,
        project_api_key: str | None = None,
        port: int | None = None,
        timeout: int = 3600,
    ):
        """Initializer for the Laminar HTTP client.

        Args:
            base_url (str | None): base URL of the Laminar API. If not
                provided, the LMNR_BASE_URL environment variable is used or we
                default to "https://api.lmnr.ai".
            project_api_key (str | None): Laminar project API key. If not
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

        self.__client = httpx.AsyncClient(
            headers=self._headers(),
            timeout=timeout,
            # Context: If the server responds with a 413, the connection becomes
            # poisoned and freezes on subsequent requests, and there is no way
            # to recover or recycle such connection.
            # Setting max_keepalive_connections to 0 will resolve this, but is
            # less efficient, as it will create a new connection
            # (not client, so still better) for each request.
            #
            # Note: from my experiments with a simple python server, forcing the
            # server to read/consume the request payload from the socket seems
            # to resolve this, but I haven't figured out how to do that in our
            # real actix-web backend server and whether it makes sense to do so.
            #
            # TODO: investigate if there are better ways to fix this rather than
            # setting keepalive_expiry to 0. Other alternative: migrate to
            # requests + aiohttp.
            #
            # limits=httpx.Limits(
            #     max_keepalive_connections=0,
            #     keepalive_expiry=0,
            # ),
        )

        # Initialize resource objects
        self.__agent = AsyncAgent(
            self.__client, self.__base_url, self.__project_api_key
        )
        self.__evals = AsyncEvals(
            self.__client, self.__base_url, self.__project_api_key
        )
        self.__evaluators = AsyncEvaluators(
            self.__client, self.__base_url, self.__project_api_key
        )
        self.__browser_events = AsyncBrowserEvents(
            self.__client, self.__base_url, self.__project_api_key
        )
        self.__tags = AsyncTags(self.__client, self.__base_url, self.__project_api_key)
        self.__datasets = AsyncDatasets(
            self.__client, self.__base_url, self.__project_api_key
        )

    @property
    def agent(self) -> AsyncAgent:
        """Get the Agent resource.

        Returns:
            AsyncAgent: The Agent resource instance.
        """
        return self.__agent

    @property
    def evals(self) -> AsyncEvals:
        """Get the Evals resource.

        Returns:
            AsyncEvals: The Evals resource instance.
        """
        return self.__evals

    @property
    def _browser_events(self) -> AsyncBrowserEvents:
        """Get the BrowserEvents resource.

        Returns:
            AsyncBrowserEvents: The BrowserEvents resource instance.
        """
        return self.__browser_events

    @property
    def tags(self) -> AsyncTags:
        """Get the Tags resource.

        Returns:
            AsyncTags: The Tags resource instance.
        """
        return self.__tags

    @property
    def evaluators(self) -> AsyncEvaluators:
        """Get the Evaluators resource.

        Returns:
            AsyncEvaluators: The Evaluators resource instance.
        """
        return self.__evaluators

    @property
    def datasets(self) -> AsyncDatasets:
        """Get the Datasets resource.

        Returns:
            AsyncDatasets: The Datasets resource instance.
        """
        return self.__datasets

    @property
    def is_closed(self) -> bool:
        return self.__client.is_closed

    async def close(self) -> None:
        """Close the underlying HTTPX client.

        The client will *not* be usable after this.
        """
        if hasattr(self, "__client"):
            await self.__client.aclose()

    async def __aenter__(self: _T) -> _T:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    def _headers(self) -> dict[str, str]:
        assert self.__project_api_key is not None, "Project API key is not set"
        return {
            "Authorization": "Bearer " + self.__project_api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
