"""
HTTP cache server for storing and serving cached LLM spans and overrides.

The cache server stores:
- Cached spans indexed by path:index (e.g., "root.llm_call:0")
- Path-to-count mapping (e.g., {"root.llm_call": 2})
- Override parameters (e.g., {"model": "gpt-4", "temperature": 0.7})
"""

import asyncio
from typing import Any, Dict, Optional

from aiohttp import web

from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)


class CacheServer:
    """
    HTTP server for caching spans and serving them to instrumentations.

    The server runs on a dynamically assigned port and provides endpoints
    for storing and retrieving cached data.
    """

    def __init__(self, port: int = 0):
        """
        Initialize the cache server.

        Args:
            port: Port to bind to. Use 0 for automatic assignment (default).
        """
        self.port = port
        self.actual_port: Optional[int] = None
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None

        # In-memory storage with locks for thread safety
        self._lock = asyncio.Lock()
        self._cached_spans: Dict[str, Dict[str, Any]] = {}
        self._path_to_count: Dict[str, int] = {}
        self._overrides: Dict[str, Any] = {}

        # Setup routes
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Setup HTTP routes for the cache server."""
        self.app.router.add_get("/health", self._health_handler)
        self.app.router.add_post("/cached", self._get_cached_handler)
        self.app.router.add_get("/path_to_count", self._get_path_to_count_handler)
        self.app.router.add_get("/overrides", self._get_overrides_handler)
        self.app.router.add_post("/spans", self._update_spans_handler)
        self.app.router.add_post("/path_to_count", self._update_path_to_count_handler)
        self.app.router.add_post("/overrides", self._update_overrides_handler)

    async def start(self) -> int:
        """
        Start the cache server.

        Returns:
            int: The actual port the server is listening on
        """
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()

        self.site = web.TCPSite(self.runner, "127.0.0.1", self.port)
        await self.site.start()

        # Get the actual port (important when port=0)
        if self.site._server and self.site._server.sockets:
            self.actual_port = self.site._server.sockets[0].getsockname()[1]
        else:
            self.actual_port = self.port

        logger.debug(f"Cache server started on http://127.0.0.1:{self.actual_port}")
        return self.actual_port

    async def stop(self) -> None:
        """Stop the cache server and cleanup resources."""
        if self.runner:
            await self.runner.cleanup()
        logger.debug("Cache server stopped")

    def get_url(self) -> str:
        """
        Get the cache server URL.

        Returns:
            str: Full URL of the cache server
        """
        if self.actual_port is None:
            raise RuntimeError("Server not started yet")
        return f"http://127.0.0.1:{self.actual_port}"

    # ========================================================================
    # HTTP Handlers
    # ========================================================================

    async def _health_handler(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({"status": "ok"})

    async def _get_cached_handler(self, request: web.Request) -> web.Response:
        """
        Fetch a cached span by path and index.

        Request body: {"path": "root.llm_call", "index": 0}
        Response: {"span": {...}, "path_to_count": {...}, "overrides": {...}} or 404
        """
        try:
            data = await request.json()
            path = data.get("path", "")
            index = data.get("index", 0)

            cache_key = f"{path}:{index}"

            async with self._lock:
                cached_span = self._cached_spans.get(cache_key)
                path_to_count = self._path_to_count.copy()
                overrides = self._overrides.copy()

            if cached_span:
                return web.json_response(
                    {
                        "span": cached_span,
                        "path_to_count": path_to_count,
                        "overrides": overrides,
                    }
                )
            else:
                # Return metadata even if span not found
                return web.json_response(
                    {
                        "span": None,
                        "path_to_count": path_to_count,
                        "overrides": overrides,
                    }
                )

        except Exception as e:
            logger.error(f"Error in _get_cached_handler: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def _get_path_to_count_handler(self, request: web.Request) -> web.Response:
        """
        Get the path-to-count mapping.

        Response: {"root.llm_call": 2, "root.other": 1}
        """
        async with self._lock:
            path_to_count = self._path_to_count.copy()
        logger.debug(f"Returning path_to_count: {path_to_count}")
        return web.json_response(path_to_count)

    async def _get_overrides_handler(self, request: web.Request) -> web.Response:
        """
        Get current overrides.

        Response: {"model": "gpt-4", "temperature": 0.7, ...}
        """
        async with self._lock:
            overrides = self._overrides.copy()
        return web.json_response(overrides)

    async def _update_spans_handler(self, request: web.Request) -> web.Response:
        """
        Update cached spans from SSE events.

        Request body: {
            "root.llm_call:0": {"response": {...}, "metadata": {...}},
            "root.llm_call:1": {"response": {...}, "metadata": {...}}
        }
        """
        try:
            data = await request.json()

            async with self._lock:
                self._cached_spans.update(data)

            logger.debug(f"Updated {len(data)} cached spans")
            return web.json_response({"status": "ok", "count": len(data)})

        except Exception as e:
            logger.error(f"Error in _update_spans_handler: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def _update_path_to_count_handler(self, request: web.Request) -> web.Response:
        """
        Update path-to-count mapping from SSE events.

        Request body: {"root.llm_call": 2, "root.other": 1}
        """
        try:
            data = await request.json()

            async with self._lock:
                self._path_to_count = data

            logger.debug(f"Updated path_to_count with {len(data)} entries: {data}")
            return web.json_response({"status": "ok", "count": len(data)})

        except Exception as e:
            logger.error(f"Error in _update_path_to_count_handler: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def _update_overrides_handler(self, request: web.Request) -> web.Response:
        """
        Update overrides from SSE events.

        Request body: {"model": "gpt-4", "temperature": 0.7, ...}
        """
        try:
            data = await request.json()

            async with self._lock:
                self._overrides = data

            logger.debug(f"Updated overrides with {len(data)} parameters")
            return web.json_response({"status": "ok", "count": len(data)})

        except Exception as e:
            logger.error(f"Error in _update_overrides_handler: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)
