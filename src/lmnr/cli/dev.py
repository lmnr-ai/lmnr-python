"""
CLI command for running rollout development sessions.

This module provides the `lmnr dev` command for interactive LLM debugging
with caching and dynamic overrides.
"""

import asyncio
import importlib.util
import inspect
import json
import os
import re
import sys
import time
import uuid
from argparse import Namespace
from typing import Any

import httpx

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.rollout.cache_server import CacheServer
from lmnr.sdk.rollout.executor import SubprocessExecutor
from lmnr.sdk.rollout.sse_client import SSEClient
from lmnr.sdk.rollout_control import (
    ROLLOUT_MODE,
    get_entrypoints,
    clear_entrypoints,
)
from lmnr.sdk.types import RolloutParam
from lmnr.sdk.utils import from_env, get_frontend_url

logger = get_default_logger(__name__)
info_logger = get_default_logger(__name__ + ".user", verbose=False)


class DevCommandHandler:
    """
    Handler for the dev command, orchestrating all rollout components.
    """

    def __init__(
        self,
        file_path: str,
        function_name: str | None,
        project_api_key: str,
        base_url: str,
        http_port: int | None = None,
        grpc_port: int | None = None,
        frontend_port: int | None = None,
    ):
        self.file_path = os.path.abspath(file_path)
        self.function_name = function_name
        self.project_api_key = project_api_key
        self.base_url = base_url
        self.http_port = http_port
        self.grpc_port = grpc_port
        self.frontend_port = frontend_port

        # Components
        self.cache_server: CacheServer | None = None
        self.sse_client: SSEClient | None = None
        self.current_executor: SubprocessExecutor | None = None
        self.current_task: asyncio.Task | None = None
        self.laminar_client: AsyncLaminarClient | None = None
        self.session_id: str | None = None

        # State
        self.running = False
        self.file_observer: Observer | None = None
        self.entrypoint_params: list[RolloutParam] = []

    async def run(self) -> None:
        """Main execution flow."""

        try:
            # 1. Discover entrypoint
            entrypoint_name = await self._discover_entrypoint()
            if not entrypoint_name:
                return

            info_logger.info(f"Serving function: {entrypoint_name} in {self.file_path}")

            # 2. Start cache server
            logger.debug("Starting cache server...")
            self.cache_server = CacheServer(port=0)
            await self.cache_server.start()
            cache_url = self.cache_server.get_url()
            logger.debug(f"Cache server: {cache_url}")

            # 3. Generate session ID and create Laminar client
            self.session_id = str(uuid.uuid4())
            self.laminar_client = AsyncLaminarClient(
                base_url=self.base_url,
                project_api_key=self.project_api_key,
                port=self.http_port,
            )

            # 4. Set up SSE client (will create session on connect)
            self.sse_client = SSEClient(
                rollout_session_id=self.session_id,
                function_name=entrypoint_name,
                params=self.entrypoint_params,
                laminar_client=self.laminar_client,
            )
            self._register_sse_handlers()

            # 5. Set up file watching for hot reload
            self._setup_file_watcher()

            # 6. Start SSE connection (this will create session and run until interrupted)
            self.running = True
            logger.debug("👀 Watching for file changes...")
            logger.debug("⏳ Waiting for execution requests...")

            # Run SSE client (blocks until disconnected or interrupted)
            await self.sse_client.connect()

        except KeyboardInterrupt:
            logger.info("\n\nShutting down...")
        except Exception as e:
            logger.error(f"Error during dev session: {e}", exc_info=True)
        finally:
            await self._cleanup()

    def _parse_function_params(self, func: Any) -> list[RolloutParam]:
        """
        Parse function parameters to extract parameter metadata.

        For now, extracts just parameter names. Future enhancements can add
        type annotations, default values, etc.

        Args:
            func: The function to parse

        Returns:
            List[RolloutParam]: List of parameter metadata
        """
        try:
            sig = inspect.signature(func)
            params: list[RolloutParam] = []

            for param_name, param in sig.parameters.items():
                # Skip self/cls for methods
                if param_name in ("self", "cls"):
                    continue

                param_info: RolloutParam = {"name": param_name}

                # Check if required (no default value)
                if param.default == inspect.Parameter.empty:
                    param_info["required"] = True
                else:
                    param_info["required"] = False
                    # Store default as string representation
                    param_info["default"] = repr(param.default)

                # Extract type annotation if available
                if param.annotation != inspect.Parameter.empty:
                    param_info["type"] = str(param.annotation)

                params.append(param_info)

            return params

        except Exception as e:
            logger.warning(f"Failed to parse function parameters: {e}")
            return []

    async def _discover_entrypoint(self) -> str | None:
        """
        Discover and validate the rollout entrypoint function.

        Returns:
            str | None: Function name if found, None otherwise
        """
        # Clear previous registrations
        clear_entrypoints()

        # Enable rollout mode before importing
        token = ROLLOUT_MODE.set(True)

        try:
            # Add file directory to Python path
            file_dir = os.path.dirname(self.file_path)
            if file_dir not in sys.path:
                sys.path.insert(0, file_dir)

            # Import the module
            module_name = f"__lmnr_rollout_{uuid.uuid4().hex[:8]}"
            spec = importlib.util.spec_from_file_location(module_name, self.file_path)
            if spec is None or spec.loader is None:
                logger.error(f"Could not load module from {self.file_path}")
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Get registered entrypoints
            entrypoints = get_entrypoints()

            if not entrypoints:
                logger.error(
                    f"No rollout entrypoints found in {self.file_path}. "
                    "Add @observe(rollout_entrypoint=True) to a function."
                )
                return None

            # Handle function selection
            selected_name = None
            if self.function_name:
                if self.function_name not in entrypoints:
                    logger.error(
                        f"Function '{self.function_name}' not found. "
                        f"Available: {', '.join(entrypoints.keys())}"
                    )
                    return None
                selected_name = self.function_name
            else:
                # Auto-discover
                if len(entrypoints) > 1:
                    logger.error(
                        f"Multiple entrypoints found: {', '.join(entrypoints.keys())}. "
                        "Please specify one with --function"
                    )
                    return None
                selected_name = next(iter(entrypoints.keys()))

            # Parse parameters of the selected function
            if selected_name and selected_name in entrypoints:
                func = entrypoints[selected_name]
                self.entrypoint_params = self._parse_function_params(func)
                logger.debug(
                    f"Parsed {len(self.entrypoint_params)} parameters: "
                    f"{[p['name'] for p in self.entrypoint_params]}"
                )

            return selected_name

        except Exception as e:
            logger.error(f"Failed to import module: {e}", exc_info=True)
            return None
        finally:
            ROLLOUT_MODE.reset(token)

    def _register_sse_handlers(self) -> None:
        """Register event handlers for SSE client."""
        self.sse_client.on("handshake", self._handle_handshake)
        self.sse_client.on("run", self._handle_run)
        self.sse_client.on("stop", self._handle_stop)
        self.sse_client.on("heartbeat", lambda _: None)  # No-op, just keep alive

    async def _handle_handshake(self, data: dict[str, Any]) -> None:
        """Handle handshake event from backend."""
        project_id = data.get("project_id", "")
        logger.debug("Handshake received")

        frontend_url = get_frontend_url(self.base_url, self.frontend_port)
        info_logger.info(
            f"View session: {frontend_url}/project/{project_id}/rollout-sessions/{self.session_id}"
        )

    async def _execute_run(self, args: dict[str, Any]) -> None:
        """Execute the rollout function (runs as background task)."""
        try:
            # Update status to RUNNING
            await self.laminar_client.rollout.update_status(
                str(self.session_id), "RUNNING"
            )

            # Create executor
            self.current_executor = SubprocessExecutor(
                target_file=self.file_path,
                target_function=self.function_name or await self._discover_entrypoint(),
                rollout_session_id=self.session_id,
                cache_server_url=self.cache_server.get_url(),
                project_api_key=self.project_api_key,
                base_url=self.base_url,
                http_port=self.http_port,
                grpc_port=self.grpc_port,
            )

            # Execute in subprocess
            start_time = time.time()
            result = await self.current_executor.execute(args)
            elapsed = time.time() - start_time

            # Log result
            if result.get("success"):
                info_logger.info(f"  ✓ Completed in {elapsed:.1f}s")
            else:
                error = result.get("error", "Unknown error")
                info_logger.error(f"  ✗ Failed in {elapsed:.1f}s")
                info_logger.error(f"  Error: {error}")

            # Update status back to PENDING
            await self.laminar_client.rollout.update_status(
                str(self.session_id), "PENDING"
            )

        except asyncio.CancelledError:
            # Update status to PENDING on cancellation
            await self.laminar_client.rollout.update_status(
                str(self.session_id), "PENDING"
            )
            raise
        except Exception as e:
            logger.error(f"Error executing run: {e}", exc_info=True)
            await self.laminar_client.rollout.update_status(
                str(self.session_id), "PENDING"
            )
        finally:
            self.current_executor = None
            self.current_task = None

    async def _handle_run(self, data: dict[str, Any]) -> None:
        """Handle run event - execute the function with provided arguments."""
        # Cancel any existing task
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
            try:
                await self.current_task
            except asyncio.CancelledError:
                pass

        # Extract run event data
        trace_id = data.get("trace_id", "")
        path_to_count = data.get("path_to_count", {})
        args = data.get("args", {})
        overrides = data.get("overrides", {})

        info_logger.info(f"Executing with args: {args}")

        # Update cache server with path_to_count and overrides
        await self._update_cache_metadata(trace_id, path_to_count, overrides)

        # Create and store the execution task (runs in background)
        self.current_task = asyncio.create_task(self._execute_run(args))

    async def _update_cache_metadata(
        self, trace_id: str, path_to_count: dict[str, int], overrides: dict[str, Any]
    ) -> None:
        """
        Update cache server with metadata and fetch cached spans from backend if needed.

        Args:
            trace_id: Trace ID to fetch spans from (if provided)
            path_to_count: Mapping of paths to number of spans to cache
            overrides: Override parameters for each path
        """

        cache_url = self.cache_server.get_url()

        # Update cache server with metadata using async HTTP client
        try:
            async with httpx.AsyncClient() as client:
                # Update path_to_count
                await client.post(
                    f"{cache_url}/path_to_count",
                    json=path_to_count,
                )

                # Update overrides
                await client.post(
                    f"{cache_url}/overrides",
                    json=overrides,
                )
        except Exception as e:
            logger.warning(f"Failed to update cache metadata: {e}")
            return

        # Fetch and cache spans from previous trace if trace_id is provided
        if not trace_id or not trace_id.strip():
            info_logger.info("No spans in cache, starting fresh")
            return

        paths = list(path_to_count.keys())
        if not paths:
            info_logger.info("No spans to cache, starting fresh")
            return

        try:
            # Query spans from backend
            query = """
                SELECT name, input, output, attributes, path
                FROM spans
                WHERE trace_id = {traceId:UUID}
                  AND path IN {paths:String[]}
                ORDER BY start_time ASC
            """

            logger.debug(f"Querying spans from trace {trace_id}...")
            spans = await self.laminar_client.sql.query(
                query,
                {"traceId": trace_id, "paths": paths},
            )
            logger.debug(f"Received {len(spans)} spans from backend")

            # Group spans by path and cache first N per path
            spans_by_path: dict[str, list] = {}
            for span in spans:
                path = span.get("path", "")
                if path not in spans_by_path:
                    spans_by_path[path] = []
                spans_by_path[path].append(span)

            # Populate cache server with spans using async HTTP client

            async with httpx.AsyncClient() as client:
                for path, path_spans in spans_by_path.items():
                    max_count = path_to_count.get(path, 0)
                    spans_to_cache = path_spans[:max_count]

                    # Build cache entries
                    cache_entries = {}
                    for index, span in enumerate(spans_to_cache):
                        # Parse JSON fields
                        try:
                            parsed_input = (
                                json.loads(span["input"])
                                if isinstance(span.get("input"), str)
                                else span.get("input")
                            )
                        except (json.JSONDecodeError, KeyError):
                            parsed_input = span.get("input")

                        try:
                            parsed_output = span.get("output")
                            if not isinstance(parsed_output, str):
                                parsed_output = json.dumps(parsed_output)
                        except Exception:
                            parsed_output = str(span.get("output", ""))

                        try:
                            parsed_attributes = (
                                json.loads(span["attributes"])
                                if isinstance(span.get("attributes"), str)
                                else span.get("attributes", {})
                            )
                        except (json.JSONDecodeError, KeyError):
                            parsed_attributes = span.get("attributes", {})

                        cached_span = {
                            "name": span.get("name", ""),
                            "input": parsed_input,
                            "output": parsed_output,
                            "attributes": parsed_attributes,
                        }

                        cache_key = f"{path}:{index}"
                        cache_entries[cache_key] = cached_span

                    # Update cache server with spans
                    if cache_entries:
                        await client.post(
                            f"{cache_url}/spans",
                            json=cache_entries,
                        )
                        info_logger.info(
                            f"Cached {len(cache_entries)} spans for path: {path}"
                        )

        except Exception as e:
            logger.warning(f"Failed to fetch and cache spans: {e}", exc_info=True)

    async def _handle_stop(self, data: dict[str, Any]) -> None:
        """Handle stop event - cancel current execution."""
        # Capture the task reference before it can be set to None in the finally block
        logger.debug("Stopping execution...")
        task = self.current_task
        if task and not task.done():
            # Cancel the task
            task.cancel()
            # Also cancel the subprocess if it exists
            if self.current_executor:
                await self.current_executor.cancel()
            # Wait for the task to finish cancelling
            try:
                await task
                info_logger.info("Execution stopped")
            except asyncio.CancelledError:
                pass

    def _setup_file_watcher(self) -> None:
        """Set up file watching for hot reload."""

        class FileChangeHandler(FileSystemEventHandler):
            def __init__(self, handler: DevCommandHandler):
                self.handler = handler
                self._last_reload_time = 0
                self._reload_debounce = 0.5  # seconds

            def _should_ignore(self, path: str) -> bool:
                """Check if path should be ignored."""
                # Ignore patterns for common large directories and files
                ignore_patterns = [
                    ".venv",
                    ".virtualenv",
                    "venv",
                    "virtualenv",
                    "node_modules",
                    ".git",
                    ".hg",
                    ".svn",
                    "__pycache__",
                    ".pytest_cache",
                    ".mypy_cache",
                    ".tox",
                    ".eggs",
                    "*.egg-info",
                    ".DS_Store",
                    "*.pyc",
                    "*.pyo",
                    "*.pyd",
                    ".coverage",
                    "htmlcov",
                    "dist",
                    "build",
                    ".idea",
                    ".vscode",
                    "*.swp",
                    "*.swo",
                    "*~",
                ]

                path_parts = path.split(os.sep)
                for part in path_parts:
                    for pattern in ignore_patterns:
                        if pattern.startswith("*"):
                            # Simple glob pattern
                            if part.endswith(pattern[1:]):
                                return True
                        else:
                            # Exact match
                            if part == pattern:
                                return True
                return False

            def on_modified(self, event):
                if isinstance(event, FileModifiedEvent):
                    # Skip ignored paths
                    if self._should_ignore(event.src_path):
                        return

                    # Skip non-Python files and non .env files
                    if not event.src_path.endswith(
                        ".py"
                    ) and not event.src_path.startswith(".env"):
                        return

                    current_time = time.time()
                    if current_time - self._last_reload_time < self._reload_debounce:
                        return

                    self._last_reload_time = current_time
                    asyncio.run(self.handler._handle_file_change(event.src_path))

        event_handler = FileChangeHandler(self)
        self.file_observer = Observer()

        # Watch the directory containing the file, recursively
        watch_dir = os.path.dirname(self.file_path)
        if not watch_dir:
            watch_dir = os.getcwd()

        self.file_observer.schedule(
            event_handler,
            path=watch_dir,
            recursive=True,  # Watch subdirectories
        )
        self.file_observer.start()

    async def _handle_file_change(self, changed_file: str) -> None:
        """
        Handle file modification - reload module.

        Args:
            changed_file: Path to the file that was modified
        """
        # Get relative path for cleaner logging
        try:
            rel_path = os.path.relpath(changed_file, os.path.dirname(self.file_path))
        except ValueError:
            rel_path = changed_file

        logger.info(f"[Reload] File changed: {rel_path}")

        # Cancel current execution if any
        # Capture the task reference before it can be set to None
        task = self.current_task
        if task and not task.done():
            task.cancel()
            if self.current_executor:
                await self.current_executor.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Reload entrypoint
        await self._discover_entrypoint()

    async def _cleanup(self) -> None:
        """Clean up resources."""
        self.running = False

        # Stop file watcher
        if self.file_observer:
            self.file_observer.stop()
            self.file_observer.join()

        # Disconnect SSE
        if self.sse_client:
            await self.sse_client.disconnect()

        # Stop cache server
        if self.cache_server:
            await self.cache_server.stop()

        # Delete session
        if self.laminar_client and self.session_id:
            try:
                await self.laminar_client.rollout.delete_session(str(self.session_id))
            except Exception as e:
                logger.debug(f"Error deleting session: {e}")

        # Close client
        if self.laminar_client:
            await self.laminar_client.close()

        logger.info("Cleanup complete")


async def run_dev(args: Namespace) -> None:
    """
    Main entry point for the dev command.

    Args:
        args: Command line arguments from argparse
    """
    file_path = args.file
    function_name = getattr(args, "function", None)
    project_api_key = args.project_api_key or from_env("LMNR_PROJECT_API_KEY")
    base_url = args.base_url or from_env("LMNR_BASE_URL") or "https://api.lmnr.ai"
    frontend_port = getattr(args, "frontend_port", None)

    # Parse ports from base_url similar to TypeScript implementation
    port_match = re.search(r":(\d{1,5})$", base_url)
    http_port = args.port or (int(port_match.group(1)) if port_match else 443)
    grpc_port = args.grpc_port or 8443

    # Validate inputs
    if not project_api_key:
        logger.error(
            "Project API key is required. "
            "Set LMNR_PROJECT_API_KEY or use --project-api-key"
        )
        return

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return

    # Create and run handler
    handler = DevCommandHandler(
        file_path=file_path,
        function_name=function_name,
        project_api_key=project_api_key,
        base_url=base_url,
        http_port=http_port,
        grpc_port=grpc_port,
        frontend_port=frontend_port,
    )

    await handler.run()
