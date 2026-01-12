"""
Subprocess executor for running rollout functions in isolation.

The executor spawns a subprocess that runs the target function, allowing for:
- Easy cancellation (SIGTERM/SIGKILL)
- Isolation from parent process
- Clean environment setup
"""

import asyncio
import json
import os
import sys
from typing import Any, Dict, Optional

from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

# Protocol prefix for structured output
PROTOCOL_PREFIX = "__LMNR_ROLLOUT__:"


class SubprocessExecutor:
    """
    Executor for running rollout functions in a subprocess.

    The subprocess runs the rollout function with provided arguments and
    returns the result via stdout using a simple JSON protocol.
    """

    def __init__(
        self,
        target_file: str,
        target_function: str,
        rollout_session_id: str,
        cache_server_url: str,
        project_api_key: str,
        base_url: str = "https://api.lmnr.ai",
        http_port: int | None = None,
        grpc_port: int | None = None,
    ):
        """
        Initialize the subprocess executor.

        Args:
            target_file: Path to the Python file containing the function
            target_function: Name of the function to execute
            rollout_session_id: Rollout session ID
            cache_server_url: URL of the cache server
            project_api_key: Project API key
            base_url: Base URL of the Laminar backend
            http_port: HTTP port for Laminar backend (defaults to 443)
            grpc_port: gRPC port for Laminar backend (defaults to 8443)
        """
        self.target_file = os.path.abspath(target_file)
        self.target_function = target_function
        self.session_id = rollout_session_id
        self.cache_server_url = cache_server_url
        self.project_api_key = project_api_key
        self.base_url = base_url
        self.http_port = http_port
        self.grpc_port = grpc_port

        self.process: Optional[asyncio.subprocess.Process] = None
        self.execution_timeout = 60.0  # Default 60 seconds

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the target function with provided arguments.

        Args:
            args: Dictionary of arguments to pass to the function

        Returns:
            Dict with 'success' and optionally 'error' keys.
            Note: The actual function return value is not included since it may
            not be JSON serializable. Function execution is traced separately.
        """
        # Prepare environment variables
        env = os.environ.copy()
        env["LMNR_ROLLOUT_SESSION_ID"] = self.session_id
        env["LMNR_ROLLOUT_STATE_SERVER_ADDRESS"] = self.cache_server_url
        env["LMNR_PROJECT_API_KEY"] = self.project_api_key
        env["LMNR_BASE_URL"] = self.base_url
        if self.http_port is not None:
            env["LMNR_HTTP_PORT"] = str(self.http_port)
        if self.grpc_port is not None:
            env["LMNR_GRPC_PORT"] = str(self.grpc_port)
        env["LMNR_ROLLOUT_SUBPROCESS"] = "true"

        try:
            # Create subprocess
            self.process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "lmnr.sdk.rollout.subprocess_runner",
                self.target_file,
                self.target_function,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            # Send arguments to subprocess via stdin
            args_json = json.dumps(args) + "\n"
            self.process.stdin.write(args_json.encode())
            await self.process.stdin.drain()
            self.process.stdin.close()

            # Stream output in real-time
            result = {"success": False, "error": "No output received"}

            async def stream_stdout():
                """Stream stdout line by line, passing through user print statements."""
                nonlocal result
                async for line in self.process.stdout:
                    line_str = line.decode("utf-8", errors="replace").rstrip()
                    if line_str.startswith(PROTOCOL_PREFIX):
                        # This is the protocol message with the result
                        json_str = line_str[len(PROTOCOL_PREFIX) :]
                        try:
                            result = json.loads(json_str)
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse subprocess output: {e}")
                    else:
                        # This is user output - pass it through
                        print(line_str)

            async def stream_stderr():
                """Stream stderr line by line to parent's stderr."""
                async for line in self.process.stderr:
                    line_str = line.decode("utf-8", errors="replace").rstrip()
                    # Pass through to stderr
                    print(line_str, file=sys.stderr)

            # Wait for completion with timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        stream_stdout(),
                        stream_stderr(),
                        self.process.wait(),
                    ),
                    timeout=self.execution_timeout,
                )
            except asyncio.TimeoutError:
                logger.error(f"Execution timeout ({self.execution_timeout}s)")
                await self.cancel()
                return {
                    "success": False,
                    "error": f"Execution timeout ({self.execution_timeout}s)",
                }

            # Check exit code
            if self.process.returncode != 0:
                logger.error(f"Subprocess exited with code {self.process.returncode}")
                if result.get("success"):
                    # Override success if process failed
                    result["success"] = False
                    result["error"] = (
                        f"Process exited with code {self.process.returncode}"
                    )

            return result

        except Exception as e:
            logger.error(f"Error executing subprocess: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }
        finally:
            self.process = None

    async def cancel(self) -> None:
        """
        Cancel the running subprocess.

        Sends SIGTERM, waits 5 seconds, then SIGKILL if needed.
        """
        if self.process is None:
            return

        logger.debug(f"Cancelling session {self.session_id}")

        try:
            # Send SIGTERM
            self.process.terminate()

            # Wait up to 5 seconds
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
                logger.debug("Subprocess terminated gracefully")
            except asyncio.TimeoutError:
                # Force kill
                logger.debug("Subprocess did not terminate, sending SIGKILL")
                self.process.kill()
                await self.process.wait()
                logger.debug("Subprocess killed")

        except ProcessLookupError:
            # Process already dead
            logger.debug("Process already terminated")
        except Exception as e:
            logger.error(f"Error cancelling executor: {e}", exc_info=True)
