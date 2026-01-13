"""
Worker entry point for rollout execution via external TypeScript CLI.

This module is invoked by the @lmnr-ai/cli package as:
    python3 -m lmnr.cli.worker

It reads configuration from stdin, executes the rollout function, and
communicates results back via stdout using a protocol.
"""

import asyncio
import importlib.util
import inspect
import json
import os
import sys
import traceback
from typing import Any

# Force unbuffered output for real-time logging
# This ensures log messages are streamed immediately to the parent process
if sys.stdout is not None:
    sys.stdout.reconfigure(line_buffering=True)
if sys.stderr is not None:
    sys.stderr.reconfigure(line_buffering=True)

from lmnr.sdk.utils import is_async
from lmnr import Laminar


# Protocol prefix for worker messages
WORKER_MESSAGE_PREFIX = "__LMNR_WORKER__:"


def send_message(message: dict[str, Any]) -> None:
    """
    Send a message to parent process via stdout.
    Uses a special prefix to distinguish from user's print/console.log output.

    Args:
        message: Message dict with 'type' and other fields
    """
    output = WORKER_MESSAGE_PREFIX + json.dumps(message)
    print(output, flush=True)


def log_info(message: str) -> None:
    """Send info log message to parent."""
    send_message({"type": "log", "level": "info", "message": message})


def log_debug(message: str) -> None:
    """Send debug log message to parent."""
    send_message({"type": "log", "level": "debug", "message": message})


def log_error(message: str) -> None:
    """Send error log message to parent."""
    send_message({"type": "log", "level": "error", "message": message})


def log_warn(message: str) -> None:
    """Send warning log message to parent."""
    send_message({"type": "log", "level": "warn", "message": message})


def send_result(data: Any) -> None:
    """Send successful result to parent."""
    send_message({"type": "result", "data": data})


def send_error(error: str, stack: str | None = None) -> None:
    """Send error to parent."""
    message = {"type": "error", "error": error}
    if stack:
        message["stack"] = stack
    send_message(message)


def try_parse_json(value: Any) -> Any:
    """
    Try to parse a value as JSON if it's a string.

    Args:
        value: Value to parse

    Returns:
        Parsed value if it was JSON string, otherwise original value
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def prepare_function_args(
    func, raw_args: dict[str, Any] | list[Any]
) -> tuple[list[Any], dict[str, Any]]:
    """
    Prepare function arguments from raw args received from CLI.

    Args can come as:
    - Dict mapping param names to values
    - List of positional values

    This function maps them to the correct order for calling the function.

    Args:
        func: Function to prepare args for
        raw_args: Raw arguments from CLI (dict or list)

    Returns:
        Tuple of (positional_args, keyword_args) ready to call function with
    """
    # Get function signature
    sig = inspect.signature(func)
    param_names = [
        name for name in sig.parameters.keys() if name not in ("self", "cls")
    ]

    # Parse JSON values
    if isinstance(raw_args, dict):
        # Dict of param_name -> value
        # Only include parameters that are actually provided
        # Let Python handle defaults for missing parameters
        kwargs = {}
        for name in param_names:
            if name in raw_args:
                kwargs[name] = try_parse_json(raw_args[name])
        return [], kwargs

    elif isinstance(raw_args, list):
        # List of positional values
        args = [try_parse_json(value) for value in raw_args]
        return args, {}

    else:
        # Unexpected format, return empty args/kwargs
        return [], {}


def load_module_from_file(file_path: str):
    """
    Load a Python module from a file path (script mode).

    Args:
        file_path: Path to the Python file

    Returns:
        Loaded module object
    """
    file_abs_path = os.path.abspath(file_path)
    file_dir = os.path.dirname(file_abs_path)

    # Add file directory to path
    if file_dir not in sys.path:
        sys.path.insert(0, file_dir)

    module_name = f"__lmnr_worker_{os.path.basename(file_path).replace('.py', '')}"
    spec = importlib.util.spec_from_file_location(module_name, file_abs_path)

    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def load_module_from_path(module_path: str):
    """
    Load a Python module from a module path (module mode).

    Args:
        module_path: Python module path (e.g., "src.myfile")

    Returns:
        Loaded module object
    """
    # Add current directory to path for module imports
    # This allows importing modules in the current directory
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    return importlib.import_module(module_path)


async def run_worker(config: dict[str, Any]) -> Any:
    """
    Main worker execution function.

    Args:
        config: WorkerConfig dict with filePath or modulePath, functionName, args, env, etc.

    Returns:
        Result of the rollout function
    """
    # Set environment variables
    env_vars = config.get("env", {})
    for key, value in env_vars.items():
        os.environ[key] = value

    # Get file path or module path
    file_path = config.get("filePath")
    module_path = config.get("modulePath")
    function_name = config.get("functionName")
    args = config.get("args", {})

    project_api_key = config.get("projectApiKey")
    base_url = config.get("baseUrl", "https://api.lmnr.ai")
    http_port = config.get("httpPort")
    grpc_port = config.get("grpcPort")

    if not project_api_key:
        # Try from environment as fallback
        project_api_key = os.environ.get("LMNR_PROJECT_API_KEY")

    if not project_api_key:
        raise ValueError("Project API key not provided in config or environment")

    log_debug("Initializing Laminar...")
    if not Laminar.is_initialized():
        Laminar.initialize(
            project_api_key=project_api_key,
            base_url=base_url,
            http_port=http_port,
            grpc_port=grpc_port,
            disable_batch=True,
        )

    # Import the target module with rollout mode enabled
    from lmnr.sdk.rollout_control import (
        ROLLOUT_MODE,
        get_entrypoints,
        clear_entrypoints,
    )

    # Clear any previous registrations
    clear_entrypoints()

    # Enable rollout mode before importing to trigger entrypoint registration
    token = ROLLOUT_MODE.set(True)

    try:
        # Load module based on mode (file path or module path)
        if file_path:
            # Script mode
            log_debug(f"Loading module from file: {file_path}")
            load_module_from_file(file_path)
            source_desc = file_path
        elif module_path:
            # Module mode
            log_debug(f"Loading module: {module_path}")
            load_module_from_path(module_path)
            source_desc = module_path
        else:
            raise ValueError("Config must contain either 'filePath' or 'modulePath'")

        # Get registered entrypoints
        entrypoints = get_entrypoints()

        if not entrypoints:
            raise ValueError(
                f"No rollout entrypoints found in {source_desc}. "
                "Add @observe(rollout_entrypoint=True) to a function."
            )

        # Select function
        if function_name:
            if function_name not in entrypoints:
                raise ValueError(
                    f"Function '{function_name}' not found. "
                    f"Available: {', '.join(entrypoints.keys())}"
                )
            func = entrypoints[function_name]
        else:
            # Auto-discover
            if len(entrypoints) > 1:
                raise ValueError(
                    f"Multiple entrypoints found: {', '.join(entrypoints.keys())}. "
                    "Please specify one with functionName in config."
                )
            func = next(iter(entrypoints.values()))

        func_name = function_name or next(iter(entrypoints.keys()))
        log_debug(f"Selected function: {func_name}")

    finally:
        ROLLOUT_MODE.reset(token)

    # Prepare function arguments
    func_args, func_kwargs = prepare_function_args(func, args)

    log_info(f"Calling function {func_name} with args: {json.dumps(args)}")

    if is_async(func):
        result = await func(*func_args, **func_kwargs)
    else:
        result = func(*func_args, **func_kwargs)

    log_info("Rollout function completed successfully")

    # Flush traces before returning
    log_debug("Flushing traces...")
    Laminar.flush()

    return result


def main() -> None:
    """Main entry point - read config from stdin and execute worker."""
    try:
        # Read configuration from stdin (single line JSON)
        config_line = sys.stdin.readline()
        if not config_line:
            send_error("No configuration received on stdin")
            sys.exit(1)

        try:
            config = json.loads(config_line)
        except json.JSONDecodeError as e:
            send_error(f"Failed to parse config JSON: {e}")
            sys.exit(1)

        # Execute the worker (async context)
        result = asyncio.run(run_worker(config))

        # Send result back to parent
        send_result(result)

        # Exit successfully
        sys.exit(0)

    except KeyboardInterrupt:
        send_error("Execution interrupted")
        sys.exit(1)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        stack_trace = traceback.format_exc()

        # Print full traceback to stderr for debugging
        print(stack_trace, file=sys.stderr, flush=True)

        # Send error via protocol
        send_error(error_msg, stack_trace)
        sys.exit(1)


if __name__ == "__main__":
    main()
