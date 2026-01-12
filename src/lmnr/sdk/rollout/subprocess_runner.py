"""
Subprocess runner entry point for executing rollout functions.

This module is invoked as: python -m lmnr.sdk.rollout.subprocess_runner <file> <function>
It reads arguments from stdin, executes the function, and returns results via stdout.
"""

import asyncio
import importlib.util
import inspect
import json
import os
import sys
from typing import Any, Callable

# Protocol prefix for structured output
PROTOCOL_PREFIX = "__LMNR_ROLLOUT__:"


def output_metadata(success: bool, error: str = None) -> None:
    """
    Output execution metadata to stdout using protocol.

    Note: We don't include the actual function return value since it may not
    be JSON serializable. The function's execution is traced separately via
    the normal tracing pipeline.

    Args:
        success: Whether execution succeeded
        error: Error message (if failed)
    """
    data = {"success": success}
    if error is not None:
        data["error"] = error

    output = PROTOCOL_PREFIX + json.dumps(data)
    print(output, flush=True)


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
    func: Callable, raw_args: dict[str, Any] | list[Any]
) -> dict[str, Any]:
    """
    Prepare function arguments from raw args received from server.

    Args can come as:
    - Dict mapping param names to values
    - List of positional values

    This function:
    1. Tries to parse each value as JSON if it's a string
    2. Maps list args to parameter names using function signature
    3. Returns kwargs ready to call the function

    Args:
        func: Function to prepare args for
        raw_args: Raw arguments from server (dict or list)

    Returns:
        Dict of keyword arguments ready to unpack with **kwargs
    """
    # Get function signature
    sig = inspect.signature(func)
    param_names = [
        name for name in sig.parameters.keys() if name not in ("self", "cls")
    ]

    # Parse JSON values
    if isinstance(raw_args, dict):
        # Dict of param_name -> value
        return {key: try_parse_json(value) for key, value in raw_args.items()}

    elif isinstance(raw_args, list):
        # List of positional values - map to parameter names
        parsed_values = [try_parse_json(value) for value in raw_args]

        # Map to kwargs using parameter names
        kwargs = {}
        for i, value in enumerate(parsed_values):
            if i < len(param_names):
                kwargs[param_names[i]] = value
            # If more args than params, ignore extras (shouldn't happen)

        return kwargs

    else:
        # Unexpected format, return empty dict
        return {}


def main():
    """Main entry point for subprocess execution."""
    if len(sys.argv) < 3:
        output_metadata(
            success=False, error="Usage: subprocess_runner.py <file> <function>"
        )
        sys.exit(1)

    target_file = sys.argv[1]
    target_function = sys.argv[2]

    try:
        # Read arguments from stdin
        args_line = sys.stdin.readline()
        if not args_line:
            output_metadata(success=False, error="No arguments provided on stdin")
            sys.exit(1)

        args = json.loads(args_line)

        # Initialize Laminar with rollout metadata
        from lmnr import Laminar
        from lmnr.sdk.utils import from_env

        session_id = from_env("LMNR_ROLLOUT_SESSION_ID")
        project_api_key = from_env("LMNR_PROJECT_API_KEY")
        base_url = from_env("LMNR_BASE_URL") or "https://api.lmnr.ai"

        # Get ports from environment
        http_port_str = from_env("LMNR_HTTP_PORT")
        grpc_port_str = from_env("LMNR_GRPC_PORT")
        http_port = int(http_port_str) if http_port_str else None
        grpc_port = int(grpc_port_str) if grpc_port_str else None

        if not project_api_key:
            output_metadata(success=False, error="LMNR_PROJECT_API_KEY not set")
            sys.exit(1)

        # Initialize with rollout session metadata
        Laminar.initialize(
            project_api_key=project_api_key,
            base_url=base_url,
            http_port=http_port,
            grpc_port=grpc_port,
            metadata={"rollout_session_id": session_id} if session_id else None,
            disable_batch=True,
        )

        # Import the target module
        file_abs_path = os.path.abspath(target_file)
        file_dir = os.path.dirname(file_abs_path)

        # Add file directory to path
        if file_dir not in sys.path:
            sys.path.insert(0, file_dir)

        module_name = f"__lmnr_rollout_subprocess_{os.path.basename(target_file).replace('.py', '')}"
        spec = importlib.util.spec_from_file_location(module_name, file_abs_path)

        if spec is None or spec.loader is None:
            output_metadata(
                success=False, error=f"Could not load module from {target_file}"
            )
            sys.exit(1)

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Get the function
        if not hasattr(module, target_function):
            output_metadata(
                success=False,
                error=f"Function '{target_function}' not found in {target_file}",
            )
            sys.exit(1)

        func = getattr(module, target_function)

        # Prepare function arguments (parse JSON and map to params)
        func_kwargs = prepare_function_args(func, args)

        # Execute the function (handle both sync and async)
        from lmnr.sdk.utils import is_async

        if is_async(func):
            # Async function
            # Result is sent via tracing, we don't need to return it
            asyncio.run(func(**func_kwargs))
        else:
            # Sync function
            # Result is sent via tracing, we don't need to return it
            func(**func_kwargs)

        # Flush traces before returning
        Laminar.shutdown()

        # Return success metadata only (actual result is in trace)
        output_metadata(success=True)
        sys.exit(0)

    except KeyboardInterrupt:
        output_metadata(success=False, error="Execution interrupted")
        sys.exit(1)
    except Exception as e:
        import traceback

        error_msg = f"{type(e).__name__}: {str(e)}"
        stack_trace = traceback.format_exc()

        # Print full traceback to stderr for debugging
        print(stack_trace, file=sys.stderr, flush=True)

        # Return error metadata via protocol
        output_metadata(success=False, error=error_msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
