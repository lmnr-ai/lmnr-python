"""
Discover rollout function metadata for the TypeScript CLI.

This module implements the `lmnr discover` command that extracts function
signatures and parameter information from Python rollout entrypoints.
"""

import importlib
import importlib.util
import inspect
import json
import os
import sys

from argparse import Namespace
from lmnr.sdk.rollout.types import FunctionMetadata
from typing import Any


def python_type_to_json_type(annotation: Any) -> str:
    """
    Convert Python type annotation to JSON-compatible type string.

    Args:
        annotation: Python type annotation

    Returns:
        JSON type string: "string", "number", "boolean", "array", "object", or "any"
    """
    if annotation is inspect.Parameter.empty:
        return "any"

    # Handle string annotations (forward references)
    if isinstance(annotation, str):
        annotation_lower = annotation.lower()
        if annotation_lower in ("str", "string"):
            return "string"
        elif annotation_lower in ("int", "float", "number"):
            return "number"
        elif annotation_lower in ("bool", "boolean"):
            return "boolean"
        elif annotation_lower in ("list", "array"):
            return "array"
        elif annotation_lower in ("dict", "object"):
            return "object"
        return "any"

    # Get the actual type name
    type_name = getattr(annotation, "__name__", str(annotation))

    # Handle typing module types
    if hasattr(annotation, "__origin__"):
        origin = annotation.__origin__
        origin_name = getattr(origin, "__name__", str(origin))

        if origin_name in ("list", "List", "Sequence"):
            return "array"
        elif origin_name in ("dict", "Dict", "Mapping"):
            return "object"
        elif origin_name == "Union":
            # For Union types, try to infer the primary type
            args = getattr(annotation, "__args__", ())
            if args:
                # Use the first non-None type
                for arg in args:
                    if arg is not type(None):
                        return python_type_to_json_type(arg)
            return "any"

    # Direct type matching
    type_mapping = {
        "str": "string",
        "string": "string",
        "int": "number",
        "float": "number",
        "number": "number",
        "bool": "boolean",
        "boolean": "boolean",
        "list": "array",
        "List": "array",
        "typing.List": "array",
        "tuple": "array",
        "Tuple": "array",
        "typing.Tuple": "array",
        "array": "array",
        "dict": "object",
        "Dict": "object",
        "typing.Dict": "object",
        "object": "object",
        "typing.Object": "object",
    }

    return type_mapping.get(type_name, "any")


def extract_function_metadata(func) -> FunctionMetadata:
    """
    Extract metadata from a function.

    Args:
        func: Function to extract metadata from

    Returns:
        Dict with 'name' and 'params' keys
    """
    sig = inspect.signature(func)

    params = []
    for param_name, param in sig.parameters.items():
        # Skip self and cls parameters
        if param_name in ("self", "cls"):
            continue

        param_type = python_type_to_json_type(param.annotation)

        params.append({"name": param_name, "type": param_type})

    # Get function name - prefer the registered name if available
    func_name = getattr(func, "__name__", "unknown")

    return {"name": func_name, "params": params}


def load_module_from_file(file_path: str):
    """
    Load a Python module from a file path (script mode).

    Args:
        file_path: Path to the Python file

    Returns:
        Loaded module
    """
    file_abs_path = os.path.abspath(file_path)
    file_dir = os.path.dirname(file_abs_path)

    # Add file directory to path
    if file_dir not in sys.path:
        sys.path.insert(0, file_dir)

    module_name = f"__lmnr_discover_{os.path.basename(file_path).replace('.py', '')}"
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
        Loaded module
    """
    # Add current directory to path for module imports
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    return importlib.import_module(module_path)


def run_discover(args: Namespace) -> None:
    """
    Execute the discover command.

    Args:
        args: Parsed command line arguments
    """
    try:
        # Enable rollout mode to trigger entrypoint registration
        from lmnr.sdk.rollout_control import (
            ROLLOUT_MODE,
            clear_entrypoints,
            get_entrypoints,
        )

        clear_entrypoints()
        token = ROLLOUT_MODE.set(True)

        try:
            # Load the module based on mode
            if args.file:
                # Script mode
                load_module_from_file(args.file)
            elif args.module:
                # Module mode
                load_module_from_path(args.module)
            else:
                print(
                    json.dumps({"error": "Either --file or --module must be provided"}),
                    file=sys.stderr,
                )
                sys.exit(1)

            # Get registered entrypoints
            entrypoints = get_entrypoints()

            if not entrypoints:
                print(
                    json.dumps(
                        {
                            "error": "No rollout entrypoints found. "
                            "Add @observe(rollout_entrypoint=True) to a function."
                        }
                    ),
                    file=sys.stderr,
                )
                sys.exit(1)

            # Select function
            if args.function:
                if args.function not in entrypoints:
                    print(
                        json.dumps(
                            {
                                "error": f"Function '{args.function}' not found. "
                                f"Available: {', '.join(entrypoints.keys())}"
                            }
                        ),
                        file=sys.stderr,
                    )
                    sys.exit(1)
                func = entrypoints[args.function]
            else:
                # Auto-discover - if multiple, use the first one
                # (TS CLI will handle the multiple entrypoints case)
                func = next(iter(entrypoints.values()))

            # Extract metadata
            metadata = extract_function_metadata(func)

            # Output JSON to stdout
            print(json.dumps(metadata))
            sys.exit(0)

        finally:
            ROLLOUT_MODE.reset(token)

    except Exception as e:
        # Output error as JSON to stderr
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)
