from pathlib import Path
from typing import Any
import csv
import orjson

from lmnr.sdk.log import get_default_logger

LOG = get_default_logger(__name__, verbose=False)


def _is_supported_file(file: Path) -> bool:
    """Check if a file is supported."""
    return file.suffix in [".json", ".csv", ".jsonl"]


def _collect_files(paths: list[Path], recursive: bool = False) -> list[Path]:
    """
    Collect all supported files from the given paths.

    Handles both files and directories. If a path is a directory,
    collects all supported files within it (recursively if specified).
    """
    collected_files = []

    for path in paths:
        if path.is_file():
            if _is_supported_file(path):
                collected_files.append(path)
            else:
                LOG.warning(f"Skipping unsupported file type: {path}")
        elif path.is_dir():
            for item in path.iterdir():
                if item.is_file() and _is_supported_file(item):
                    collected_files.append(item)
                elif recursive and item.is_dir():
                    # Recursively collect files from subdirectories
                    collected_files.extend(_collect_files([item], recursive=True))
        else:
            LOG.warning(f"Path does not exist or is not accessible: {path}")

    return collected_files


def _read_file(file: Path) -> list[dict[str, Any]]:
    """Read data from a single file and return as a list of dictionaries."""
    if file.suffix == ".json":
        result = orjson.loads(file.read_bytes())
        if isinstance(result, list):
            return result
        else:
            return [result]
    elif file.suffix == ".csv":
        return [dict(row) for row in csv.DictReader(file.read_text().splitlines())]
    elif file.suffix == ".jsonl":
        return [
            orjson.loads(line) for line in file.read_text().splitlines() if line.strip()
        ]
    else:
        raise ValueError(f"Unsupported file type: {file.suffix}")


def load_from_paths(paths: list[Path], recursive: bool = False) -> list[dict[str, Any]]:
    """
    Load data from all files in the specified paths.

    First collects all file paths, then reads each file's data.
    """
    files = _collect_files(paths, recursive)

    if not files:
        LOG.warning("No supported files found in the specified paths")
        return []

    LOG.info(f"Found {len(files)} file(s) to read")

    result = []
    for file in files:
        try:
            data = _read_file(file)
            result.extend(data)
            LOG.info(f"Read {len(data)} record(s) from {file}")
        except Exception as e:
            LOG.error(f"Error reading file {file}: {e}")
            raise

    return result


def parse_paths(paths: list[str]) -> list[Path]:
    """Parse paths."""
    return [Path(path) for path in paths]
