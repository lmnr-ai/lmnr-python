from argparse import Namespace
from pathlib import Path
from typing import Any

import csv
import sys

import orjson

from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
from lmnr.sdk.datasets.file_utils import load_from_paths, parse_paths
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.types import Datapoint

LOG = get_default_logger(__name__, verbose=False)
DEFAULT_DATASET_PULL_BATCH_SIZE = 100
DEFAULT_DATASET_PUSH_BATCH_SIZE = 100


def _dump_json(data: Any, do_indent: bool = True) -> str:
    return orjson.dumps(
        data,
        option=(orjson.OPT_INDENT_2 if do_indent else 0)
        | orjson.OPT_SERIALIZE_DATACLASS
        | orjson.OPT_SERIALIZE_UUID
        | orjson.OPT_UTC_Z
        | orjson.OPT_NON_STR_KEYS,
    ).decode()


async def _pull_all_data(
    client: AsyncLaminarClient,
    identifier: dict,
    batch_size: int,
    offset: int = 0,
    limit: int | None = None,
) -> list[Datapoint]:
    """
    Pull all data from a dataset. This function does not close the client.

    Args:
        client: The AsyncLaminarClient instance
        identifier: Dict with either 'name' or 'id' key
        batch_size: Size of batches to pull
        offset: Starting offset for pulling data
        limit: Maximum number of items to pull (None for all)

    Returns:
        List of all pulled datapoints
    """
    has_more = True
    current_offset = offset
    stop_at = offset + limit if limit else None

    result: list[Datapoint] = []
    while has_more and (stop_at is None or current_offset < stop_at):
        data = await client.datasets.pull(
            **identifier,
            offset=current_offset,
            limit=batch_size,
        )
        result.extend(data.items)
        if stop_at is not None and current_offset + batch_size >= stop_at:
            has_more = False
        elif (
            data.total_count is not None
            and current_offset + batch_size >= data.total_count
        ):
            has_more = False
        current_offset += batch_size

    if limit is not None:
        return result[:limit]
    return result


def _write_data_to_file(
    data: list[Datapoint],
    output_path: Path,
    output_format: str | None = None,
) -> bool:
    """
    Write datapoints to a file.

    Args:
        data: List of datapoints to write
        output_path: Path to write the file to
        output_format: Optional format override (json, csv, jsonl)

    Returns:
        True if successful, False otherwise
    """
    if output_path.is_dir():
        LOG.error(f"Output path is a directory: {output_path}")
        return False

    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine output format
    format = output_format or output_path.suffix[1:]
    if output_format and output_format != output_path.suffix[1:]:
        LOG.warning(
            f"Output format {output_format} does not match file extension {output_path.suffix[1:]}"
        )

    if format not in ["json", "csv", "jsonl"]:
        LOG.error(f"Unsupported output format: {format}")
        return False

    # Write output file
    if format == "json":
        output_path.write_text(_dump_json([item.model_dump() for item in data]))
    elif format == "csv":
        if not data:
            LOG.error("No data to write to CSV")
            return False
        with output_path.open("w") as f:
            writer = csv.writer(f)
            keys = list(data[0].model_dump().keys())
            writer.writerow(keys)
            for item in data:
                writer.writerow([item.model_dump()[key] for key in keys])
    elif format == "jsonl":
        with output_path.open("w") as f:
            for item in data:
                f.write(_dump_json(item.model_dump(), do_indent=False) + "\n")

    return True


def _print_data_to_console(data: list[Datapoint], output_format: str = "json") -> bool:
    """
    Print datapoints to console.

    Args:
        data: List of datapoints to print
        output_format: Format to use (json, csv, jsonl)

    Returns:
        True if successful, False otherwise
    """
    if output_format not in ["json", "csv", "jsonl"]:
        LOG.error(f"Unsupported output format: {output_format}")
        return False

    if output_format == "json":
        print(_dump_json([item.model_dump() for item in data]))
    elif output_format == "csv":
        if not data:
            LOG.error("No data to print")
            return False
        writer = csv.writer(sys.stdout)
        keys = list(data[0].model_dump().keys())
        writer.writerow(keys)
        for item in data:
            writer.writerow([item.model_dump()[key] for key in keys])
    elif output_format == "jsonl":
        for item in data:
            print(_dump_json(item.model_dump(), do_indent=False))
    print()

    return True


async def handle_datasets_list(args: Namespace) -> None:
    """
    Handle datasets list command.

    Lists all datasets in a formatted table.
    """
    client = AsyncLaminarClient(
        project_api_key=args.project_api_key,
        base_url=args.base_url,
        port=args.port,
    )

    try:
        datasets = await client.datasets.list_datasets()
    except Exception as e:
        LOG.error(f"Failed to list datasets: {e}")
        return
    finally:
        await client.close()

    if not datasets:
        print("No datasets found.")
        return

    # Print table header
    id_width = 36  # UUID length
    created_at_width = 19  # YYYY-MM-DD HH:MM:SS format

    print(f"\n{'ID':<{id_width}}  {'Created At':<{created_at_width}}  Name")
    print(f"{'-' * id_width}  {'-' * created_at_width}  {'-' * 20}")

    # Print each dataset row
    for dataset in datasets:
        created_at_str = dataset.created_at.strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"{str(dataset.id):<{id_width}}  {created_at_str:<{created_at_width}}  {dataset.name}"
        )

    print(f"\nTotal: {len(datasets)} dataset(s)\n")


async def handle_datasets_push(args: Namespace) -> None:
    """
    Handle datasets push command.

    Pushes data from file(s) to an existing dataset.
    """
    if not args.name and not args.id:
        LOG.error("Either name or id must be provided")
        return
    if args.name and args.id:
        LOG.error("Only one of name or id must be provided")
        return
    identifier = {"name": args.name} if args.name else {"id": args.id}
    client = AsyncLaminarClient(
        project_api_key=args.project_api_key,
        base_url=args.base_url,
        port=args.port,
    )
    data = load_from_paths(parse_paths(args.paths), recursive=args.recursive)
    if len(data) == 0:
        LOG.warning("No data to push. Skipping")
        return
    try:
        await client.datasets.push(
            data,
            **identifier,
            batch_size=args.batch_size or DEFAULT_DATASET_PUSH_BATCH_SIZE,
        )
        LOG.info(f"Pushed {len(data)} data points to dataset {args.name or args.id}")
    except Exception as e:
        LOG.error(f"Failed to push dataset: {e}")
    finally:
        await client.close()


async def handle_datasets_pull(args: Namespace) -> None:
    """
    Handle datasets pull command.

    Pulls data from a dataset and saves it to a file.
    """
    if not args.name and not args.id:
        LOG.error("Either name or id must be provided")
        return
    if args.name and args.id:
        LOG.error("Only one of name or id must be provided")
        return
    identifier = {"name": args.name} if args.name else {"id": args.id}
    client = AsyncLaminarClient(
        project_api_key=args.project_api_key,
        base_url=args.base_url,
        port=args.port,
    )

    # Pull data from dataset
    try:
        result = await _pull_all_data(
            client=client,
            identifier=identifier,
            batch_size=args.batch_size or DEFAULT_DATASET_PULL_BATCH_SIZE,
            offset=args.offset or 0,
            limit=args.limit,
        )
    except Exception as e:
        LOG.error(f"Failed to pull dataset: {e}")
        return
    finally:
        await client.close()

    # Write to file or print to console
    if args.output_path:
        if not _write_data_to_file(
            data=result,
            output_path=Path(args.output_path),
            output_format=args.output_format,
        ):
            return
    else:
        if not _print_data_to_console(
            data=result,
            output_format=args.output_format or "json",
        ):
            return


async def handle_datasets_create(args: Namespace) -> None:
    """
    Handle datasets create command.

    Creates a dataset from input files, pushes the data to it, and then pulls it back
    in Laminar format to save to the output file.
    """
    client = AsyncLaminarClient(
        project_api_key=args.project_api_key,
        base_url=args.base_url,
        port=args.port,
    )

    # Load data from input files
    data = load_from_paths(parse_paths(args.paths), recursive=args.recursive)
    if len(data) == 0:
        LOG.warning("No data to push. Skipping")
        return

    # Push data to create/populate the dataset
    LOG.info(f"Pushing {len(data)} data points to dataset '{args.name}'...")
    try:
        await client.datasets.push(
            data,
            name=args.name,
            batch_size=args.batch_size or DEFAULT_DATASET_PUSH_BATCH_SIZE,
            create_dataset=True,
        )
        LOG.info(
            f"Successfully pushed {len(data)} data points to dataset '{args.name}'"
        )
    except Exception as e:
        LOG.error(f"Failed to create dataset: {e}")
        return

    # Pull data back from the dataset
    LOG.info(f"Pulling data from dataset '{args.name}'...")
    try:
        result = await _pull_all_data(
            client=client,
            identifier={"name": args.name},
            batch_size=args.batch_size or DEFAULT_DATASET_PULL_BATCH_SIZE,
            offset=0,
            limit=None,
        )
    except Exception as e:
        LOG.error(f"Failed to pull dataset after creation: {e}")
        return
    finally:
        await client.close()

    # Save to output file
    if not _write_data_to_file(
        data=result,
        output_path=Path(args.output_file),
        output_format=args.output_format,
    ):
        return

    LOG.info(
        f"Successfully created dataset '{args.name}' and saved {len(result)} data points to {args.output_file}"
    )


async def handle_datasets_command(args: Namespace) -> None:
    """
    Handle datasets subcommand dispatching.

    Dispatches to the appropriate handler based on the command.
    """
    if args.command == "list":
        await handle_datasets_list(args)
    elif args.command == "push":
        await handle_datasets_push(args)
    elif args.command == "pull":
        await handle_datasets_pull(args)
    elif args.command == "create":
        await handle_datasets_create(args)
    else:
        LOG.error(f"Unknown datasets command: {args.command}")
