from argparse import ArgumentParser, _SubParsersAction
import asyncio

from lmnr.cli.datasets import handle_datasets_command
from lmnr.cli.evals import run_evaluation
from lmnr.cli.rules import add_cursor_rules
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.utils import from_env

LOG = get_default_logger(__name__)
EVAL_DIR = "evals"
DEFAULT_DATASET_PULL_BATCH_SIZE = 100
DEFAULT_DATASET_PUSH_BATCH_SIZE = 100


def setup_eval_parser(subparsers: _SubParsersAction) -> None:
    """Setup the eval subcommand parser."""
    parser_eval = subparsers.add_parser(
        "eval",
        description="Run an evaluation",
        help="Run an evaluation",
    )
    parser_eval.add_argument(
        "file",
        nargs="*",
        help="Files or a file containing the evaluation to run. "
        + "If no file name is provided, all evaluation files in the `evals` directory are run as long "
        + "as they match *_eval.py or eval_*.py",
        default=[],
    )
    parser_eval.add_argument(
        "--continue-on-error",
        action="store_true",
        default=False,
        help="Continue execution upon errors",
    )
    parser_eval.add_argument(
        "--output-file",
        help="Output file to write the results to. Outputs are written in JSON format.",
        nargs="?",
    )


def setup_add_cursor_rules_parser(subparsers: _SubParsersAction) -> None:
    """Setup the add-cursor-rules subcommand parser."""
    subparsers.add_parser(
        "add-cursor-rules",
        description="Download laminar.mdc file and add it to .cursor/rules",
        help="Download laminar.mdc file and add it to .cursor/rules",
    )


def setup_laminar_args(parser: ArgumentParser) -> None:
    """Setup the laminar arguments parser."""
    parser.add_argument(
        "--project-api-key",
        help="[Optional] Project API key to use for the command. "
        + "If no project API key is provided, the project API key will be read "
        + "from the environment variable LMNR_PROJECT_API_KEY.",
        default=from_env("LMNR_PROJECT_API_KEY"),
    )
    parser.add_argument(
        "--base-url",
        help="[Optional] Base URL to use for the command. "
        + "If no base URL is provided, the base URL will be read from the "
        + "'LMNR_BASE_URL' environment variable or we default to 'https://api.lmnr.ai'.",
        default=from_env("LMNR_BASE_URL") or "https://api.lmnr.ai",
    )
    parser.add_argument(
        "--port",
        help="[Optional] Port to use for the command. "
        + "If no port is provided, the port defaults to '443'.",
        type=int,
    )


def setup_datasets_list_parser(subparsers: _SubParsersAction) -> None:
    """Setup the datasets list subcommand parser."""
    subparsers.add_parser(
        "list",
        description="List datasets",
        help="List datasets",
    )


def setup_datasets_push_parser(subparsers: _SubParsersAction) -> None:
    """Setup the datasets push subcommand parser."""
    parser_datasets_push: ArgumentParser = subparsers.add_parser(
        "push",
        description="Push datapoints to an existing dataset",
        help="Push datapoints to an existing dataset",
    )
    parser_datasets_push.add_argument(
        "--name",
        "-n",
        help="Name of the dataset to push data to. Exactly one of name or id must be provided.",
        default=None,
    )
    parser_datasets_push.add_argument(
        "--id",
        help="ID of the dataset to push data to. Exactly one of name or id must be provided.",
        default=None,
    )
    parser_datasets_push.add_argument(
        "paths",
        nargs="*",
        help="Paths to the files or directories containing the data to push to the dataset. "
        + "Supported formats: JSON, CSV, JSONL",
    )
    parser_datasets_push.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        default=False,
        help="Recursively read all files in the directories and their subdirectories.",
    )
    parser_datasets_push.add_argument(
        "--batch-size",
        type=int,
        help="Batch size to push data in. If no batch size is provided, "
        + f"data is pushed in batches of '{DEFAULT_DATASET_PUSH_BATCH_SIZE}'.",
        default=DEFAULT_DATASET_PUSH_BATCH_SIZE,
    )


def setup_datasets_pull_parser(subparsers: _SubParsersAction) -> None:
    """Setup the datasets pull subcommand parser."""
    parser_datasets_pull: ArgumentParser = subparsers.add_parser(
        "pull",
        description="Pull data from a dataset",
        help="Pull data from a dataset",
    )
    parser_datasets_pull.add_argument(
        "--name",
        "-n",
        help="Name of the dataset to pull data from",
        default=None,
    )
    parser_datasets_pull.add_argument(
        "--id",
        help="ID of the dataset to pull data from",
        default=None,
    )
    parser_datasets_pull.add_argument(
        "output_path",
        help="Path to the file to save the data to. "
        + "If no path is provided, data is printed to the console in the format "
        + "specified by '--output-format'.",
        nargs="?",
    )
    parser_datasets_pull.add_argument(
        "--output-format",
        choices=["json", "csv", "jsonl"],
        help="Output format to save the data to. "
        + "If no format is provided, it is inferred from the file extension.",
    )
    parser_datasets_pull.add_argument(
        "--batch-size",
        type=int,
        help="Batch size to pull data in. If no batch size is provided, "
        + f"data is pulled in batches of '{DEFAULT_DATASET_PULL_BATCH_SIZE}'.",
        default=DEFAULT_DATASET_PULL_BATCH_SIZE,
    )
    parser_datasets_pull.add_argument(
        "--limit",
        type=int,
        help="Limit the number of data points to pull. "
        + "If no limit is provided, all data points are pulled.",
    )
    parser_datasets_pull.add_argument(
        "--offset",
        type=int,
        help="Offset the number of data points to pull. "
        + "If no offset is provided, data is pulled from the beginning.",
    )


def setup_datasets_create_parser(subparsers: _SubParsersAction) -> None:
    """Setup the datasets create subcommand parser."""
    parser_datasets_create: ArgumentParser = subparsers.add_parser(
        "create",
        description="Create a dataset from input files and download it in Laminar format",
        help="Create a dataset from input files and download it in Laminar format",
    )
    parser_datasets_create.add_argument(
        "name",
        help="Name of the dataset to create",
    )
    parser_datasets_create.add_argument(
        "paths",
        nargs="+",
        help="Paths to the files or directories containing the data to push to the dataset. "
        + "Supported formats: JSON, CSV, JSONL",
    )
    parser_datasets_create.add_argument(
        "-o",
        "--output-file",
        required=True,
        help="Path to the file to save the pulled data to",
    )
    parser_datasets_create.add_argument(
        "--output-format",
        choices=["json", "csv", "jsonl"],
        help="Output format to save the data to. "
        + "If no format is provided, it is inferred from the output file extension.",
    )
    parser_datasets_create.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        default=False,
        help="Recursively read all files in the directories and their subdirectories.",
    )
    parser_datasets_create.add_argument(
        "--batch-size",
        type=int,
        help="Batch size to push/pull data in. If no batch size is provided, "
        + f"data is processed in batches of '{DEFAULT_DATASET_PUSH_BATCH_SIZE}'.",
        default=DEFAULT_DATASET_PUSH_BATCH_SIZE,
    )


def setup_datasets_parser(subparsers: _SubParsersAction) -> None:
    """Setup the datasets subcommand parser and its subcommands."""
    parser_datasets: ArgumentParser = subparsers.add_parser(
        "datasets",
        description="Manage datasets",
        help="Manage datasets",
    )

    setup_laminar_args(parser_datasets)

    parser_datasets_subparsers = parser_datasets.add_subparsers(
        title="command",
        dest="command",
    )

    # Setup all dataset subcommands
    setup_datasets_list_parser(parser_datasets_subparsers)
    setup_datasets_push_parser(parser_datasets_subparsers)
    setup_datasets_pull_parser(parser_datasets_subparsers)
    setup_datasets_create_parser(parser_datasets_subparsers)


def cli() -> None:
    """Main CLI entry point."""
    parser = ArgumentParser(
        prog="lmnr",
        description="CLI for Laminar. "
        + "Call `lmnr [subcommand] --help` for more information on each subcommand.",
    )

    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    # Setup all subcommand parsers
    setup_eval_parser(subparsers)
    setup_add_cursor_rules_parser(subparsers)
    setup_datasets_parser(subparsers)

    # Parse arguments and dispatch to appropriate handler
    parsed = parser.parse_args()

    if parsed.subcommand == "eval":
        asyncio.run(run_evaluation(parsed))
    elif parsed.subcommand == "add-cursor-rules":
        add_cursor_rules()
    elif parsed.subcommand == "datasets":
        asyncio.run(handle_datasets_command(parsed))
    else:
        parser.print_help()
