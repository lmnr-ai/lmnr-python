import asyncio
from argparse import ArgumentParser
from typing import Protocol

from lmnr.cli.datasets import DatasetsArgs, handle_datasets_command
from lmnr.cli.dev import run_dev
from lmnr.cli.evals import EvalArgs, run_evaluation
from lmnr.cli.rules import add_cursor_rules
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.utils import from_env

LOG = get_default_logger(__name__)
EVAL_DIR = "evals"
DEFAULT_DATASET_PULL_BATCH_SIZE = 100
DEFAULT_DATASET_PUSH_BATCH_SIZE = 100


class SubParsersAction(Protocol):
    """Structural stand-in for argparse's private `_SubParsersAction`."""

    def add_parser(
        self,
        name: str,
        *,
        help: str | None = None,
        description: str | None = None,
    ) -> ArgumentParser: ...


class _CliNamespace(  # pyright: ignore[reportUnsafeMultipleInheritance]
    DatasetsArgs, EvalArgs
):
    """Namespace subclass declaring the `dest` set by `parser.add_subparsers()`.

    `DatasetsArgs.__init__` and `EvalArgs.__init__` both cooperatively call
    `super().__init__(**kwargs)`, so the MRO chain (`_CliNamespace` ->
    `DatasetsArgs` -> `EvalArgs` -> `Namespace`) initializes both `paths` and
    `file` correctly; verified at runtime that neither list is shared across
    instances.
    """

    subcommand: str | None = None


def setup_eval_parser(subparsers: SubParsersAction):
    """Setup the eval subcommand parser."""
    parser_eval: ArgumentParser = subparsers.add_parser(
        "eval",
        description="Run an evaluation",
        help="Run an evaluation",
    )
    _file_action = parser_eval.add_argument(
        "file",
        nargs="*",
        help="Files or a file containing the evaluation to run. "
        + "If no file name is provided, all evaluation files in the `evals` directory are run as long "
        + "as they match *_eval.py or eval_*.py",
        default=[],
    )
    _continue_action = parser_eval.add_argument(
        "--continue-on-error",
        action="store_true",
        default=False,
        help="Continue execution upon errors",
    )
    _output_file_action = parser_eval.add_argument(
        "--output-file",
        help="Output file to write the results to. Outputs are written in JSON format.",
        nargs="?",
    )
    _frontend_port_action = parser_eval.add_argument(
        "--frontend-port",
        help="[Optional] Port for the frontend when running locally. "
        + "If no port is provided, defaults to '5667'.",
        type=int,
        default=None,
    )


def setup_dev_parser(subparsers: SubParsersAction):
    """Setup the dev subcommand parser."""
    parser_dev: ArgumentParser = subparsers.add_parser(
        "dev",
        description="Start a debuger session for interactive LLM debugging",
        help="Start a debugger session",
    )
    _file_action = parser_dev.add_argument(
        "file",
        help="Path to Python file containing entrypoint function(s)",
    )
    _fulnction_argument = parser_dev.add_argument(
        "--function",
        "-f",
        help="[Optional] Specific function name to use as entrypoint. "
        + "If not provided, automatically discovers the entrypoint. "
        + "Required if multiple entrypoints exist in the file.",
        default=None,
    )
    _grpc_port_action = parser_dev.add_argument(
        "--grpc-port",
        help="[Optional] Port to use for the gRPC server. "
        + "If no port is provided, the port defaults to '8443'.",
        type=int,
        default=8443,
    )
    _frontend_port_action = parser_dev.add_argument(
        "--frontend-port",
        help="[Optional] Port for the frontend when running locally. "
        + "If no port is provided, defaults to '5667'.",
        type=int,
        default=None,
    )
    setup_laminar_args(parser_dev)


def setup_add_cursor_rules_parser(subparsers: SubParsersAction):
    """Setup the add-cursor-rules subcommand parser."""
    _parser = subparsers.add_parser(
        "add-cursor-rules",
        description="Download laminar.mdc file and add it to .cursor/rules",
        help="Download laminar.mdc file and add it to .cursor/rules",
    )


def setup_laminar_args(parser: ArgumentParser):
    """Setup the laminar arguments parser."""
    _project_api_key_aciton = parser.add_argument(
        "--project-api-key",
        help="[Optional] Project API key to use for the command. "
        + "If no project API key is provided, the project API key will be read "
        + "from the environment variable LMNR_PROJECT_API_KEY.",
        default=from_env("LMNR_PROJECT_API_KEY"),
    )
    _base_url_action = parser.add_argument(
        "--base-url",
        help="[Optional] Base URL to use for the command. "
        + "If no base URL is provided, the base URL will be read from the "
        + "'LMNR_BASE_URL' environment variable or we default to 'https://api.lmnr.ai'.",
        default=from_env("LMNR_BASE_URL") or "https://api.lmnr.ai",
    )
    _port_action = parser.add_argument(
        "--port",
        help="[Optional] Port to use for the command. "
        + "If no port is provided, the port defaults to '443'.",
        type=int,
    )


def setup_datasets_list_parser(subparsers: SubParsersAction):
    """Setup the datasets list subcommand parser."""
    _parser = subparsers.add_parser(
        "list",
        description="List datasets",
        help="List datasets",
    )


def setup_datasets_push_parser(subparsers: SubParsersAction):
    """Setup the datasets push subcommand parser."""
    parser_datasets_push: ArgumentParser = subparsers.add_parser(
        "push",
        description="Push datapoints to an existing dataset",
        help="Push datapoints to an existing dataset",
    )
    _name_action = parser_datasets_push.add_argument(
        "--name",
        "-n",
        help="Name of the dataset to push data to. Exactly one of name or id must be provided.",
        default=None,
    )
    _id_action = parser_datasets_push.add_argument(
        "--id",
        help="ID of the dataset to push data to. Exactly one of name or id must be provided.",
        default=None,
    )
    _paths_action = parser_datasets_push.add_argument(
        "paths",
        nargs="*",
        help="Paths to the files or directories containing the data to push to the dataset. "
        + "Supported formats: JSON, CSV, JSONL",
    )
    _recursive_action = parser_datasets_push.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        default=False,
        help="Recursively read all files in the directories and their subdirectories.",
    )
    _batch_size_action = parser_datasets_push.add_argument(
        "--batch-size",
        type=int,
        help="Batch size to push data in. If no batch size is provided, "
        + f"data is pushed in batches of '{DEFAULT_DATASET_PUSH_BATCH_SIZE}'.",
        default=DEFAULT_DATASET_PUSH_BATCH_SIZE,
    )


def setup_datasets_pull_parser(subparsers: SubParsersAction):
    """Setup the datasets pull subcommand parser."""
    parser_datasets_pull: ArgumentParser = subparsers.add_parser(
        "pull",
        description="Pull data from a dataset",
        help="Pull data from a dataset",
    )
    _name_action = parser_datasets_pull.add_argument(
        "--name",
        "-n",
        help="Name of the dataset to pull data from",
        default=None,
    )
    _id_action = parser_datasets_pull.add_argument(
        "--id",
        help="ID of the dataset to pull data from",
        default=None,
    )
    _output_path_action = parser_datasets_pull.add_argument(
        "output_path",
        help="Path to the file to save the data to. "
        + "If no path is provided, data is printed to the console in the format "
        + "specified by '--output-format'.",
        nargs="?",
    )
    _output_format_action = parser_datasets_pull.add_argument(
        "--output-format",
        choices=["json", "csv", "jsonl"],
        help="Output format to save the data to. "
        + "If no format is provided, it is inferred from the file extension.",
    )
    _batch_size_action = parser_datasets_pull.add_argument(
        "--batch-size",
        type=int,
        help="Batch size to pull data in. If no batch size is provided, "
        + f"data is pulled in batches of '{DEFAULT_DATASET_PULL_BATCH_SIZE}'.",
        default=DEFAULT_DATASET_PULL_BATCH_SIZE,
    )
    _limit_action = parser_datasets_pull.add_argument(
        "--limit",
        type=int,
        help="Limit the number of data points to pull. "
        + "If no limit is provided, all data points are pulled.",
    )
    _offset_action = parser_datasets_pull.add_argument(
        "--offset",
        type=int,
        help="Offset the number of data points to pull. "
        + "If no offset is provided, data is pulled from the beginning.",
    )


def setup_datasets_create_parser(subparsers: SubParsersAction):
    """Setup the datasets create subcommand parser."""
    parser_datasets_create: ArgumentParser = subparsers.add_parser(
        "create",
        description="Create a dataset from input files and download it in Laminar format",
        help="Create a dataset from input files and download it in Laminar format",
    )
    _name_action = parser_datasets_create.add_argument(
        "name",
        help="Name of the dataset to create",
    )
    _paths_action = parser_datasets_create.add_argument(
        "paths",
        nargs="+",
        help="Paths to the files or directories containing the data to push to the dataset. "
        + "Supported formats: JSON, CSV, JSONL",
    )
    _output_file_action = parser_datasets_create.add_argument(
        "-o",
        "--output-file",
        required=True,
        help="Path to the file to save the pulled data to",
    )
    _output_format_action = parser_datasets_create.add_argument(
        "--output-format",
        choices=["json", "csv", "jsonl"],
        help="Output format to save the data to. "
        + "If no format is provided, it is inferred from the output file extension.",
    )
    _recursive_action = parser_datasets_create.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        default=False,
        help="Recursively read all files in the directories and their subdirectories.",
    )
    _batch_size_action = parser_datasets_create.add_argument(
        "--batch-size",
        type=int,
        help="Batch size to push/pull data in. If no batch size is provided, "
        + f"data is processed in batches of '{DEFAULT_DATASET_PUSH_BATCH_SIZE}'.",
        default=DEFAULT_DATASET_PUSH_BATCH_SIZE,
    )


def setup_datasets_parser(subparsers: SubParsersAction):
    """Setup the datasets subcommand parser and its subcommands."""
    parser_datasets: ArgumentParser = subparsers.add_parser(
        "datasets",
        description="Manage datasets",
        help="Manage datasets",
    )

    setup_laminar_args(parser_datasets)

    parser_datasets_subparsers: SubParsersAction = parser_datasets.add_subparsers(
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

    subparsers: SubParsersAction = parser.add_subparsers(
        title="subcommands", dest="subcommand"
    )

    # Setup all subcommand parsers
    setup_eval_parser(subparsers)
    setup_dev_parser(subparsers)
    setup_add_cursor_rules_parser(subparsers)
    setup_datasets_parser(subparsers)

    # Parse arguments and dispatch to appropriate handler
    parsed = parser.parse_args(namespace=_CliNamespace())
    subcommand = parsed.subcommand

    if subcommand == "eval":
        asyncio.run(run_evaluation(parsed))
    elif subcommand == "dev":
        asyncio.run(run_dev(parsed))
    elif subcommand == "add-cursor-rules":
        add_cursor_rules()
    elif subcommand == "datasets":
        asyncio.run(handle_datasets_command(parsed))
    else:
        parser.print_help()
