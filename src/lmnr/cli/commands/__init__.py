"""Command registry for the Laminar CLI hybrid smart-output commands.

Each command module exposes a ``setup_parser`` function to register its
argparse subparser, and one or more handler functions invoked by the CLI
dispatch loop.
"""

from argparse import _SubParsersAction

from lmnr.cli.commands.traces import setup_traces_parser
from lmnr.cli.commands.evals import setup_evals_parser
from lmnr.cli.commands.sql import setup_sql_parser
from lmnr.cli.commands.status import setup_status_parser


def register_commands(subparsers: _SubParsersAction) -> None:
    """Register all Team 3 hybrid-output commands with the CLI parser."""
    setup_traces_parser(subparsers)
    setup_evals_parser(subparsers)
    setup_sql_parser(subparsers)
    setup_status_parser(subparsers)
