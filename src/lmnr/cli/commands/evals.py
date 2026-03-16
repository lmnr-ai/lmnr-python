"""``lmnr evals list`` command.

Queries ClickHouse evaluation_datapoints via the SDK's async SQL resource
and formats results through the smart OutputFormatter.
"""

from __future__ import annotations

import asyncio
import sys
from argparse import ArgumentParser, Namespace, _SubParsersAction

from lmnr.cli.formatter import OutputFormatter
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient


def _esc(value: str) -> str:
    """Escape a string for safe use in a SQL single-quoted literal."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


# -- Parser setup ------------------------------------------------------------

def setup_evals_parser(subparsers: _SubParsersAction) -> None:
    """Register the ``evals`` subcommand group."""
    evals_parser: ArgumentParser = subparsers.add_parser(
        "evals",
        description="Evaluation operations",
        help="List evaluations",
    )
    evals_sub = evals_parser.add_subparsers(
        title="evals commands",
        dest="evals_command",
    )

    list_parser = evals_sub.add_parser("list", help="List evaluations")
    list_parser.add_argument(
        "-n", "--limit", type=int, default=20,
        help="Number of evaluations to return (default: 20)",
    )
    list_parser.add_argument(
        "--group", type=str,
        help="Filter by group name",
    )


# -- Handlers ----------------------------------------------------------------

TABLE_COLUMNS = [
    "evaluation_id", "datapoints", "groups", "created_at", "last_updated",
]


async def _evals_list(args: Namespace, formatter: OutputFormatter) -> None:
    """List evaluations with datapoint counts."""
    limit = args.limit or 20

    query = (
        "SELECT "
        "evaluation_id, "
        "count() as datapoints, "
        "groupArray(distinct group_id) as groups, "
        "formatDateTime(min(created_at), '%Y-%m-%dT%H:%i:%S.%fZ') as created_at, "
        "formatDateTime(max(created_at), '%Y-%m-%dT%H:%i:%S.%fZ') as last_updated "
        "FROM evaluation_datapoints "
    )

    if args.group:
        query += f"WHERE group_id = '{_esc(args.group)}' "

    query += (
        "GROUP BY evaluation_id "
        "ORDER BY max(created_at) DESC "
        f"LIMIT {limit}"
    )

    async with AsyncLaminarClient(
        base_url=args.api_url,
        project_api_key=args.api_key,
    ) as client:
        data = await client.sql.query(query)

    formatter.output(data, columns=TABLE_COLUMNS)


def handle_evals_command(args: Namespace, formatter: OutputFormatter) -> None:
    """Dispatch evals subcommands."""
    cmd = getattr(args, "evals_command", None)
    if cmd == "list":
        asyncio.run(_evals_list(args, formatter))
    else:
        print("Usage: lmnr evals {list} ...", file=sys.stderr)
        print("Run 'lmnr evals --help' for details.", file=sys.stderr)
        sys.exit(1)
