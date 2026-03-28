"""SQL query command for the Laminar CLI."""

import sys
from argparse import Namespace, _SubParsersAction

import httpx

from lmnr.cli.output import print_error, print_json, print_table
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient


def setup_sql_parser(subparsers: _SubParsersAction) -> None:
    """Setup the sql subcommand parser."""
    from lmnr.cli import setup_laminar_args

    parser_sql = subparsers.add_parser(
        "sql",
        description="Execute a SQL query against the project's data",
        help="Execute a SQL query against the project's data",
    )
    parser_sql.add_argument(
        "query",
        help="SQL query string to execute (e.g., \"SELECT count() FROM traces\")",
    )
    parser_sql.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output results as JSON instead of a table",
    )
    setup_laminar_args(parser_sql)


async def handle_sql_command(args: Namespace) -> None:
    """Execute a SQL query and display results."""
    try:
        client = AsyncLaminarClient(
            project_api_key=args.project_api_key,
            base_url=args.base_url,
            port=args.port,
        )
    except ValueError as e:
        print_error(
            str(e),
            "Set LMNR_PROJECT_API_KEY or pass --project-api-key.",
        )
        sys.exit(1)

    try:
        results = await client.sql.query(args.query)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            print_error(
                "Authentication failed.",
                "Check that LMNR_PROJECT_API_KEY is set correctly.",
            )
        elif e.response.status_code == 400:
            print_error(
                f"Invalid query: {e.response.text}",
                "Check your SQL syntax.",
            )
        else:
            print_error(f"API error [{e.response.status_code}]: {e.response.text}")
        sys.exit(1)
    except httpx.ConnectError:
        print_error(
            f"Could not connect to Laminar API at {args.base_url}.",
            "Check your --base-url or LMNR_BASE_URL.",
        )
        sys.exit(1)
    except Exception as e:
        print_error(f"Failed to execute query: {e}")
        sys.exit(1)
    finally:
        await client.close()

    if not results:
        print("No results returned.")
        return

    if args.json:
        print_json(results)
    else:
        # Dynamically build columns from the first row's keys
        columns = []
        for key in results[0].keys():
            # Heuristic column width: min 10, max 40, based on header length
            width = max(10, min(40, len(key) + 4))
            columns.append((key, key, width))

        print_table(results, columns)
        print(f"\n{len(results)} row(s) returned.")
