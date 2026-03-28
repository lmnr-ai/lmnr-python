"""``lmnr sql <query>`` command.

Executes arbitrary ClickHouse SQL via the SDK's async SQL resource.
The query engine automatically scopes queries to the project.
"""

from __future__ import annotations

import asyncio
import sys
from argparse import ArgumentParser, Namespace, _SubParsersAction

from lmnr.cli.formatter import OutputFormatter
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient


# Schema reference printed with --schema
_SCHEMA = """\
Available tables and columns (project-scoped automatically):

  spans
    span_id, name, span_type (DEFAULT/LLM/TOOL), start_time, end_time,
    duration, input_cost, output_cost, total_cost, input_tokens,
    output_tokens, total_tokens, request_model, response_model, model,
    trace_id, provider, path, input, output, status, parent_span_id,
    attributes, tags

  traces
    id, start_time, end_time, input_tokens, output_tokens, total_tokens,
    input_cost, output_cost, total_cost, duration, metadata, session_id,
    user_id, status, top_span_id, top_span_name, top_span_type,
    trace_type, tags, has_browser_session

  evaluation_datapoints
    id, evaluation_id, data, target, metadata, executor_output, index,
    trace_id, group_id, scores, created_at, dataset_id,
    dataset_datapoint_id, dataset_datapoint_created_at

  dataset_datapoints
    id, created_at, dataset_id, data, target, metadata

  signal_events
    id, signal_id, trace_id, run_id, name, payload, timestamp

  signal_runs
    signal_id, job_id, trigger_id, run_id, trace_id, status, event_id,
    updated_at
"""


# -- Parser setup ------------------------------------------------------------

def setup_sql_parser(subparsers: _SubParsersAction) -> None:
    """Register the ``sql`` subcommand."""
    sql_parser: ArgumentParser = subparsers.add_parser(
        "sql",
        description="Execute a SQL query against your project data",
        help="Execute SQL query",
    )
    sql_parser.add_argument(
        "query",
        nargs="*",
        help="SQL query (multi-word allowed without quotes)",
    )
    sql_parser.add_argument(
        "--schema", action="store_true",
        help="Show available table schema and exit",
    )


# -- Handler -----------------------------------------------------------------

async def _sql_execute(args: Namespace, formatter: OutputFormatter) -> None:
    """Execute SQL query and format results."""
    query = " ".join(args.query)
    if not query.strip():
        print("Error: No SQL query provided.", file=sys.stderr)
        print("Usage: lmnr sql \"SELECT ...\"", file=sys.stderr)
        sys.exit(1)

    async with AsyncLaminarClient(
        base_url=args.api_url,
        project_api_key=args.api_key,
    ) as client:
        data = await client.sql.query(query)

    formatter.output(data)


def handle_sql_command(args: Namespace, formatter: OutputFormatter) -> None:
    """Dispatch sql command."""
    if getattr(args, "schema", False):
        print(_SCHEMA)
        return

    if not getattr(args, "query", None):
        print("Error: No SQL query provided.", file=sys.stderr)
        print("Usage: lmnr sql \"SELECT ...\"", file=sys.stderr)
        print("Run 'lmnr sql --schema' for available tables.", file=sys.stderr)
        sys.exit(1)

    asyncio.run(_sql_execute(args, formatter))
