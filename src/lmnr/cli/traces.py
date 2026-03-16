"""Traces commands for the Laminar CLI."""

import sys
from argparse import Namespace, _SubParsersAction
from typing import Any

import httpx

from lmnr.cli.output import (
    format_cost,
    format_duration_seconds,
    format_tokens,
    print_error,
    print_json,
    print_table,
    resolve_time_filters,
    truncate_id,
    validate_uuid,
)
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient


def setup_traces_parser(subparsers: _SubParsersAction) -> None:
    """Setup the traces subcommand parser with list and get subcommands."""
    from lmnr.cli import setup_laminar_args

    parser_traces = subparsers.add_parser(
        "traces",
        description="List and inspect traces",
        help="List and inspect traces",
    )

    traces_subparsers = parser_traces.add_subparsers(
        title="command",
        dest="traces_command",
    )

    # traces list
    parser_list = traces_subparsers.add_parser(
        "list",
        description="List recent traces",
        help="List recent traces",
    )
    parser_list.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max number of traces to return (default: 50)",
    )
    parser_list.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset for pagination (default: 0)",
    )
    parser_list.add_argument(
        "--status",
        help="Filter by status (e.g., 'error' or 'ok')",
        default=None,
    )
    parser_list.add_argument(
        "--start-time",
        help="ISO 8601 start time filter (e.g., '2026-03-15T00:00:00Z')",
        default=None,
    )
    parser_list.add_argument(
        "--end-time",
        help="ISO 8601 end time filter (e.g., '2026-03-16T00:00:00Z')",
        default=None,
    )
    parser_list.add_argument(
        "--past-hours",
        type=float,
        help="Look back N hours from now (alternative to --start-time/--end-time)",
        default=None,
    )
    parser_list.add_argument(
        "--trace-type",
        default="DEFAULT",
        help="Filter by trace type: DEFAULT, EVALUATION, EVENT (default: DEFAULT)",
    )
    parser_list.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output as JSON instead of table",
    )
    setup_laminar_args(parser_list)

    # traces get
    parser_get = traces_subparsers.add_parser(
        "get",
        description="Get detailed information about a trace",
        help="Get detailed information about a trace",
    )
    parser_get.add_argument(
        "trace_id",
        help="UUID of the trace to inspect",
    )
    parser_get.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output as JSON instead of formatted text",
    )
    setup_laminar_args(parser_get)


async def handle_traces_list(args: Namespace) -> None:
    """List traces with optional filtering."""
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

    start_time, end_time = resolve_time_filters(args)

    # Build query dynamically
    conditions = ["trace_type = {trace_type:String}"]
    parameters: dict[str, Any] = {
        "trace_type": args.trace_type,
        "limit": args.limit,
        "offset": args.offset,
    }

    if args.status:
        conditions.append("status = {status:String}")
        parameters["status"] = args.status

    if start_time:
        conditions.append("start_time >= {start_time:String}")
        parameters["start_time"] = start_time

    if end_time:
        conditions.append("start_time <= {end_time:String}")
        parameters["end_time"] = end_time

    where_clause = " AND ".join(conditions)
    sql = f"""SELECT
  id,
  formatDateTime(start_time, '%Y-%m-%dT%H:%i:%S.%fZ') as start_time,
  formatDateTime(end_time, '%Y-%m-%dT%H:%i:%S.%fZ') as end_time,
  top_span_name,
  status,
  total_tokens,
  total_cost,
  input_tokens,
  output_tokens
FROM traces
WHERE {where_clause}
ORDER BY start_time DESC
LIMIT {{limit:UInt64}}
OFFSET {{offset:UInt64}}"""

    try:
        results = await client.sql.query(sql, parameters)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            print_error(
                "Authentication failed.",
                "Check that LMNR_PROJECT_API_KEY is set correctly.",
            )
        elif e.response.status_code == 400:
            print_error(
                f"Invalid query: {e.response.text}",
                "Check your filter parameters.",
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
        print("No traces found matching your filters.")
        return

    if args.json:
        print_json(results)
    else:
        # Format display values
        for row in results:
            row["total_cost"] = format_cost(row.get("total_cost"))
            row["total_tokens"] = format_tokens(row.get("total_tokens"))

        columns = [
            ("id", "ID", 36),
            ("start_time", "Start Time", 26),
            ("top_span_name", "Name", 20),
            ("status", "Status", 8),
            ("total_tokens", "Tokens", 10),
            ("total_cost", "Cost", 10),
        ]
        print_table(results, columns)
        print(f"\nShowing {len(results)} trace(s).")


async def handle_traces_get(args: Namespace) -> None:
    """Get detailed information about a single trace, including its span tree."""
    trace_id = validate_uuid(args.trace_id, "trace ID")

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

    # Query 1: Trace summary
    trace_sql = """SELECT
  id,
  formatDateTime(start_time, '%Y-%m-%dT%H:%i:%S.%fZ') as start_time,
  formatDateTime(end_time, '%Y-%m-%dT%H:%i:%S.%fZ') as end_time,
  status,
  input_tokens,
  output_tokens,
  total_tokens,
  input_cost,
  output_cost,
  total_cost,
  metadata,
  tags,
  session_id,
  user_id,
  trace_type
FROM traces
WHERE id = {trace_id:UUID}
LIMIT 1"""

    # Query 2: Spans tree
    spans_sql = """SELECT
  span_id,
  parent_span_id,
  name,
  span_type,
  formatDateTime(start_time, '%Y-%m-%dT%H:%i:%S.%fZ') as start_time,
  formatDateTime(end_time, '%Y-%m-%dT%H:%i:%S.%fZ') as end_time,
  status,
  input_tokens,
  output_tokens,
  total_tokens,
  input_cost,
  output_cost,
  total_cost,
  model
FROM spans
WHERE trace_id = {trace_id:UUID}
ORDER BY start_time ASC"""

    try:
        trace_results = await client.sql.query(
            trace_sql, {"trace_id": trace_id}
        )
        spans_results = await client.sql.query(
            spans_sql, {"trace_id": trace_id}
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            print_error(
                "Authentication failed.",
                "Check that LMNR_PROJECT_API_KEY is set correctly.",
            )
        elif e.response.status_code == 400:
            print_error(
                f"Invalid query: {e.response.text}",
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

    if not trace_results:
        print(f"No trace found with ID '{trace_id}'.")
        return

    trace = trace_results[0]

    if args.json:
        print_json({"trace": trace, "spans": spans_results})
    else:
        # Print trace summary
        duration = format_duration_seconds(
            trace.get("start_time"), trace.get("end_time")
        )
        total_tokens = format_tokens(trace.get("total_tokens"))
        input_tokens = format_tokens(trace.get("input_tokens"))
        output_tokens = format_tokens(trace.get("output_tokens"))
        total_cost = format_cost(trace.get("total_cost"))
        input_cost = format_cost(trace.get("input_cost"))
        output_cost = format_cost(trace.get("output_cost"))

        print(f"Trace: {trace.get('id', '-')}")
        print(f"  Status:       {trace.get('status', '-')}")
        print(f"  Start:        {trace.get('start_time', '-')}")
        print(f"  End:          {trace.get('end_time', '-')}")
        print(f"  Duration:     {duration}")
        print(f"  Tokens:       {total_tokens} (in: {input_tokens}, out: {output_tokens})")
        print(f"  Cost:         {total_cost} (in: {input_cost}, out: {output_cost})")

        session_id = trace.get("session_id")
        if session_id:
            print(f"  Session:      {session_id}")

        user_id = trace.get("user_id")
        if user_id:
            print(f"  User:         {user_id}")

        tags = trace.get("tags")
        if tags:
            if isinstance(tags, list):
                print(f"  Tags:         {', '.join(str(t) for t in tags)}")
            else:
                print(f"  Tags:         {tags}")

        metadata = trace.get("metadata")
        if metadata:
            print(f"  Metadata:     {metadata}")

        # Print span tree
        if spans_results:
            print("\nSpans:")
            tree_roots = build_span_tree(spans_results)
            flat_spans = flatten_tree(tree_roots)
            print_span_tree(flat_spans)
        else:
            print("\nNo spans found for this trace.")


def build_span_tree(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a tree of spans from a flat list, adding 'children' and 'depth' fields."""
    by_id: dict[str, dict[str, Any]] = {}
    for s in spans:
        by_id[s["span_id"]] = {**s, "children": [], "depth": 0}

    roots: list[dict[str, Any]] = []
    for span in by_id.values():
        parent_id = span.get("parent_span_id")
        if parent_id and parent_id in by_id:
            by_id[parent_id]["children"].append(span)
        else:
            roots.append(span)

    def set_depth(node: dict[str, Any], depth: int) -> None:
        node["depth"] = depth
        for child in node["children"]:
            set_depth(child, depth + 1)

    for root in roots:
        set_depth(root, 0)

    return roots


def flatten_tree(roots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten tree back to list in display order with depth."""
    result: list[dict[str, Any]] = []

    def walk(node: dict[str, Any]) -> None:
        result.append(node)
        for child in sorted(
            node["children"], key=lambda s: s.get("start_time", "")
        ):
            walk(child)

    for root in roots:
        walk(root)
    return result


def print_span_tree(spans: list[dict[str, Any]]) -> None:
    """Print spans as an indented tree."""
    for span in spans:
        depth = span.get("depth", 0)
        indent = "  " * (depth + 1)
        span_type = span.get("span_type", "DEFAULT")
        name = span.get("name", "-")
        duration = format_duration_seconds(
            span.get("start_time"), span.get("end_time")
        )

        parts = [f"{indent}[{span_type}] {name} ({duration})"]

        model = span.get("model")
        if model:
            parts.append(f"- {model}")

        total_tokens = span.get("total_tokens")
        if total_tokens:
            parts.append(f"- {format_tokens(total_tokens)} tokens")

        print(" ".join(parts))


async def handle_traces_command(args: Namespace) -> None:
    """Dispatch traces subcommands."""
    if not hasattr(args, "traces_command") or args.traces_command is None:
        print_error("No traces command specified. Use 'list' or 'get'.")
        print_error(
            "Usage: lmnr traces {list,get} [options]",
        )
        sys.exit(1)
    elif args.traces_command == "list":
        await handle_traces_list(args)
    elif args.traces_command == "get":
        await handle_traces_get(args)
    else:
        print_error(f"Unknown traces command: '{args.traces_command}'. Use 'list' or 'get'.")
        sys.exit(1)
