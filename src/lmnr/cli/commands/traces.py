"""``lmnr traces list`` and ``lmnr traces get <id>`` commands.

Both commands query ClickHouse via the SDK's async SQL resource and
format results through the smart OutputFormatter.
"""

from __future__ import annotations

import asyncio
import sys
from argparse import ArgumentParser, Namespace, _SubParsersAction
from typing import Any

from lmnr.cli.formatter import OutputFormatter, OutputMode
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient


def _esc(value: str) -> str:
    """Escape a string for safe use in a SQL single-quoted literal."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


# -- Parser setup ------------------------------------------------------------

def setup_traces_parser(subparsers: _SubParsersAction) -> None:
    """Register the ``traces`` subcommand group."""
    traces_parser: ArgumentParser = subparsers.add_parser(
        "traces",
        description="Trace operations",
        help="List and inspect traces",
    )
    traces_sub = traces_parser.add_subparsers(
        title="traces commands",
        dest="traces_command",
    )

    # --- traces list ---
    list_parser = traces_sub.add_parser("list", help="List recent traces")
    list_parser.add_argument(
        "-n", "--limit", type=int, default=20,
        help="Number of traces to return (default: 20)",
    )
    list_parser.add_argument(
        "--past-hours", type=int, default=24,
        help="Time window in hours (default: 24)",
    )
    list_parser.add_argument(
        "--status", choices=["error", "success"],
        help="Filter by status",
    )
    list_parser.add_argument(
        "--type", default="DEFAULT",
        help="Filter by trace type (default: DEFAULT)",
    )
    list_parser.add_argument(
        "--tag", type=str,
        help="Filter by tag",
    )

    # --- traces get ---
    get_parser = traces_sub.add_parser("get", help="Get trace details")
    get_parser.add_argument(
        "trace_id", help="Trace ID (UUID)",
    )
    get_parser.add_argument(
        "--spans-only", action="store_true",
        help="Show only the span tree, no LLM details",
    )
    get_parser.add_argument(
        "--full", action="store_true",
        help="Don't truncate inputs/outputs",
    )


# -- Handlers ----------------------------------------------------------------

TABLE_COLUMNS = [
    "id", "start_time", "top_span_name", "status",
    "total_cost", "total_tokens", "duration",
]


async def _traces_list(args: Namespace, formatter: OutputFormatter) -> None:
    """List traces with optional filters."""
    limit = args.limit or 20
    trace_type = getattr(args, "type", "DEFAULT") or "DEFAULT"
    past_hours = args.past_hours or 24

    query = (
        "SELECT id, "
        "formatDateTime(start_time, '%Y-%m-%dT%H:%i:%S.%fZ') as start_time, "
        "formatDateTime(end_time, '%Y-%m-%dT%H:%i:%S.%fZ') as end_time, "
        "top_span_name, status, total_cost, total_tokens, "
        "input_tokens, output_tokens, session_id, user_id, tags, "
        "trace_type, duration "
        "FROM traces "
        f"WHERE trace_type = '{_esc(trace_type)}' "
    )

    if args.status:
        query += f"AND status = '{_esc(args.status)}' "
    if past_hours:
        query += f"AND start_time >= now() - interval {past_hours} hour "
    if args.tag:
        query += f"AND has(tags, '{_esc(args.tag)}') "

    query += f"ORDER BY start_time DESC LIMIT {limit}"

    async with AsyncLaminarClient(
        base_url=args.api_url,
        project_api_key=args.api_key,
    ) as client:
        data = await client.sql.query(query)

    formatter.output(data, columns=TABLE_COLUMNS)


async def _traces_get(args: Namespace, formatter: OutputFormatter) -> None:
    """Get detailed trace with span tree."""
    trace_id = _esc(args.trace_id)
    full = getattr(args, "full", False)

    # Fetch trace summary
    trace_query = (
        "SELECT id, "
        "formatDateTime(start_time, '%Y-%m-%dT%H:%i:%S.%fZ') as start_time, "
        "formatDateTime(end_time, '%Y-%m-%dT%H:%i:%S.%fZ') as end_time, "
        "top_span_name, status, total_cost, total_tokens, "
        "input_tokens, output_tokens, session_id, duration "
        "FROM traces "
        f"WHERE id = '{trace_id}'"
    )

    # Fetch spans
    input_output = "input, output, " if not getattr(args, "spans_only", False) else ""
    spans_query = (
        "SELECT span_id, name, span_type, "
        "formatDateTime(start_time, '%Y-%m-%dT%H:%i:%S.%fZ') as start_time, "
        "duration, "
        f"{input_output}"
        "status, parent_span_id, model "
        "FROM spans "
        f"WHERE trace_id = '{trace_id}' "
        "ORDER BY start_time ASC"
    )

    async with AsyncLaminarClient(
        base_url=args.api_url,
        project_api_key=args.api_key,
    ) as client:
        trace_rows = await client.sql.query(trace_query)
        spans = await client.sql.query(spans_query)

    if not trace_rows:
        print(f"Error: Trace {trace_id} not found.", file=sys.stderr)
        sys.exit(1)

    trace = trace_rows[0]

    if formatter.mode == OutputMode.TABLE:
        _print_trace_table(trace, spans, formatter, full)
    else:
        result: dict[str, Any] = {
            "trace_id": trace.get("id", trace_id),
            "duration": trace.get("duration"),
            "total_cost": trace.get("total_cost"),
            "total_tokens": trace.get("total_tokens"),
            "input_tokens": trace.get("input_tokens"),
            "output_tokens": trace.get("output_tokens"),
            "status": trace.get("status"),
            "session_id": trace.get("session_id"),
            "start_time": trace.get("start_time"),
            "end_time": trace.get("end_time"),
            "spans": spans,
        }
        formatter.output(result)


def _print_trace_table(
    trace: dict,
    spans: list[dict],
    formatter: OutputFormatter,
    full: bool,
) -> None:
    """Print a rich trace detail view for terminal output."""
    tid = trace.get("id", "")
    dur = trace.get("duration", "-")
    cost = trace.get("total_cost", "-")
    tok = trace.get("total_tokens", "-")
    itok = trace.get("input_tokens", "-")
    otok = trace.get("output_tokens", "-")
    status = trace.get("status", "-")
    sess = trace.get("session_id", "-")

    lines = [
        formatter._colorize(f"Trace: {tid}", "bold"),
        (
            f"Duration: {dur}s | Cost: ${cost} | "
            f"Tokens: {tok} ({itok} in / {otok} out)"
        ),
        f"Status: {status} | Session: {sess}",
        "",
        formatter._colorize("SPAN TREE:", "bold"),
        "legend: span_name (id, parent_id, type)",
    ]

    for span in spans:
        sid = str(span.get("span_id", ""))[:8]
        parent = str(span.get("parent_span_id", ""))[:8] or "None"
        stype = span.get("span_type", "default")
        sname = span.get("name", "")
        lines.append(f"  - {sname} ({sid}, {parent}, {stype})")

    # LLM & Tool span details (unless --spans-only)
    llm_tool_spans = [
        s for s in spans
        if s.get("span_type") in ("LLM", "TOOL") and "input" in s
    ]
    if llm_tool_spans:
        lines.append("")
        lines.append(formatter._colorize("LLM & TOOL SPANS:", "bold"))
        for span in llm_tool_spans:
            sname = span.get("name", "")
            sid = str(span.get("span_id", ""))[:8]
            model = span.get("model", "")
            sdur = span.get("duration", "-")

            header = f"  {sname} ({sid})"
            if model:
                header += f" | Model: {model}"
            header += f" | Duration: {sdur}s"
            lines.append(formatter._colorize(header, "cyan"))

            inp = span.get("input", "")
            out = span.get("output", "")
            if inp:
                inp_str = str(inp)
                if not full and len(inp_str) > 200:
                    inp_str = inp_str[:197] + "..."
                lines.append(f"    Input:  {inp_str}")
            if out:
                out_str = str(out)
                if not full and len(out_str) > 200:
                    out_str = out_str[:197] + "..."
                lines.append(f"    Output: {out_str}")
            lines.append("")

    print("\n".join(lines))


def handle_traces_command(args: Namespace, formatter: OutputFormatter) -> None:
    """Dispatch traces subcommands."""
    cmd = getattr(args, "traces_command", None)
    if cmd == "list":
        asyncio.run(_traces_list(args, formatter))
    elif cmd == "get":
        asyncio.run(_traces_get(args, formatter))
    else:
        print("Usage: lmnr traces {list,get} ...", file=sys.stderr)
        print("Run 'lmnr traces --help' for details.", file=sys.stderr)
        sys.exit(1)
