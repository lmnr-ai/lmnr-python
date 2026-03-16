"""``lmnr status`` command.

Provides a project status overview -- a quick dashboard in the terminal.
Runs multiple SQL queries and presents a consolidated view.
"""

from __future__ import annotations

import asyncio
import sys
from argparse import ArgumentParser, Namespace, _SubParsersAction
from typing import Any

from lmnr.cli.formatter import OutputFormatter, OutputMode
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient


# -- Parser setup ------------------------------------------------------------

def setup_status_parser(subparsers: _SubParsersAction) -> None:
    """Register the ``status`` subcommand."""
    status_parser: ArgumentParser = subparsers.add_parser(
        "status",
        description="Project status overview",
        help="Show project status overview",
    )
    status_parser.add_argument(
        "--past-hours", type=int, default=24,
        help="Time window in hours (default: 24)",
    )


# -- Handler -----------------------------------------------------------------

async def _status_overview(args: Namespace, formatter: OutputFormatter) -> None:
    """Show project status overview."""
    hours = args.past_hours or 24

    traces_summary_query = (
        f"SELECT count() as total, "
        f"countIf(status = 'error') as errors, "
        f"sum(total_cost) as total_cost, "
        f"sum(total_tokens) as total_tokens, "
        f"avg(duration) as avg_duration "
        f"FROM traces "
        f"WHERE start_time >= now() - interval {hours} hour "
        f"AND trace_type = 'DEFAULT'"
    )

    spans_summary_query = (
        f"SELECT span_type, count() as cnt, avg(duration) as avg_dur "
        f"FROM spans "
        f"WHERE start_time >= now() - interval {hours} hour "
        f"GROUP BY span_type ORDER BY cnt DESC"
    )

    recent_evals_query = (
        "SELECT evaluation_id, count() as points, "
        "formatDateTime(max(created_at), '%Y-%m-%dT%H:%i:%S.%fZ') as last_run "
        "FROM evaluation_datapoints "
        "GROUP BY evaluation_id "
        "ORDER BY max(created_at) DESC LIMIT 5"
    )

    top_models_query = (
        f"SELECT model, count() as calls, sum(total_cost) as cost "
        f"FROM spans WHERE span_type = 'LLM' "
        f"AND start_time >= now() - interval {hours} hour "
        f"GROUP BY model ORDER BY calls DESC LIMIT 5"
    )

    async with AsyncLaminarClient(
        base_url=args.api_url,
        project_api_key=args.api_key,
    ) as client:
        traces_summary = await client.sql.query(traces_summary_query)
        spans_summary = await client.sql.query(spans_summary_query)
        recent_evals = await client.sql.query(recent_evals_query)
        top_models = await client.sql.query(top_models_query)

    result: dict[str, Any] = {
        "time_window": f"past {hours}h",
        "traces": traces_summary[0] if traces_summary else {},
        "spans_by_type": spans_summary,
        "recent_evaluations": recent_evals,
        "top_models": top_models,
    }

    if formatter.mode == OutputMode.TABLE:
        _print_status_table(result, formatter)
    else:
        formatter.output(result)


def _print_status_table(result: dict, formatter: OutputFormatter) -> None:
    """Print a rich status display for terminal output."""
    traces = result.get("traces", {})
    spans = result.get("spans_by_type", [])
    models = result.get("top_models", [])
    evals = result.get("recent_evaluations", [])
    window = result.get("time_window", "past 24h")

    total = traces.get("total", 0)
    errors = traces.get("errors", 0)
    error_pct = (errors / total * 100) if total else 0
    avg_dur = traces.get("avg_duration", 0)
    cost = traces.get("total_cost", 0)
    tokens = traces.get("total_tokens", 0)

    lines = [
        formatter._colorize(f"Laminar Project Status ({window})", "bold"),
        "=" * 40,
        "",
        formatter._colorize("TRACES", "bold"),
        f"  Total: {total:,} | Errors: {errors:,} ({error_pct:.1f}%) | Avg Duration: {avg_dur:.1f}s",
        f"  Total Cost: ${cost:.4f} | Total Tokens: {tokens:,}",
    ]

    if spans:
        lines.append("")
        lines.append(formatter._colorize("SPAN TYPES", "bold"))
        lines.append(
            formatter._format_list_table(
                spans, columns=["span_type", "cnt", "avg_dur"]
            )
        )

    if models:
        lines.append("")
        lines.append(formatter._colorize("TOP MODELS (by calls)", "bold"))
        lines.append(
            formatter._format_list_table(
                models, columns=["model", "calls", "cost"]
            )
        )

    if evals:
        lines.append("")
        lines.append(formatter._colorize("RECENT EVALUATIONS", "bold"))
        lines.append(
            formatter._format_list_table(
                evals, columns=["evaluation_id", "points", "last_run"]
            )
        )

    print("\n".join(lines))


def handle_status_command(args: Namespace, formatter: OutputFormatter) -> None:
    """Dispatch status command."""
    asyncio.run(_status_overview(args, formatter))
