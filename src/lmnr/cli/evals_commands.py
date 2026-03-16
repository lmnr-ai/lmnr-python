"""Evaluations listing/inspection commands for the Laminar CLI.

Named evals_commands.py to avoid collision with the existing evals.py
which handles `lmnr eval <file>` (running evaluations).
"""

import sys
from argparse import Namespace, _SubParsersAction
from typing import Any

import httpx

from lmnr.cli.output import (
    print_error,
    print_json,
    print_table,
    truncate_id,
    validate_uuid,
)
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient


def setup_evals_commands_parser(subparsers: _SubParsersAction) -> None:
    """Setup the evals subcommand parser with list and get subcommands."""
    from lmnr.cli import setup_laminar_args

    parser_evals = subparsers.add_parser(
        "evals",
        description="List and inspect evaluations",
        help="List and inspect evaluations",
    )

    evals_subparsers = parser_evals.add_subparsers(
        title="command",
        dest="evals_command",
    )

    # evals list
    parser_list = evals_subparsers.add_parser(
        "list",
        description="List evaluations in the current project",
        help="List evaluations in the current project",
    )
    parser_list.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max evaluations to return (default: 50)",
    )
    parser_list.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset for pagination (default: 0)",
    )
    parser_list.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output as JSON instead of table",
    )
    setup_laminar_args(parser_list)

    # evals get
    parser_get = evals_subparsers.add_parser(
        "get",
        description="Get evaluation details and score statistics",
        help="Get evaluation details and score statistics",
    )
    parser_get.add_argument(
        "eval_id",
        help="UUID of the evaluation to inspect",
    )
    parser_get.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max datapoints to show (default: 20)",
    )
    parser_get.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output as JSON instead of formatted text",
    )
    setup_laminar_args(parser_get)


async def handle_evals_list(args: Namespace) -> None:
    """List evaluations by querying evaluation_datapoints grouped by evaluation_id."""
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

    sql = """SELECT
  evaluation_id,
  count() as datapoints_count,
  min(formatDateTime(created_at, '%Y-%m-%dT%H:%i:%S.%fZ')) as first_datapoint_at
FROM evaluation_datapoints
GROUP BY evaluation_id
ORDER BY first_datapoint_at DESC
LIMIT {limit:UInt64}
OFFSET {offset:UInt64}"""

    parameters: dict[str, Any] = {
        "limit": args.limit,
        "offset": args.offset,
    }

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
        print("No evaluations found.")
        return

    if args.json:
        print_json(results)
    else:
        columns = [
            ("evaluation_id", "Evaluation ID", 36),
            ("datapoints_count", "Datapoints", 12),
            ("first_datapoint_at", "First Datapoint", 26),
        ]
        print_table(results, columns)
        print(f"\nShowing {len(results)} evaluation(s).")


async def handle_evals_get(args: Namespace) -> None:
    """Get evaluation details, including score statistics."""
    eval_id = validate_uuid(args.eval_id, "evaluation ID")

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

    # Query 1: Summary statistics
    summary_sql = """SELECT
  count() as total_datapoints,
  min(formatDateTime(created_at, '%Y-%m-%dT%H:%i:%S.%fZ')) as first_created,
  max(formatDateTime(created_at, '%Y-%m-%dT%H:%i:%S.%fZ')) as last_created
FROM evaluation_datapoints
WHERE evaluation_id = {eval_id:String}"""

    # Query 2: Recent datapoints with scores
    datapoints_sql = """SELECT
  scores
FROM evaluation_datapoints
WHERE evaluation_id = {eval_id:String}
ORDER BY created_at DESC
LIMIT {limit:UInt64}"""

    try:
        summary_results = await client.sql.query(
            summary_sql, {"eval_id": eval_id}
        )
        datapoints_results = await client.sql.query(
            datapoints_sql, {"eval_id": eval_id, "limit": args.limit}
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

    if not summary_results or summary_results[0].get("total_datapoints", 0) == 0:
        print(f"No evaluation found with ID '{eval_id}'.")
        return

    summary = summary_results[0]

    if args.json:
        # Compute score averages for JSON output
        score_averages = _compute_score_averages(datapoints_results)
        print_json({
            "evaluation_id": eval_id,
            "total_datapoints": summary.get("total_datapoints"),
            "first_created": summary.get("first_created"),
            "last_created": summary.get("last_created"),
            "score_averages": score_averages,
            "datapoints_sample": datapoints_results,
        })
    else:
        total = summary.get("total_datapoints", 0)
        print(f"Evaluation: {eval_id}")
        print(f"  Datapoints:   {total}")
        print(f"  First Run:    {summary.get('first_created', '-')}")
        print(f"  Last Run:     {summary.get('last_created', '-')}")

        # Compute and display score averages
        score_averages = _compute_score_averages(datapoints_results)
        if score_averages:
            print("\nScore Averages:")
            for name, avg in sorted(score_averages.items()):
                print(f"  {name:<16}{avg:.4f}")

        # Print sample datapoints
        if datapoints_results:
            sample_count = len(datapoints_results)
            print(f"\nRecent Datapoints (showing {sample_count} of {total}):")
            for i, dp in enumerate(datapoints_results, 1):
                scores = dp.get("scores", {})
                if isinstance(scores, dict):
                    score_strs = [
                        f"{k}={v:.2f}" if isinstance(v, (int, float)) else f"{k}={v}"
                        for k, v in sorted(scores.items())
                    ]
                    scores_display = ", ".join(score_strs) if score_strs else "-"
                else:
                    scores_display = str(scores) if scores else "-"
                print(f"  {i:>4}  {scores_display}")


def _compute_score_averages(
    datapoints: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute average scores across datapoints.

    Args:
        datapoints: List of datapoint dicts, each expected to have a 'scores' field.

    Returns:
        Dictionary mapping score name to average value.
    """
    score_sums: dict[str, float] = {}
    score_counts: dict[str, int] = {}

    for dp in datapoints:
        scores = dp.get("scores")
        if not isinstance(scores, dict):
            continue
        for name, value in scores.items():
            if isinstance(value, (int, float)):
                score_sums[name] = score_sums.get(name, 0.0) + float(value)
                score_counts[name] = score_counts.get(name, 0) + 1

    return {
        name: score_sums[name] / score_counts[name]
        for name in score_sums
        if score_counts.get(name, 0) > 0
    }


async def handle_evals_command(args: Namespace) -> None:
    """Dispatch evals subcommands."""
    if not hasattr(args, "evals_command") or args.evals_command is None:
        print_error("No evals command specified. Use 'list' or 'get'.")
        print_error(
            "Usage: lmnr evals {list,get} [options]",
        )
        sys.exit(1)
    elif args.evals_command == "list":
        await handle_evals_list(args)
    elif args.evals_command == "get":
        await handle_evals_get(args)
    else:
        print_error(f"Unknown evals command: '{args.evals_command}'. Use 'list' or 'get'.")
        sys.exit(1)
