"""Evaluation-related MCP tools."""

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from lmnr.mcp.client import LaminarMcpClient


def register_eval_tools(server: FastMCP, mcp_client: LaminarMcpClient) -> None:
    """Register evaluation-related tools on the MCP server."""

    @server.tool(
        name="list_evaluations",
        description=(
            "List evaluations that have been run in your Laminar project. "
            "Shows evaluation IDs, groups, and datapoint counts. "
            "Use get_evaluation for detailed scores."
        ),
    )
    async def list_evaluations(
        limit: int = 20,
        group_id: str | None = None,
    ) -> str:
        """List evaluations with optional group filter."""
        try:
            client = await mcp_client.get_client()

            conditions: list[str] = []
            params: dict[str, Any] = {}

            if group_id:
                conditions.append("group_id = {group_id:String}")
                params["group_id"] = group_id

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            limit = min(max(limit, 1), 100)
            params["limit"] = limit

            sql = f"""
                SELECT
                    evaluation_id,
                    group_id,
                    min(created_at) as created_at,
                    count() as datapoint_count
                FROM evaluation_datapoints
                WHERE {where_clause}
                GROUP BY evaluation_id, group_id
                ORDER BY created_at DESC
                LIMIT {{limit:Int64}}
            """

            rows = await client.sql.query(sql, params)
            return json.dumps(
                {"evaluations": rows, "count": len(rows)},
                indent=2,
                default=str,
            )
        except ValueError as e:
            return json.dumps({"error": str(e)})
        except Exception as e:
            return json.dumps({"error": f"Failed to list evaluations: {e}"})

    @server.tool(
        name="get_evaluation",
        description=(
            "Get detailed results for a specific evaluation, including "
            "individual datapoint scores, executor outputs, and aggregate "
            "statistics. Use list_evaluations first to find evaluation IDs."
        ),
    )
    async def get_evaluation(
        evaluation_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        """Get detailed evaluation results with datapoint scores."""
        try:
            client = await mcp_client.get_client()

            params: dict[str, Any] = {
                "evaluation_id": evaluation_id,
                "limit": min(max(limit, 1), 200),
                "offset": max(offset, 0),
            }

            # Get datapoints
            sql = """
                SELECT id, evaluation_id, group_id, data, target,
                       executor_output, scores, trace_id, created_at, index
                FROM evaluation_datapoints
                WHERE evaluation_id = {evaluation_id:UUID}
                ORDER BY index ASC
                LIMIT {limit:Int64}
                OFFSET {offset:Int64}
            """

            rows = await client.sql.query(sql, params)

            if not rows:
                return json.dumps(
                    {"error": f"No datapoints found for evaluation {evaluation_id}"}
                )

            # Get total count
            count_sql = """
                SELECT count() as total
                FROM evaluation_datapoints
                WHERE evaluation_id = {evaluation_id:UUID}
            """
            count_rows = await client.sql.query(
                count_sql, {"evaluation_id": evaluation_id}
            )
            total_count = count_rows[0]["total"] if count_rows else len(rows)

            # Compute aggregate scores across ALL datapoints (not just
            # the paginated subset) by fetching only the scores column.
            scores_sql = """
                SELECT scores
                FROM evaluation_datapoints
                WHERE evaluation_id = {evaluation_id:UUID}
            """
            all_score_rows = await client.sql.query(
                scores_sql, {"evaluation_id": evaluation_id}
            )
            all_scores: dict[str, list[float]] = {}
            for score_row in all_score_rows:
                scores = score_row.get("scores")
                if isinstance(scores, dict):
                    for score_name, score_val in scores.items():
                        if isinstance(score_val, (int, float)):
                            all_scores.setdefault(score_name, []).append(
                                score_val
                            )

            score_summary: dict[str, dict[str, float]] = {}
            for score_name, values in all_scores.items():
                if values:
                    score_summary[score_name] = {
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                    }

            result = {
                "evaluation_id": evaluation_id,
                "group_id": rows[0].get("group_id") if rows else None,
                "datapoints": rows,
                "total_count": total_count,
                "aggregate_scores": score_summary,
            }

            return json.dumps(result, indent=2, default=str)
        except ValueError as e:
            return json.dumps({"error": str(e)})
        except Exception as e:
            return json.dumps({"error": f"Failed to get evaluation: {e}"})
