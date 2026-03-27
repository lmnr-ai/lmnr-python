"""Raw SQL query MCP tool."""

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from lmnr.mcp.client import LaminarMcpClient

_SQL_TOOL_DESCRIPTION = (
    "Execute a ClickHouse SQL query against Laminar data. Only SELECT queries "
    "allowed. Queries are scoped to your project. "
    "Available tables: spans, traces, events, signal_events, signal_runs, "
    "evaluation_datapoints, dataset_datapoints. "
    "Key columns - spans: (span_id, name, span_type, start_time, end_time, "
    "duration, input_cost, output_cost, total_cost, input_tokens, output_tokens, "
    "total_tokens, request_model, response_model, model, trace_id, provider, "
    "path, input, output, status, parent_span_id, attributes, tags), "
    "traces: (id, start_time, end_time, input_tokens, output_tokens, total_tokens, "
    "input_cost, output_cost, total_cost, duration, metadata, session_id, user_id, "
    "status, top_span_id, top_span_name, top_span_type, trace_type, tags, "
    "has_browser_session), "
    "evaluation_datapoints: (id, evaluation_id, data, target, metadata, "
    "executor_output, index, trace_id, group_id, scores, created_at, dataset_id), "
    "dataset_datapoints: (id, created_at, dataset_id, data, target, metadata). "
    "Joins: spans.trace_id = traces.id"
)


def register_sql_tools(server: FastMCP, mcp_client: LaminarMcpClient) -> None:
    """Register the raw SQL query tool on the MCP server."""

    @server.tool(
        name="run_sql_query",
        description=_SQL_TOOL_DESCRIPTION,
    )
    async def run_sql_query(
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> str:
        """Execute a raw SQL query against the Laminar backend."""
        try:
            # Basic validation: only allow SELECT queries
            stripped = query.strip().upper()
            if not stripped.startswith("SELECT") and not stripped.startswith("WITH"):
                return json.dumps(
                    {"error": "Only SELECT (and WITH ... SELECT) queries are allowed"}
                )

            client = await mcp_client.get_client()
            rows = await client.sql.query(query, parameters)
            return json.dumps(
                {"data": rows, "row_count": len(rows)},
                indent=2,
                default=str,
            )
        except ValueError as e:
            return json.dumps({"error": str(e)})
        except Exception as e:
            return json.dumps({"error": f"Failed to execute SQL query: {e}"})
