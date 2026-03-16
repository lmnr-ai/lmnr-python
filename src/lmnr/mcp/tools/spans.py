"""Span search MCP tools."""

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from lmnr.mcp.client import LaminarMcpClient


def register_span_tools(server: FastMCP, mcp_client: LaminarMcpClient) -> None:
    """Register span-related tools on the MCP server."""

    @server.tool(
        name="search_spans",
        description=(
            "Search spans across all traces by name, type, status, model, "
            "or other attributes. Returns matching spans with their trace context."
        ),
    )
    async def search_spans(
        name: str | None = None,
        span_type: str | None = None,
        status: str | None = None,
        model: str | None = None,
        min_duration: float | None = None,
        min_cost: float | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 20,
    ) -> str:
        """Search spans across all traces with optional filters."""
        try:
            client = await mcp_client.get_client()

            conditions: list[str] = []
            params: dict[str, Any] = {}

            if name:
                conditions.append("name LIKE {name_pattern:String}")
                params["name_pattern"] = f"%{name}%"
            if span_type:
                conditions.append("span_type = {span_type:String}")
                params["span_type"] = span_type
            if status:
                conditions.append("status = {status:String}")
                params["status"] = status
            if model:
                conditions.append("model LIKE {model_pattern:String}")
                params["model_pattern"] = f"%{model}%"
            if min_duration is not None:
                conditions.append("duration >= {min_duration:Float64}")
                params["min_duration"] = min_duration
            if min_cost is not None:
                conditions.append("total_cost >= {min_cost:Float64}")
                params["min_cost"] = min_cost
            if start_time:
                conditions.append("start_time >= {start_time:String}")
                params["start_time"] = start_time
            if end_time:
                conditions.append("end_time <= {end_time:String}")
                params["end_time"] = end_time

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            limit = min(max(limit, 1), 100)
            params["limit"] = limit

            sql = f"""
                SELECT span_id, trace_id, name, span_type, start_time,
                       duration, status, model, input_tokens, output_tokens,
                       total_cost, path, tags
                FROM spans
                WHERE {where_clause}
                ORDER BY start_time DESC
                LIMIT {{limit:Int64}}
            """

            rows = await client.sql.query(sql, params)
            return json.dumps(
                {"spans": rows, "count": len(rows)},
                indent=2,
                default=str,
            )
        except ValueError as e:
            return json.dumps({"error": str(e)})
        except Exception as e:
            return json.dumps({"error": f"Failed to search spans: {e}"})
