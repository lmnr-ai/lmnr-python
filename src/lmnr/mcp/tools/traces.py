"""Trace-related MCP tools."""

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from lmnr.mcp.client import LaminarMcpClient


def _build_span_tree(spans: list[dict[str, Any]]) -> str:
    """Build an indented text representation of the span hierarchy.

    Args:
        spans: List of span dicts, each with span_id, parent_span_id, name,
               span_type, duration, and status fields.

    Returns:
        A string with an indented tree view of the span hierarchy.
    """
    if not spans:
        return "(no spans)"

    # Build parent -> children mapping
    children_map: dict[str | None, list[dict[str, Any]]] = {}
    for span in spans:
        parent = span.get("parent_span_id")
        children_map.setdefault(parent, []).append(span)

    lines: list[str] = []

    def _render(span: dict[str, Any], depth: int) -> None:
        indent = "  " * depth
        name = span.get("name", "unknown")
        span_type = span.get("span_type", "")
        duration = span.get("duration", "?")
        status = span.get("status", "?")
        model = span.get("model", "")
        model_str = f" model={model}" if model else ""
        lines.append(
            f"{indent}- {name} [{span_type}] "
            f"duration={duration}s status={status}{model_str}"
        )
        span_id = span.get("span_id")
        for child in children_map.get(span_id, []):
            _render(child, depth + 1)

    # Render from root spans (those with no parent)
    roots = children_map.get(None, []) + children_map.get("", [])
    # Deduplicate in case both None and "" map to the same spans
    seen = set()
    for root in roots:
        sid = root.get("span_id")
        if sid not in seen:
            seen.add(sid)
            _render(root, 0)

    # If no roots found (orphaned spans), render all flat
    if not lines:
        for span in spans:
            _render(span, 0)

    return "\n".join(lines)


def register_trace_tools(server: FastMCP, mcp_client: LaminarMcpClient) -> None:
    """Register trace-related tools on the MCP server."""

    @server.tool(
        name="list_traces",
        description=(
            "List recent traces from your Laminar project. "
            "Returns trace IDs, timestamps, status, cost, token usage, "
            "and top-level span info. "
            "Use this to find traces to investigate further with get_trace."
        ),
    )
    async def list_traces(
        limit: int = 20,
        status: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        name: str | None = None,
        trace_type: str | None = None,
    ) -> str:
        """List recent traces with optional filters."""
        try:
            client = await mcp_client.get_client()

            conditions: list[str] = []
            params: dict[str, Any] = {}

            if status:
                conditions.append("status = {status:String}")
                params["status"] = status
            if start_time:
                conditions.append("start_time >= {start_time:String}")
                params["start_time"] = start_time
            if end_time:
                conditions.append("end_time <= {end_time:String}")
                params["end_time"] = end_time
            if name:
                conditions.append("top_span_name LIKE {name_pattern:String}")
                params["name_pattern"] = f"%{name}%"
            if trace_type:
                conditions.append("trace_type = {trace_type:String}")
                params["trace_type"] = trace_type

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            limit = min(max(limit, 1), 100)
            params["limit"] = limit

            sql = f"""
                SELECT id, start_time, end_time, duration, status,
                       top_span_name, top_span_type, total_tokens,
                       total_cost, tags, trace_type
                FROM traces
                WHERE {where_clause}
                ORDER BY start_time DESC
                LIMIT {{limit:Int64}}
            """

            rows = await client.sql.query(sql, params)
            return json.dumps(
                {"traces": rows, "count": len(rows)},
                indent=2,
                default=str,
            )
        except ValueError as e:
            return json.dumps({"error": str(e)})
        except Exception as e:
            return json.dumps({"error": f"Failed to list traces: {e}"})

    @server.tool(
        name="get_trace",
        description=(
            "Get detailed information about a specific trace, including its "
            "complete span tree with timing, inputs/outputs, and error details. "
            "Use list_traces first to find trace IDs."
        ),
    )
    async def get_trace(
        trace_id: str,
        include_io: bool = False,
    ) -> str:
        """Get detailed trace info including the span tree."""
        try:
            client = await mcp_client.get_client()

            # Query 1: trace metadata
            trace_rows = await client.sql.query(
                "SELECT id, start_time, end_time, duration, status, "
                "top_span_name, top_span_type, total_tokens, total_cost, "
                "input_tokens, output_tokens, input_cost, output_cost, "
                "tags, trace_type, metadata, session_id, user_id "
                "FROM traces WHERE id = {trace_id:UUID} LIMIT 1",
                {"trace_id": trace_id},
            )

            if not trace_rows:
                return json.dumps({"error": f"Trace {trace_id} not found"})

            # Query 2: all spans for the trace
            span_cols = (
                "span_id, name, span_type, start_time, end_time, duration, "
                "status, parent_span_id, model, input_tokens, output_tokens, "
                "total_cost, path"
            )
            if include_io:
                span_cols += ", input, output"

            span_rows = await client.sql.query(
                f"SELECT {span_cols} FROM spans "
                "WHERE trace_id = {trace_id:UUID} "
                "ORDER BY start_time ASC",
                {"trace_id": trace_id},
            )

            # Build span tree text representation
            tree = _build_span_tree(span_rows)

            return json.dumps(
                {
                    "trace": trace_rows[0] if trace_rows else None,
                    "spans": span_rows,
                    "span_tree": tree,
                },
                indent=2,
                default=str,
            )
        except ValueError as e:
            return json.dumps({"error": str(e)})
        except Exception as e:
            return json.dumps({"error": f"Failed to get trace: {e}"})
