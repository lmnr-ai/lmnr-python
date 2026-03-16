"""Laminar MCP server -- exposes Laminar observability tools for AI agents.

This module creates a local MCP server using stdio transport that provides
composable, agent-friendly tools for querying traces, spans, evaluations,
datasets, and running raw SQL against the Laminar backend.
"""

import logging

from mcp.server.fastmcp import FastMCP

from lmnr.mcp.client import LaminarMcpClient
from lmnr.mcp.tools.traces import register_trace_tools
from lmnr.mcp.tools.spans import register_span_tools
from lmnr.mcp.tools.evals import register_eval_tools
from lmnr.mcp.tools.datasets import register_dataset_tools
from lmnr.mcp.tools.sql import register_sql_tools

logger = logging.getLogger(__name__)


def create_server(
    project_api_key: str | None = None,
    base_url: str | None = None,
    port: int | None = None,
) -> tuple[FastMCP, LaminarMcpClient]:
    """Create and configure the Laminar MCP server with all tools.

    Args:
        project_api_key: Laminar project API key. Falls back to
            LMNR_PROJECT_API_KEY environment variable.
        base_url: Laminar API base URL. Falls back to LMNR_BASE_URL
            environment variable or https://api.lmnr.ai.
        port: Port for the Laminar API. Defaults to 443.

    Returns:
        A tuple of (FastMCP server, LaminarMcpClient).
    """
    server = FastMCP(
        name="laminar",
        instructions=(
            "Laminar observability server. Use these tools to query traces, "
            "spans, evaluations, and datasets from your Laminar project. "
            "Start with list_traces or search_spans to find data, then use "
            "get_trace for details. Use run_sql_query for custom queries."
        ),
    )

    mcp_client = LaminarMcpClient(
        project_api_key=project_api_key,
        base_url=base_url,
        port=port,
    )

    # Register all tool domains
    register_trace_tools(server, mcp_client)
    register_span_tools(server, mcp_client)
    register_eval_tools(server, mcp_client)
    register_dataset_tools(server, mcp_client)
    register_sql_tools(server, mcp_client)

    return server, mcp_client


async def serve(
    project_api_key: str | None = None,
    base_url: str | None = None,
    port: int | None = None,
) -> None:
    """Start the Laminar MCP server over stdio.

    This is the main entry point for running the MCP server. It creates
    the server, registers all tools, and starts the stdio transport loop.

    Args:
        project_api_key: Laminar project API key.
        base_url: Laminar API base URL.
        port: Port for the Laminar API.
    """
    server, mcp_client = create_server(
        project_api_key=project_api_key,
        base_url=base_url,
        port=port,
    )
    try:
        await server.run_stdio_async()
    finally:
        await mcp_client.close()
