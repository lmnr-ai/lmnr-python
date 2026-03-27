"""Laminar MCP (Model Context Protocol) server module.

This module provides a local MCP server that exposes Laminar observability
tools for AI agents (Claude Code, Cursor, VS Code Copilot, etc.) over
stdio transport.

Usage:
    CLI:
        lmnr mcp serve

    Programmatic:
        import asyncio
        from lmnr.mcp import serve
        asyncio.run(serve())

    IDE configuration (Claude Code .mcp.json):
        {
            "mcpServers": {
                "laminar": {
                    "command": "lmnr",
                    "args": ["mcp", "serve"],
                    "env": {
                        "LMNR_PROJECT_API_KEY": "your-key-here"
                    }
                }
            }
        }
"""

from lmnr.mcp.server import create_server, serve

__all__ = ["create_server", "serve"]
