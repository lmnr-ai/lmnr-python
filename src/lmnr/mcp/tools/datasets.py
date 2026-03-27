"""Dataset-related MCP tools."""

import json
import uuid as uuid_module
from typing import Any

from mcp.server.fastmcp import FastMCP

from lmnr.mcp.client import LaminarMcpClient


def register_dataset_tools(server: FastMCP, mcp_client: LaminarMcpClient) -> None:
    """Register dataset-related tools on the MCP server."""

    @server.tool(
        name="list_datasets",
        description=(
            "List all datasets in your Laminar project. "
            "Returns dataset names, IDs, and creation dates."
        ),
    )
    async def list_datasets() -> str:
        """List all datasets in the project."""
        try:
            client = await mcp_client.get_client()
            datasets = await client.datasets.list_datasets()
            result = [
                {
                    "id": str(ds.id),
                    "name": ds.name,
                    "created_at": ds.created_at.isoformat(),
                }
                for ds in datasets
            ]
            return json.dumps(
                {"datasets": result, "count": len(result)},
                indent=2,
            )
        except ValueError as e:
            return json.dumps({"error": str(e)})
        except Exception as e:
            return json.dumps({"error": f"Failed to list datasets: {e}"})

    @server.tool(
        name="get_dataset",
        description=(
            "Get a dataset and its datapoints. Returns the dataset metadata "
            "and data rows. Use list_datasets first to find dataset names or IDs."
        ),
    )
    async def get_dataset(
        name: str | None = None,
        id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        """Get a dataset with its datapoints."""
        try:
            if not name and not id:
                return json.dumps(
                    {"error": "Either 'name' or 'id' must be provided"}
                )
            if name and id:
                return json.dumps(
                    {"error": "Only one of 'name' or 'id' should be provided, not both"}
                )

            client = await mcp_client.get_client()

            dataset_id: uuid_module.UUID | None = None
            if id:
                try:
                    dataset_id = uuid_module.UUID(id)
                except ValueError:
                    return json.dumps({"error": f"Invalid UUID format: {id}"})

            limit = min(max(limit, 1), 200)
            offset = max(offset, 0)

            response = await client.datasets.pull(
                name=name,
                id=dataset_id,
                limit=limit,
                offset=offset,
            )

            datapoints = [
                {
                    "id": str(dp.id) if dp.id else None,
                    "data": dp.data,
                    "target": dp.target,
                    "metadata": dp.metadata,
                    "created_at": (
                        dp.created_at.isoformat() if dp.created_at else None
                    ),
                }
                for dp in response.items
            ]

            result: dict[str, Any] = {
                "dataset": {
                    "name": name,
                    "id": id,
                },
                "datapoints": datapoints,
                "total_count": response.total_count,
            }

            return json.dumps(result, indent=2, default=str)
        except ValueError as e:
            return json.dumps({"error": str(e)})
        except Exception as e:
            return json.dumps({"error": f"Failed to get dataset: {e}"})
