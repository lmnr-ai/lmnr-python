from __future__ import annotations

from importlib.metadata import version

# The dev dependency pins mcp 2.x; CI runs the 1.x module under a
# `uv run --with mcp==1.x` overlay. Neither imports under the other major.
collect_ignore = (
    ["test_mcp_v1.py"] if int(version("mcp").split(".")[0]) >= 2 else ["test_mcp_v2.py"]
)
