"""Constants of lmnr-atif, Laminar's extension to ATIF.

The spec itself, its JSON schema and its fixtures live in `spec/lmnr-atif/` at
the repository root and are copied verbatim into every SDK that implements it.
`tests/test_lmnr_atif.py` checks that `SPEC_VERSION` matches all of them.
"""

from __future__ import annotations

from typing import Literal

# Version of `spec/lmnr-atif/lmnr-atif.md` this package implements.
SPEC_VERSION = "0.1.0"

# Every Laminar field lives under `extra[NAMESPACE]`.
NAMESPACE = "lmnr"

# ATIF version writers emit, and the oldest one the extension can live in
# (`tool_call.extra`, `observation_result.extra` and `llm_call_count` are v1.7).
ATIF_SCHEMA_VERSION = "ATIF-v1.8"
MIN_ATIF_VERSION = (1, 7)

DefKind = Literal["system_prompts", "tool_definitions", "input_messages"]
