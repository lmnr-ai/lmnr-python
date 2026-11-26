"""State the plugin shares with the `EvalRunner` adapters in the Harbor process."""

from __future__ import annotations

from lmnr.sdk.types import LaminarSpanContext

LAMINAR_AGENT_IMPORT_PATH = "lmnr.integrations.harbor.runner:LaminarAgent"
LAMINAR_VERIFIER_IMPORT_PATH = "lmnr.integrations.harbor.runner:LaminarVerifier"

# Open verifier spans by trial name. The plugin registers one in its
# VERIFICATION_START hook, which Harbor awaits before it runs the verifier, so
# `LaminarVerifier` can trace `verify()` under it.
verifier_spans: dict[str, LaminarSpanContext] = {}
