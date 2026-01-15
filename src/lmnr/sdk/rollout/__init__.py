"""
Rollout module for interactive LLM debugging.

This module provides client-side infrastructure for rollout mode,
working with the external @lmnr-ai/cli package.
"""

from .cache_client import CacheClient
from .instrumentation import RolloutInstrumentationWrapper

__all__ = [
    "CacheClient",
    "RolloutInstrumentationWrapper",
]
