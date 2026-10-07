"""Laminar plugin for Harbor (https://harborframework.com).

Install `lmnr` next to `harbor` and run `harbor run ... --plugin laminar`.
"""

from lmnr.integrations.harbor.plugin import LaminarPlugin

__all__ = ["LaminarPlugin"]
