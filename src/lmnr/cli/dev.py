"""
The functionality of the dev is implemented in the @lmnr-ai/cli package.

This simply shows a deprecation message and exits.
"""

import sys
from argparse import Namespace


async def run_dev(args: Namespace) -> None:
    """Show deprecation message for dev command."""
    print(
        "\033[33m\n"  # Yellow text
        "The 'lmnr dev' command has been moved to a separate package.\n\n"
        "Please use:\n"
        "  npx @lmnr-ai/cli@latest dev [file] [options]\n\n"
        "See https://docs.lmnr.ai/cli/dev for more information.\n"
        "\033[0m",  # Reset color
        file=sys.stderr,
    )
    sys.exit(1)
