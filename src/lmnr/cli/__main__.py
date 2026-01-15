"""
Entry point for python -m lmnr.cli.worker invocation.

This allows the TypeScript CLI to invoke the worker as:
    python3 -m lmnr.cli.worker
"""

from lmnr.cli.worker import main

if __name__ == "__main__":
    main()
