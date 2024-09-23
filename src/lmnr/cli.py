from argparse import ArgumentParser
import asyncio
import importlib
import os
import sys

from lmnr.sdk.evaluations import set_global_evaluation


# TODO: Refactor this code
async def run_evaluation(args):
    sys.path.insert(0, os.getcwd())

    with set_global_evaluation(True):
        file = os.path.abspath(args.file)

        spec = importlib.util.spec_from_file_location("run_eval", file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        from lmnr.sdk.evaluations import _evaluation
        evaluation = _evaluation
        await evaluation.run()


def cli():
    parser = ArgumentParser(
        prog="lmnr",
        description="CLI for Laminar",
    )

    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    parser_eval = subparsers.add_parser("eval", description="Run an evaluation")
    parser_eval.add_argument("file", help="A file containing the evaluation to run")
    parser_eval.set_defaults(func=run_evaluation)

    parsed = parser.parse_args()
    asyncio.run(parsed.func(parsed))
