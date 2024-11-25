from argparse import ArgumentParser
import asyncio
import importlib.util
import os
import sys

from .sdk.eval_control import PREPARE_ONLY, EVALUATION_INSTANCE


async def run_evaluation(args):
    sys.path.append(os.getcwd())

    prep_token = PREPARE_ONLY.set(True)
    try:
        file = os.path.abspath(args.file)
        name = "user_module"

        spec = importlib.util.spec_from_file_location(name, file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module specification from {file}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod

        spec.loader.exec_module(mod)
        evaluation = EVALUATION_INSTANCE.get()
        if evaluation is None:
            raise RuntimeError("Evaluation instance not found")

        await evaluation.run()
    finally:
        PREPARE_ONLY.reset(prep_token)


def cli():
    parser = ArgumentParser(
        prog="lmnr",
        description="CLI for Laminar",
    )

    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    parser_eval = subparsers.add_parser(
        "eval",
        description="Run an evaluation",
        help="Run an evaluation",
    )
    parser_eval.add_argument("file", help="A file containing the evaluation to run")

    parsed = parser.parse_args()
    if parsed.subcommand == "eval":
        asyncio.run(run_evaluation(parsed))
    else:
        parser.print_help()
