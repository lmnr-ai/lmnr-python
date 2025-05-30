from argparse import ArgumentParser
import asyncio
import importlib.util
import os
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path

from lmnr.sdk.evaluations import Evaluation

from .sdk.eval_control import PREPARE_ONLY, EVALUATION_INSTANCES
from .sdk.log import get_default_logger

LOG = get_default_logger(__name__)


EVAL_DIR = "evals"


def add_cursor_rules():
    """Download laminar.mdc file from a hardcoded public URL and save it to .cursor/rules/laminar.mdc"""
    # Hardcoded URL for the laminar.mdc file
    url = "https://raw.githubusercontent.com/lmnr-ai/lmnr/dev/rules/laminar.mdc"
    
    # Create .cursor/rules directory if it doesn't exist
    rules_dir = Path(".cursor/rules")
    rules_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the target file path
    target_file = rules_dir / "laminar.mdc"
    
    try:
        LOG.info(f"Downloading laminar.mdc from {url}")
        
        # Download the file
        with urllib.request.urlopen(url) as response:
            content = response.read()
        
        # Write the content to the target file (this will overwrite if it exists)
        with open(target_file, 'wb') as f:
            f.write(content)
        
        LOG.info(f"Successfully downloaded laminar.mdc to {target_file}")
        
    except urllib.error.URLError as e:
        LOG.error(f"Failed to download file from {url}: {e}")
        sys.exit(1)
    except Exception as e:
        LOG.error(f"Unexpected error: {e}")
        sys.exit(1)


async def run_evaluation(args):
    sys.path.append(os.getcwd())

    if args.file is None:
        files = [
            os.path.join(EVAL_DIR, f)
            for f in os.listdir(EVAL_DIR)
            if re.match(r".*_eval\.py$", f) or re.match(r"eval_.*\.py$", f)
        ]
        if len(files) == 0:
            LOG.error("No evaluation files found in `evals` directory")
            LOG.info(
                "Eval files must be located in the `evals` directory and must be named *_eval.py or eval_*.py"
            )
            return
        files.sort()
        LOG.info(f"Located {len(files)} evaluation files in {EVAL_DIR}")

    else:
        files = [args.file]

    prep_token = PREPARE_ONLY.set(True)
    try:
        for file in files:
            LOG.info(f"Running evaluation from {file}")
            file = os.path.abspath(file)
            name = "user_module" + file

            spec = importlib.util.spec_from_file_location(name, file)
            if spec is None or spec.loader is None:
                LOG.error(f"Could not load module specification from {file}")
                if args.fail_on_error:
                    return
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod

            spec.loader.exec_module(mod)
            evaluations: list[Evaluation] | None = EVALUATION_INSTANCES.get()
            if evaluations is None:
                LOG.warning("Evaluation instance not found")
                if args.fail_on_error:
                    return
                continue

            LOG.info(f"Loaded {len(evaluations)} evaluations from {file}")

            for evaluation in evaluations:
                try:
                    await evaluation.run()
                except Exception as e:
                    LOG.error(f"Error running evaluation: {e}")
                    if args.fail_on_error:
                        raise
    finally:
        PREPARE_ONLY.reset(prep_token)


def cli():
    parser = ArgumentParser(
        prog="lmnr",
        description="CLI for Laminar. Call `lmnr [subcommand] --help` for more information on each subcommand.",
    )

    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    parser_eval = subparsers.add_parser(
        "eval",
        description="Run an evaluation",
        help="Run an evaluation",
    )
    parser_eval.add_argument(
        "file",
        nargs="?",
        help="A file containing the evaluation to run."
        + "If no file name is provided, all evaluation files in the `evals` directory are run as long"
        + "as they match *_eval.py or eval_*.py",
        default=None,
    )

    parser_eval.add_argument(
        "--fail-on-error",
        action="store_true",
        default=False,
        help="Fail on error",
    )

    parser_download = subparsers.add_parser(
        "add-cursor-rules",
        description="Download laminar.mdc file and add it to .cursor/rules",
        help="Download laminar.mdc file and add it to .cursor/rules",
    )

    parsed = parser.parse_args()
    if parsed.subcommand == "eval":
        asyncio.run(run_evaluation(parsed))
    elif parsed.subcommand == "add-cursor-rules":
        add_cursor_rules()
    else:
        parser.print_help()
