import asyncio
import glob
import importlib.util
import json
import os
import re
import sys
from argparse import Namespace
from typing import TypedDict

from lmnr.sdk.evaluations.control_vars import EVALUATION_INSTANCES, PREPARE_ONLY
from lmnr.sdk.evaluations.evaluation import Evaluation
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.types import Numeric

LOG = get_default_logger(__name__)
EVAL_DIR = "evals"
DEFAULT_DATASET_PULL_BATCH_SIZE = 100
DEFAULT_DATASET_PUSH_BATCH_SIZE = 100


class EvalArgs(Namespace):
    """Namespace fields populated by the CLI's `eval` subcommand."""

    file: list[str]
    continue_on_error: bool = False
    output_file: str | None = None
    frontend_port: int | None = None

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.file = []


class _Score(TypedDict):
    file: str
    scores: dict[str, Numeric]
    evaluation_id: str
    url: str


def log_evaluation_instance_not_found() -> None:
    LOG.warning(
        "Evaluation instance not found. " +
        "`evaluate` must be called at the top level of the file, " +
        "not inside a function when running evaluations from the CLI."
    )


def _write_scores(output_file: str, scores: list[_Score]) -> None:
    with open(output_file, "w") as f:
        json.dump(scores, f, indent=2)


async def run_evaluation(args: EvalArgs) -> None:
    sys.path.append(os.getcwd())

    # Set frontend port in environment if specified
    if args.frontend_port:
        os.environ["LMNR_FRONTEND_PORT"] = str(args.frontend_port)

    files: list[str]
    if len(args.file) == 0:
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
        files = []
        for pattern in args.file:
            matches = glob.glob(pattern)
            if matches:
                files.extend(matches)
            else:
                # If no matches found, treat as literal filename
                files.append(pattern)

    prep_token = PREPARE_ONLY.set(True)
    scores: list[_Score] = []
    try:
        for file in files:
            # Reset EVALUATION_INSTANCES before loading each file
            _ = EVALUATION_INSTANCES.set([])

            LOG.info(f"Running evaluation from {file}")
            file = os.path.abspath(file)
            name = "user_module" + file

            spec = importlib.util.spec_from_file_location(name, file)
            if spec is None or spec.loader is None:
                LOG.error(f"Could not load module specification from {file}")
                if args.continue_on_error:
                    continue
                return
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod

            spec.loader.exec_module(mod)
            try:
                evaluations: list[Evaluation] | None = EVALUATION_INSTANCES.get()
                if evaluations is None:
                    raise LookupError()
            # may be raised by `get()` or manually by us above
            except LookupError:
                log_evaluation_instance_not_found()
                if args.continue_on_error:
                    continue
                return

            LOG.info(f"Loaded {len(evaluations)} evaluations from {file}")

            for evaluation in evaluations:
                try:
                    eval_result = await evaluation.run()
                    scores.append(
                        {
                            "file": file,
                            "scores": eval_result["average_scores"],
                            "evaluation_id": str(eval_result["evaluation_id"]),
                            "url": eval_result["url"],
                        }
                    )
                except Exception as e:
                    LOG.error(f"Error running evaluation: {e}")
                    if not args.continue_on_error:
                        raise

        if args.output_file:
            # `open()` is a blocking call; run it off the event loop thread.
            await asyncio.to_thread(_write_scores, args.output_file, scores)
    finally:
        PREPARE_ONLY.reset(prep_token)
