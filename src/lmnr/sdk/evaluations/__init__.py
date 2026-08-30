import asyncio
import uuid
from typing import Any

from tqdm import tqdm
from typing_extensions import TypedDict

from lmnr.opentelemetry_lib.tracing.instruments import Instruments
from lmnr.sdk.datasets import EvaluationDataset
from lmnr.sdk.evaluations.consts import DEFAULT_BATCH_SIZE, MAX_EXPORT_BATCH_SIZE
from lmnr.sdk.evaluations.control_vars import EVALUATION_INSTANCES, PREPARE_ONLY
from lmnr.sdk.evaluations.evaluation import Evaluation
from lmnr.sdk.types import (
    Datapoint,
    EvaluationResultDatapoint,
    EvaluatorFunction,
    ExecutorFunction,
    HumanEvaluator,
    Numeric,
)
from lmnr.sdk.utils import from_env, get_frontend_url


class EvaluationRunResult(TypedDict):
    average_scores: dict[str, Numeric]
    evaluation_id: uuid.UUID
    project_id: uuid.UUID
    url: str
    error_message: str | None


def get_evaluation_url(
    project_id: str,
    evaluation_id: str,
    base_url: str | None = None,
    frontend_port: int | None = None,
):
    """
    Get the frontend URL for an evaluation.

    Args:
        project_id: Project ID
        evaluation_id: Evaluation ID
        base_url: Base API URL
        frontend_port: Optional frontend port for localhost (defaults to 5667)

    Returns:
        Full URL to the evaluation in the frontend
    """

    # Check environment variable if frontend_port not explicitly provided
    if frontend_port is None:
        port_str = from_env("LMNR_FRONTEND_PORT")
        if port_str:
            try:
                frontend_port = int(port_str)
            except ValueError:
                pass

    frontend_url = get_frontend_url(base_url, frontend_port)
    return f"{frontend_url}/project/{project_id}/evaluations/{evaluation_id}"


# Evaluation-metadata key that links an eval run to its debug session. Kept as
# `rollout.session_id` (matching the trace-metadata key the debugger stamps) so
# the eval and its debug session cross-reference under one identifier.
SESSION_METADATA_KEY = "rollout.session_id"


def _with_debugger_session_metadata(
    metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Stamp the debug session id into eval metadata when running under debug.

    When this eval runs under a debug session, auto-stamp the session id into
    the evaluation metadata so the created evaluation links back to it with no
    extra step. The session id is resolved by the debug runtime EXACTLY like
    traces — `LMNR_DEBUG_SESSION_ID` env → `.lmnr/debug-session.json` → freshly
    minted — so a plain `LMNR_DEBUG=1 <run-your-eval>` groups the eval under the
    current debug session (no CLI wrapper needed). `get_runtime()` is None when
    debug mode is off, so the metadata is returned unchanged.

    The backend writes the `evaluation` block from this key at eval creation;
    notes are attached separately as `text` blocks keyed by the same session id
    (`lmnr-cli debug session add-note`). Called after `Laminar.initialize()`, so
    the runtime (and its resolved session id) already exist.
    """
    from lmnr.sdk.debug import get_runtime

    runtime = get_runtime()
    session_id = runtime.session_id if runtime is not None else None
    if session_id is None:
        return metadata
    return {**(metadata or {}), SESSION_METADATA_KEY: session_id}


def get_average_scores(results: list[EvaluationResultDatapoint]) -> dict[str, Numeric]:
    per_score_values = {}
    for result in results:
        for key, value in result.scores.items():
            if key not in per_score_values:
                per_score_values[key] = []
            per_score_values[key].append(value)

    average_scores = {}
    for key, values in per_score_values.items():
        scores = [v for v in values if v is not None]

        # If there are no scores, we don't want to include the key in the average scores
        if len(scores) > 0:
            average_scores[key] = sum(scores) / len(scores)

    return average_scores


class EvaluationReporter:
    def __init__(self, base_url, frontend_port: int | None = None):
        self.base_url = base_url
        self.frontend_port = frontend_port

    def start(self, length: int):
        self.cli_progress = tqdm(
            total=length,
            bar_format="{bar} {percentage:3.0f}% | ETA: {remaining}s | {n_fmt}/{total_fmt}",
            ncols=60,
        )

    def update(self, batch_length: int):
        self.cli_progress.update(batch_length)

    def stop_with_error(self, error: Exception):
        if hasattr(self, "cli_progress"):
            self.cli_progress.close()
        raise error

    def stop(
        self, average_scores: dict[str, Numeric], project_id: str, evaluation_id: str
    ):
        self.cli_progress.close()
        print("Average scores:")
        for name, score in average_scores.items():
            print(f"{name}: {score}")
        print(
            f"Check the results at {get_evaluation_url(project_id, evaluation_id, self.base_url, self.frontend_port)}\n"
        )


def evaluate(
    data: EvaluationDataset | list[Datapoint | dict],
    executor: ExecutorFunction,
    evaluators: dict[str, EvaluatorFunction | HumanEvaluator],
    name: str | None = None,
    group_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    concurrency_limit: int = DEFAULT_BATCH_SIZE,
    project_api_key: str | None = None,
    base_url: str | None = None,
    base_http_url: str | None = None,
    http_port: int | None = None,
    grpc_port: int | None = None,
    frontend_port: int | None = None,
    instruments: (
        set[Instruments] | list[Instruments] | tuple[Instruments] | None
    ) = None,
    disabled_instruments: (
        set[Instruments] | list[Instruments] | tuple[Instruments] | None
    ) = None,
    max_export_batch_size: int | None = MAX_EXPORT_BATCH_SIZE,
    trace_export_timeout_seconds: int | None = None,
) -> EvaluationRunResult | None:
    """
    If added to the file which is called through `lmnr eval` command, then
    registers the evaluation; otherwise, runs the evaluation.

    If there is no event loop, creates it and runs the evaluation until
    completion.
    If there is an event loop, returns an awaitable handle immediately. IMPORTANT:
    You must await the call to `evaluate`.

    Parameters:
        data (list[EvaluationDatapoint|dict] | EvaluationDataset):\
            List of data points to evaluate or an evaluation dataset.
                `data` is the input to the executor function,
                `target` is the input to the evaluator function.
        executor (Callable[..., Any]): The executor function.\
            Takes the data point + any additional arguments\
            and returns the output to evaluate.
        evaluators (dict[str, Callable[..., Any] | HumanEvaluator]): Evaluator\
            functions and HumanEvaluator instances with names. Each evaluator\
            function takes the output of the executor _and_ the target data,\
            and returns a score. The score can be a single number or a dict\
            of string keys and number values. If the score is a single number,\
            it will be named after the evaluator function.\
            HumanEvaluator instances create empty spans for manual evaluation.\
            Evaluator function names must contain only letters, digits, hyphens,\
            underscores, or spaces.
        name (str | None, optional): Optional name of the evaluation.\
            Used to identify the evaluation in the group. If not provided, a\
            random name will be generated.
            Defaults to None.
        group_name (str | None, optional): An identifier to group evaluations.\
            Only evaluations within the same group_name can be visually compared.\
            If not provided, set to "default".
            Defaults to None
        metadata (dict[str, Any] | None, optional): Optional metadata to associate with\
        concurrency_limit (int, optional): The concurrency limit for evaluation.
                        Defaults to DEFAULT_BATCH_SIZE.
        project_api_key (str | None, optional): The project API key.
                        Defaults to None.
        base_url (str | None, optional): The base URL for Laminar API.\
                        Useful if self-hosted elsewhere. Do NOT include the\
                        port, use `http_port` and `grpc_port` instead.
                        Defaults to "https://api.lmnr.ai".
        base_http_url (str | None, optional): The base HTTP URL for Laminar API.\
                        Only set this if your Laminar backend HTTP is proxied\
                        through a different host. If not specified, defaults\
                        to https://api.lmnr.ai.
        http_port (int | None, optional): The port for Laminar API's HTTP\
                        service. 443 is used if not specified.
                        Defaults to None.
        grpc_port (int | None, optional): The port for Laminar API's gRPC\
                        service. 8443 is used if not specified.
                        Defaults to None.
        frontend_port (int | None, optional): The port for the Laminar frontend.\
                        Defaults to 5667 if not specified.
        instruments (set[Instruments] | None, optional): Set of modules to\
                        auto-instrument. If None, all available instruments\
                        will be used.
                        Defaults to None.
        disabled_instruments (set[Instruments] | None, optional): Set of modules\
                        to disable auto-instrumentations. If None, no\
                        If None, only modules passed as `instruments` will be disabled.
                        Defaults to None.
        trace_export_timeout_seconds (int | None, optional): The timeout for\
                        trace export on OpenTelemetry exporter. Defaults to None.
    """
    evaluation = Evaluation(
        data=data,
        executor=executor,
        evaluators=evaluators,
        group_name=group_name,
        metadata=metadata,
        name=name,
        concurrency_limit=concurrency_limit,
        project_api_key=project_api_key,
        base_url=base_url,
        base_http_url=base_http_url,
        http_port=http_port,
        grpc_port=grpc_port,
        frontend_port=frontend_port,
        instruments=instruments,
        disabled_instruments=disabled_instruments,
        max_export_batch_size=max_export_batch_size,
        trace_export_timeout_seconds=trace_export_timeout_seconds,
    )

    if PREPARE_ONLY.get():
        existing_evaluations = EVALUATION_INSTANCES.get([])
        new_evaluations = (existing_evaluations or []) + [evaluation]
        EVALUATION_INSTANCES.set(new_evaluations)
        return None
    else:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            return asyncio.run(evaluation.run())

        if loop.is_running():
            return evaluation.run()
        return asyncio.run(evaluation.run())
