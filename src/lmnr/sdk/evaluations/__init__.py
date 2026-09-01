import asyncio
import uuid
from typing import Any

from typing_extensions import TypedDict

from lmnr.opentelemetry_lib.tracing.instruments import Instruments
from lmnr.sdk.datasets import EvaluationDataset
from lmnr.sdk.evaluations.control_vars import EVALUATION_INSTANCES, PREPARE_ONLY
from lmnr.sdk.evaluations.evaluation import Evaluation
from lmnr.sdk.evaluations.models import (
    DEFAULT_BATCH_SIZE,
    MAX_EXPORT_BATCH_SIZE,
    EvaluationRunResult,
    EvaluatorFunction,
    ExecutorFunction,
    HumanEvaluator,
)
from lmnr.sdk.types import Datapoint


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
