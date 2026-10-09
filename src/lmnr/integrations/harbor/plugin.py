"""Harbor job plugin that reports Harbor runs to Laminar as evaluations.

Usage:

    harbor run --dataset terminal-bench@2.0 --agent claude-code \\
        --model anthropic/claude-sonnet-4-5 --plugin laminar

Each Harbor job becomes one Laminar evaluation. Each trial becomes a datapoint
whose trace contains the trial phases (environment setup, agent setup, agent,
verifier) and, when the agent writes an ATIF trajectory, one LLM / TOOL span per
recorded step. Verifier rewards become evaluation scores.
"""

import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from opentelemetry.context import Context
from opentelemetry.trace import Status, StatusCode

from lmnr.integrations.harbor.trajectory import (
    emit_trajectory_spans,
    final_agent_message,
    parse_timestamp_ns,
)
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
from lmnr.sdk.evaluations import get_evaluation_url
from lmnr.sdk.laminar import Laminar
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.types import EvaluationResultDatapoint, PartialEvaluationDatapoint
from lmnr.sdk.utils import from_env, json_dumps

if TYPE_CHECKING:
    from harbor.job import Job
    from harbor.models.job.result import JobResult
    from harbor.trial.hooks import TrialHookEvent

logger = get_default_logger(__name__)

DEFAULT_GROUP_NAME = "harbor"


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _uuid(value: int) -> uuid.UUID:
    return uuid.UUID(int=value)


@dataclass
class _TrialState:
    """A trial attempt in progress. Retries of a trial reuse its datapoint."""

    datapoint_id: uuid.UUID
    index: int
    data: dict[str, Any]
    metadata: dict[str, Any]
    root_span: Any = None
    trace_id: uuid.UUID | None = None
    cancelled: bool = False
    # Spans emitted while the trial is still running, so the trace shows up in
    # Laminar before the trial ends.
    setup_emitted: bool = False
    agent_span: Any = None
    agent_exception: Any = None
    # The trial's latest upload. Every upload waits for it, since saves upsert
    # by datapoint id and a retry must not be overwritten by an earlier attempt.
    last_upload: asyncio.Task | None = None


class LaminarPlugin:
    """Report Harbor jobs to Laminar as evaluations.

    Implements Harbor's `JobPlugin` protocol (`on_job_start` / `on_job_end`) and
    is registered under the `harbor.plugins` entry point as `laminar`. Options
    are passed with `--pk key=value` (Harbor ignores `plugins` in job configs):

    - `project_api_key`: Laminar project API key. Defaults to
      `LMNR_PROJECT_API_KEY`.
    - `base_url`, `http_port`, `grpc_port`: for self-hosted Laminar. Default to
      `LMNR_BASE_URL` and the SDK defaults.
    - `evaluation_name`: name of the Laminar evaluation. Defaults to the
      Harbor job name. Env: `HARBOR_LAMINAR_EVALUATION`.
    - `group_name`: evaluation group used to compare runs. Defaults to the
      dataset name (e.g. `terminal-bench@2.0`). Env: `HARBOR_LAMINAR_GROUP`.
    - `trajectory_spans`: convert agent ATIF trajectories into LLM and tool
      spans. Defaults to true.
    - `fail_fast`: raise on Laminar errors instead of logging a warning and
      letting the Harbor run continue. Defaults to false. Env:
      `HARBOR_LAMINAR_FAIL_FAST`.
    """

    def __init__(
        self,
        *,
        project_api_key: str | None = None,
        base_url: str | None = None,
        http_port: int | None = None,
        grpc_port: int | None = None,
        evaluation_name: str | None = None,
        group_name: str | None = None,
        trajectory_spans: bool | str | None = None,
        fail_fast: bool | str | None = None,
    ):
        self.project_api_key = project_api_key or from_env("LMNR_PROJECT_API_KEY")
        self.base_url = base_url or from_env("LMNR_BASE_URL")
        self.http_port = int(http_port) if http_port is not None else None
        self.grpc_port = int(grpc_port) if grpc_port is not None else None
        resolved_evaluation_name = evaluation_name or from_env(
            "HARBOR_LAMINAR_EVALUATION"
        )
        self.evaluation_name = (
            str(resolved_evaluation_name) if resolved_evaluation_name else None
        )
        resolved_group_name = group_name or from_env("HARBOR_LAMINAR_GROUP")
        self.group_name = str(resolved_group_name) if resolved_group_name else None
        self.trajectory_spans = (
            True if trajectory_spans is None else _parse_bool(trajectory_spans)
        )
        self.fail_fast = _parse_bool(
            fail_fast if fail_fast is not None else from_env("HARBOR_LAMINAR_FAIL_FAST")
        )

        self._client: AsyncLaminarClient | None = None
        self._eval_id: uuid.UUID | None = None
        self._project_id: str | None = None
        self._resolved_group_name: str = DEFAULT_GROUP_NAME
        self._trials: dict[str, _TrialState] = {}
        self._uploads: list[asyncio.Task] = []
        self._seen_reward_keys: list[str] = []
        self._next_index = 0

    # Harbor JobPlugin protocol

    async def on_job_start(self, job: "Job") -> None:
        try:
            await self._start_evaluation(job)
        except Exception as e:
            await self._close_client()
            self._handle_error("Failed to create Laminar evaluation", e)
            return

        job.on_trial_started(self._on_trial_started)
        job.on_agent_started(self._on_agent_started)
        job.on_verification_started(self._on_verification_started)
        job.on_trial_cancelled(self._on_trial_cancelled)
        job.on_trial_ended(self._on_trial_ended)

    async def on_job_end(self, job_result: "JobResult") -> None:
        if self._eval_id is None:
            return
        try:
            # Trials interrupted before END never got their root span closed.
            for state in self._trials.values():
                if state.root_span is not None:
                    state.root_span.set_status(Status(StatusCode.ERROR, "unfinished"))
                    state.root_span.end()
                    state.root_span = None
            await self._await_uploads()
            Laminar.flush()
        except Exception as e:
            self._handle_error("Failed to finish Laminar evaluation", e)
        finally:
            await self._close_client()
        logger.info(f"Laminar evaluation: {self.evaluation_url}")

    async def _close_client(self) -> None:
        if self._client is not None:
            client, self._client = self._client, None
            await client.close()

    # Job setup

    @property
    def evaluation_url(self) -> str | None:
        if self._eval_id is None or self._project_id is None:
            return None
        return get_evaluation_url(self._project_id, str(self._eval_id), self.base_url)

    async def _start_evaluation(self, job: "Job") -> None:
        if not self.project_api_key:
            raise ValueError(
                "Laminar project API key is not set. Set LMNR_PROJECT_API_KEY or "
                "pass --pk project_api_key=..."
            )
        if not Laminar.is_initialized():
            # Harbor-process agents (e.g. terminus-2) call LLMs through LiteLLM
            # outside of any span we own, so auto-instrumentation would only
            # produce orphan traces. Their calls are recovered from the
            # trajectory instead.
            Laminar.initialize(
                project_api_key=self.project_api_key,
                base_url=self.base_url,
                http_port=self.http_port,
                grpc_port=self.grpc_port,
                instruments=set(),
            )
        self._client = AsyncLaminarClient(
            base_url=self.base_url,
            project_api_key=self.project_api_key,
            port=self.http_port,
        )

        config = job.config
        self._resolved_group_name = (
            self.group_name or _default_group_name(job) or DEFAULT_GROUP_NAME
        )
        agents = [
            {"name": agent.name or agent.import_path, "model": agent.model_name}
            for agent in config.agents
        ]
        evaluation = await self._client.evals.init(
            name=self.evaluation_name or config.job_name,
            group_name=self._resolved_group_name,
            metadata={
                "source": "harbor",
                "harbor_job_id": str(job.id),
                "harbor_job_name": config.job_name,
                "agents": agents,
            },
        )
        self._eval_id = evaluation.id
        self._project_id = str(evaluation.projectId)
        logger.info(f"Laminar evaluation: {self.evaluation_url}")

    # Trial hooks

    async def _on_trial_started(self, event: "TrialHookEvent") -> None:
        try:
            self._start_trial(event)
        except Exception as e:
            self._handle_error(
                f"Failed to start Laminar trace for {event.trial_name}", e
            )

    async def _on_agent_started(self, event: "TrialHookEvent") -> None:
        # Environment and agent setup are done by now.
        try:
            state = self._trials.get(event.trial_name)
            if state is not None and state.root_span is not None:
                self._emit_setup(state, event)
        except Exception as e:
            self._handle_error(f"Failed to report {event.trial_name} setup", e)

    async def _on_verification_started(self, event: "TrialHookEvent") -> None:
        # The agent is done and its logs and trajectory are synced by now.
        try:
            state = self._trials.get(event.trial_name)
            if state is not None and state.root_span is not None:
                self._emit_agent_once(state, event)
        except Exception as e:
            self._handle_error(f"Failed to report {event.trial_name} agent run", e)

    async def _on_trial_cancelled(self, event: "TrialHookEvent") -> None:
        state = self._trials.get(event.trial_name)
        if state is not None:
            state.cancelled = True

    async def _on_trial_ended(self, event: "TrialHookEvent") -> None:
        try:
            self._end_trial(event)
        except Exception as e:
            self._handle_error(f"Failed to report {event.trial_name} to Laminar", e)

    def _start_trial(self, event: "TrialHookEvent") -> None:
        state = self._trials.get(event.trial_name)
        if state is None:
            agent = event.config.agent
            data = {
                "task_name": event.task_name,
                "instruction": _read_instruction(event),
            }
            state = _TrialState(
                datapoint_id=uuid.uuid4(),
                index=self._next_index,
                data=data,
                metadata={
                    "trial_name": event.trial_name,
                    "agent": agent.name or agent.import_path,
                    "model": agent.model_name,
                },
            )
            self._next_index += 1
            self._trials[event.trial_name] = state
        else:
            # A retry: close the previous attempt's trace if it never ended.
            if state.root_span is not None:
                state.root_span.end()
            state.cancelled = False
            state.setup_emitted = False
            state.agent_span = None
            state.agent_exception = None

        state.root_span = Laminar.start_span(
            event.trial_name,
            input=state.data,
            span_type="EVALUATION",
            # Every trial is its own trace, regardless of the caller's context.
            context=Context(),
            metadata={
                "evaluation_id": str(self._eval_id),
                "harbor.trial_name": event.trial_name,
                "harbor.trial_id": str(event.trial_id),
                "harbor.task_name": event.task_name,
            },
            start_time=parse_timestamp_ns(event.timestamp),
        )
        # An open root is never exported, so without a finished child the trace
        # doesn't exist until the first phase ends (env setup can take minutes).
        start_ns = parse_timestamp_ns(event.timestamp)
        Laminar.start_span(
            "trial_started",
            parent_span_context=state.root_span.get_laminar_span_context(),
            start_time=start_ns,
        ).end(end_time=start_ns)
        span_context = state.root_span.get_span_context()
        state.trace_id = _uuid(span_context.trace_id)
        partial = PartialEvaluationDatapoint(
            id=state.datapoint_id,
            data=state.data,
            target={},
            index=state.index,
            trace_id=state.trace_id,
            executor_span_id=_uuid(span_context.span_id),
            metadata=state.metadata,
        )
        state.last_upload = self._upload([partial], after=state.last_upload)

    def _emit_trial_spans(
        self,
        root: Any,
        state: _TrialState,
        result: Any,
        event: "TrialHookEvent",
    ) -> Any:
        """Emit the phase spans not emitted yet and end the trial's root span."""
        self._emit_setup(state, event)
        self._emit_agent_once(state, event)
        exception = result.exception_info
        # An exception the agent span doesn't carry happened after the agent ran,
        # so it goes on the verifier span, or on the root if there is none.
        unrecorded = None if exception == state.agent_exception else exception
        rewards = result.verifier_result.rewards if result.verifier_result else None
        verifier_output: dict[str, Any] = {"rewards": rewards}
        test_output = _read_text(_trial_dir(event), "verifier", "test-stdout.txt")
        if test_output and test_output.strip():
            verifier_output["test_output"] = test_output
        verifier_span = _emit_phase(
            root.get_laminar_span_context(),
            "verifier",
            result.verifier,
            span_type="EVALUATOR",
            input=_test_script(event),
            output=verifier_output,
            exception=unrecorded,
        )

        output: dict[str, Any] = {"rewards": rewards}
        if exception is not None:
            output["exception"] = {
                "type": exception.exception_type,
                "message": exception.exception_message,
            }
            if unrecorded is not None and verifier_span is None:
                _record_exception(root, exception)
            root.set_status(Status(StatusCode.ERROR, exception.exception_message))
        elif state.cancelled:
            root.set_status(Status(StatusCode.ERROR, "cancelled"))
        root.set_output(output)
        root.end(end_time=parse_timestamp_ns(result.finished_at))
        return state.agent_span

    def _emit_setup(self, state: _TrialState, event: "TrialHookEvent") -> None:
        if state.setup_emitted:
            return
        state.setup_emitted = True
        result = event.result
        parent = state.root_span.get_laminar_span_context()
        _emit_phase(
            parent,
            "environment_setup",
            result.environment_setup,
            input=_environment_info(event),
        )
        agent = event.config.agent
        _emit_phase(
            parent,
            "agent_setup",
            result.agent_setup,
            input={"agent": agent.name or agent.import_path, "model": agent.model_name},
            output=_agent_setup_output(event),
        )

    def _emit_agent_once(self, state: _TrialState, event: "TrialHookEvent") -> None:
        if state.agent_span is not None:
            return
        exception = event.result.exception_info
        state.agent_span = self._emit_agent(
            state.root_span.get_laminar_span_context(), event, exception
        )
        if state.agent_span is not None:
            state.agent_exception = exception

    def _end_trial(self, event: "TrialHookEvent") -> None:
        state = self._trials.get(event.trial_name)
        if state is None or state.root_span is None:
            return
        root = state.root_span
        result = event.result
        try:
            agent_span = self._emit_trial_spans(root, state, result, event)
        except Exception:
            # Don't leave the root open with a historical start time.
            root.set_status(Status(StatusCode.ERROR, "failed to report trial"))
            root.end()
            raise
        finally:
            state.root_span = None
        exception = result.exception_info
        rewards = result.verifier_result.rewards if result.verifier_result else None

        executor_span_id = (
            agent_span.get_span_context().span_id
            if agent_span is not None
            else root.get_span_context().span_id
        )
        metadata = {**state.metadata, "trial_uri": result.trial_uri}
        if exception is not None:
            metadata["exception_type"] = exception.exception_type
        if state.cancelled:
            metadata["cancelled"] = True
        datapoint = EvaluationResultDatapoint(
            id=state.datapoint_id,
            index=state.index,
            data=state.data,
            target={},
            executor_output=self._executor_output(event),
            scores=self._scores(rewards, cancelled=state.cancelled),
            trace_id=state.trace_id,
            executor_span_id=_uuid(executor_span_id),
            metadata=metadata,
        )
        state.last_upload = self._upload([datapoint], after=state.last_upload)

    def _emit_agent(
        self, parent: Any, event: "TrialHookEvent", exception: Any = None
    ) -> Any:
        result = event.result
        timing = result.agent_execution
        if timing is None:
            return None
        agent_result = result.agent_result
        span = Laminar.start_span(
            "agent",
            input=_read_instruction(event),
            span_type="EXECUTOR",
            parent_span_context=parent,
            attributes=_agent_attributes(result),
            start_time=parse_timestamp_ns(timing.started_at),
        )
        end_ns = parse_timestamp_ns(timing.finished_at)
        if self.trajectory_spans:
            trajectory_path = _trajectory_path(event)
            if trajectory_path is not None:
                try:
                    emit_trajectory_spans(
                        trajectory_path,
                        span.get_laminar_span_context(),
                        start_ns=parse_timestamp_ns(timing.started_at),
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not convert trajectory {trajectory_path}: {e}"
                    )
        span.set_output(self._executor_output(event))
        if agent_result is not None and agent_result.metadata:
            span.set_attribute(
                "harbor.agent.metadata", json_dumps(agent_result.metadata)
            )
        if exception is not None:
            _record_exception(span, exception)
            span.set_status(Status(StatusCode.ERROR, exception.exception_message))
        span.end(end_time=end_ns)
        return span

    def _executor_output(self, event: "TrialHookEvent") -> Any:
        trajectory_path = _trajectory_path(event)
        if trajectory_path is not None:
            message = final_agent_message(trajectory_path)
            if message:
                return message
        agent_result = event.result.agent_result
        if agent_result is not None and agent_result.metadata:
            return agent_result.metadata
        return None

    def _scores(
        self, rewards: dict[str, Any] | None, cancelled: bool
    ) -> dict[str, float | int]:
        if rewards:
            scores = {}
            for key, value in rewards.items():
                if isinstance(value, bool):
                    value = int(value)
                if isinstance(value, (int, float)):
                    scores[key] = value
                    if key not in self._seen_reward_keys:
                        self._seen_reward_keys.append(key)
            return scores
        if cancelled:
            return {}
        # Harbor counts a trial without rewards (e.g. the agent crashed) as 0
        # in its metrics; do the same so Laminar averages match Harbor's.
        return {key: 0 for key in self._seen_reward_keys or ["reward"]}

    # Uploads

    def _upload(
        self,
        datapoints: list[EvaluationResultDatapoint | PartialEvaluationDatapoint],
        after: asyncio.Task | None = None,
    ) -> asyncio.Task:
        client, eval_id = self._client, self._eval_id
        assert client is not None and eval_id is not None

        async def upload() -> None:
            if after is not None:
                # Upserts by id: saves of a datapoint must land in order.
                await asyncio.gather(after, return_exceptions=True)
            await client.evals.save_datapoints(
                eval_id, datapoints, self._resolved_group_name
            )

        task = asyncio.ensure_future(upload())
        self._uploads.append(task)
        return task

    async def _await_uploads(self) -> None:
        results = await asyncio.gather(*self._uploads, return_exceptions=True)
        self._uploads = []
        for result in results:
            if isinstance(result, Exception):
                self._handle_error("Failed to save Laminar datapoints", result)

    def _handle_error(self, message: str, error: Exception) -> None:
        if self.fail_fast:
            raise error
        logger.warning(f"{message}: {error}")


def _default_group_name(job: "Job") -> str | None:
    config = job.config
    if len(config.datasets) == 1:
        dataset = config.datasets[0]
        name = dataset.name or (dataset.path.name if dataset.path else None)
        if name is not None and dataset.version:
            return f"{name}@{dataset.version}"
        return name
    if len(config.tasks) == 1:
        try:
            return config.tasks[0].get_task_id().get_name().split("/")[-1]
        except Exception:
            return None
    return None


def _read_instruction(event: "TrialHookEvent") -> str | None:
    try:
        path = Path(event.config.task.get_local_path()) / "instruction.md"
        return path.read_text()
    except Exception:
        return None


def _trial_dir(event: "TrialHookEvent") -> Path | None:
    try:
        return Path(event.config.trials_dir) / event.config.trial_name
    except Exception:
        return None


def _trajectory_path(event: "TrialHookEvent") -> Path | None:
    trial_dir = _trial_dir(event)
    if trial_dir is None:
        return None
    path = trial_dir / "agent" / "trajectory.json"
    return path if path.is_file() else None


def _read_text(directory: Path | None, *parts: str) -> str | None:
    if directory is None:
        return None
    try:
        return directory.joinpath(*parts).read_text(errors="replace")
    except OSError:
        return None


def _environment_info(event: "TrialHookEvent") -> dict[str, Any] | None:
    """What the environment was asked for: its type, image, and resources."""
    env = event.config.environment
    env_type = getattr(env.type, "value", env.type) or env.import_path
    task_env = _task_environment(event)
    info: dict[str, Any] = {
        "type": env_type,
        "docker_image": task_env.get("docker_image"),
    }
    for key in ("cpus", "memory_mb", "storage_mb", "gpus"):
        override = getattr(env, f"override_{key}", None)
        info[key] = override if override is not None else task_env.get(key)
    info = {key: value for key, value in info.items() if value is not None}
    return info or None


def _task_environment(event: "TrialHookEvent") -> dict[str, Any]:
    """The `[environment]` table of the task's task.toml."""
    try:
        try:
            import tomllib
        except ImportError:
            # Python 3.10. Harbor itself requires 3.12+, so this only matters
            # outside a real Harbor run (e.g. our tests on 3.10).
            import tomli as tomllib

        task_toml = Path(event.config.task.get_local_path()) / "task.toml"
        return tomllib.loads(task_toml.read_text()).get("environment", {})
    except Exception:
        return {}


def _agent_setup_output(event: "TrialHookEvent") -> dict[str, Any] | None:
    output: dict[str, Any] = {}
    agent_info = event.result.agent_info
    if agent_info is not None and agent_info.version:
        output["version"] = agent_info.version
    # Installed agents may write their install logs here.
    trial_dir = _trial_dir(event)
    setup_dir = trial_dir / "agent" / "setup" if trial_dir is not None else None
    if setup_dir is not None and setup_dir.is_dir():
        logs = {
            path.name: text
            for path in sorted(setup_dir.iterdir())
            if path.is_file() and (text := _read_text(setup_dir, path.name))
        }
        if logs:
            output["logs"] = logs
    return output or None


def _test_script(event: "TrialHookEvent") -> str | None:
    try:
        tests_dir = Path(event.config.task.get_local_path()) / "tests"
        scripts = sorted(tests_dir.glob("test.*"))
    except Exception:
        return None
    return _read_text(tests_dir, scripts[0].name) if scripts else None


def _agent_attributes(result: Any) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    agent_info = result.agent_info
    if agent_info is not None:
        attributes["harbor.agent.name"] = agent_info.name
        if agent_info.version:
            attributes["harbor.agent.version"] = agent_info.version
    agent_result = result.agent_result
    if agent_result is not None:
        for key, attr in (
            ("n_input_tokens", "harbor.agent.input_tokens"),
            ("n_output_tokens", "harbor.agent.output_tokens"),
            ("n_cache_tokens", "harbor.agent.cache_tokens"),
            ("cost_usd", "harbor.agent.cost_usd"),
        ):
            value = getattr(agent_result, key, None)
            if value is not None:
                attributes[attr] = value
    return attributes


def _record_exception(span: Any, exception: Any) -> None:
    """Record Harbor's `ExceptionInfo` as an OTel exception event.

    Harbor only keeps the exception's type, message, and traceback as strings,
    so this builds the event `span.record_exception` would emit.
    """
    occurred_at = getattr(exception, "occurred_at", None)
    span.add_event(
        "exception",
        attributes={
            "exception.type": exception.exception_type,
            "exception.message": exception.exception_message,
            "exception.stacktrace": getattr(exception, "exception_traceback", "") or "",
            "exception.escaped": True,
        },
        timestamp=parse_timestamp_ns(occurred_at) if occurred_at else None,
    )


def _emit_phase(
    parent: Any,
    name: str,
    timing: Any,
    span_type: str = "DEFAULT",
    input: Any = None,
    output: Any = None,
    exception: Any = None,
) -> Any:
    if timing is None or timing.started_at is None:
        return None
    span = Laminar.start_span(
        name,
        input=input,
        span_type=span_type,
        parent_span_context=parent,
        start_time=parse_timestamp_ns(timing.started_at),
    )
    if output is not None:
        span.set_output(output)
    if exception is not None:
        _record_exception(span, exception)
        span.set_status(Status(StatusCode.ERROR, exception.exception_message))
    span.end(end_time=parse_timestamp_ns(timing.finished_at))
    return span
