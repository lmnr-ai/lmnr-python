"""Harbor adapters that run an `EvalRunner` (`lmnr.evals`) as a trial's agent and
verifier. lmnr-cli writes them into the job config for `agent/eval.py` datasets:

    agents:
      - import_path: lmnr.integrations.harbor.runner:LaminarAgent
        kwargs: {runner: agent/eval.py:SupportEval}
    verifier:
      import_path: lmnr.integrations.harbor.runner:LaminarVerifier
      kwargs: {runner: agent/eval.py:SupportEval}

The runner path is relative to Harbor's cwd, the dataset folder. Unlike the rest
of `lmnr`, this module imports Harbor at runtime: only Harbor loads it.
"""

from __future__ import annotations

import asyncio
import json
import os
import traceback
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import (  # pyright: ignore[reportMissingImports]
    BaseInstalledAgent,
)
from harbor.models.trial.config import (  # pyright: ignore[reportMissingImports]
    TrialConfig,
)
from harbor.models.trial.paths import (  # pyright: ignore[reportMissingImports]
    EnvironmentPaths,
)
from harbor.models.verifier.result import (  # pyright: ignore[reportMissingImports]
    VerifierResult,
)
from harbor.utils.env import resolve_env_vars  # pyright: ignore[reportMissingImports]
from harbor.verifier.base import BaseVerifier  # pyright: ignore[reportMissingImports]

from lmnr.evals.context import (
    RESULT_FILE,
    AgentContext,
    Case,
    Secrets,
    VerifyContext,
    normalize_rewards,
)
from lmnr.evals.errors import EvalConfigError
from lmnr.evals.mocks.host import RunningMocks, collect_mocks, start_mock_server
from lmnr.evals.runner import EvalRunner, load_runner
from lmnr.evals.services.base import snapshot_services, start_services
from lmnr.integrations.harbor.spans import verifier_spans
from lmnr.sdk.laminar import Laminar
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

# How long `collect` may take after `run` was cancelled (e.g. the agent timed out).
COLLECT_AFTER_CANCEL_TIMEOUT_SEC = 120


class LaminarAgent(BaseInstalledAgent):  # pyright: ignore[reportUntypedBaseClass]
    """Runs the `EvalRunner`'s `setup`, `run` and `collect` in the agent container.

    `runner` is `"<path to eval.py>:<class>"`, relative to Harbor's cwd.
    """

    def __init__(self, logs_dir: Path, runner: str, *args: Any, **kwargs: Any):
        super().__init__(logs_dir, *args, **kwargs)
        self.runner: str = runner

    @staticmethod
    def name() -> str:
        return "laminar-eval-runner"

    async def install(self, environment: Any) -> None:
        pass  # lmnr-cli built the image before the run

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        runner_cls, eval_path = load_runner(self.runner)
        runner = runner_cls()
        # Harbor already sets `agents[].env` (lmnr-cli writes the runner's env
        # there) on every command of the agent phase, so only what it lacks is
        # added per command.
        harbor_env: dict[str, str] = dict(self.extra_env)
        runner_env = runner_cls.resolved_env(dict(os.environ))
        ctx = AgentContext(
            environment=environment,
            case=Case.from_dir(self._task_dir()),
            runner_dir=eval_path.parent,
            model=self.model_name or runner_cls.model,
            env={**runner_env, **{k: v for k, v in harbor_env.items() if k in runner_cls.env}},
            exec_env={k: v for k, v in runner_env.items() if k not in harbor_env},
            host_logs_dir=self.logs_dir,
        )

        # The trial's steps, in order: services, the mock server, setup, run,
        # collect.
        if runner_cls.services:
            service_env = await start_services(runner_cls.services, ctx)
            # Harbor's own agents[].env wins over per-command env anyway.
            ctx.add_env(service_env, {k: v for k, v in service_env.items() if k not in harbor_env})
        mocks = None
        specs = runner_cls.mock_specs()
        if specs:
            mocks = await start_mock_server(ctx, specs, eval_path=eval_path)
            ctx.add_env(mocks.env, {k: v for k, v in mocks.env.items() if k not in harbor_env})
        # Once the mock server runs, every exit goes through _collect: it
        # stops the server and saves its call log, which Harbor's verifier
        # still reads after an agent timeout (setup's included). `collect()`
        # itself only runs once setup finished.
        set_up = False
        try:
            await runner.setup(ctx)
            set_up = True
            await runner.run(ctx)
        except asyncio.CancelledError:
            await _collect_after_failure(
                runner, ctx, mocks, user_collect=set_up, timeout=COLLECT_AFTER_CANCEL_TIMEOUT_SEC
            )
            raise
        except Exception as error:
            try:
                await _collect_after_failure(runner, ctx, mocks, user_collect=set_up)
            except asyncio.CancelledError:
                # Cancelled while collecting (the agent timed out, say): the
                # run's error is still what failed the trial, so it wins.
                raise error from None
            raise
        await _collect(runner, ctx, mocks)

    def populate_context_post_run(self, context: Any) -> None:
        try:
            result = json.loads((self.logs_dir / RESULT_FILE).read_text())
        except (OSError, ValueError):
            return
        if isinstance(result, dict) and result:
            context.metadata = {**(context.metadata or {}), **result}

    def _task_dir(self) -> Path:
        """Harbor doesn't give the agent its task; the trial's config.json has it."""
        config = TrialConfig.model_validate_json((self.logs_dir.parent / "config.json").read_text())
        return Path(config.task.get_local_path())


class LaminarVerifier(BaseVerifier):  # pyright: ignore[reportUntypedBaseClass]
    """Grades a trial with the `EvalRunner`'s `verify(ctx)`, in the Harbor process.

    `ctx.container` is the environment Harbor passes: the agent container in
    `shared` mode, the fresh verifier container in `separate` mode.
    With the Laminar plugin, `verify()` is traced under the trial's verifier span.
    """

    def __init__(self, *, runner: str, **kwargs: Any):
        super().__init__(**kwargs)
        self.runner: str = runner

    async def verify(self) -> Any:
        runner_cls, eval_path = load_runner(self.runner)
        if runner_cls.grade_with_test_sh:
            raise EvalConfigError(
                f"{runner_cls.__name__} sets grade_with_test_sh = True; use Harbor's default "
                + "verifier."
            )
        runner = runner_cls()
        trial_paths = self.trial_paths
        verifier_config = self.task.config.verifier
        mode = verifier_config.environment_mode
        # Harbor's default: separate when the verifier has its own environment.
        if mode is None:
            verifier_mode = "separate" if verifier_config.environment else "shared"
        else:
            verifier_mode = (
                "separate" if str(getattr(mode, "value", mode)) == "separate" else "shared"
            )
        runner_env = runner_cls.resolved_env(dict(os.environ))
        # Services run in the agent container, which only a shared verifier sees.
        if verifier_mode == "shared":
            for service in runner_cls.services:
                runner_env = {**runner_env, **service.env()}
        # What Harbor's own verifier passes to tests/test.sh.
        verifier_env = resolve_env_vars(
            {
                **self.task.config.verifier.env,
                **(self.verifier_env or {}),
                **self.override_env,
            }
        )
        ctx = VerifyContext(
            environment=self.environment,
            case=Case.from_dir(self.task.paths.task_dir),
            runner_dir=eval_path.parent,
            model=_trial_model(trial_paths.config_path) or runner_cls.model,
            env=runner_env,
            exec_env={**runner_env, **verifier_env},
            host_agent_dir=trial_paths.agent_dir,
            secrets=Secrets(runner_cls.secret_names(), os.environ),
            host_artifacts_dir=trial_paths.artifacts_dir,
            artifacts=list(runner_cls.artifacts),
            verifier_mode=verifier_mode,
            mock_keys=list(runner_cls.mocks),
            snapshot_options={s.name: s.snapshot_option for s in runner_cls.services},
        )

        try:
            rewards = await self._run_verify(runner, ctx, trial_paths.trial_dir.name)
        except Exception:
            self._write_log(ctx, runner_cls, traceback.format_exc())
            raise
        self._write_log(ctx, runner_cls, f"rewards: {json.dumps(rewards)}")
        await self._write_rewards(ctx, rewards)
        return VerifierResult(rewards=rewards)

    async def _run_verify(
        self, runner: EvalRunner, ctx: VerifyContext, trial_name: str
    ) -> dict[str, float | int]:
        parent = verifier_spans.get(trial_name)
        if parent is None:
            # No Laminar plugin in this run: nothing to nest under.
            return normalize_rewards(await runner.verify(ctx))
        with Laminar.start_as_current_span(
            "verify", input={"case": ctx.case.name}, parent_span_context=parent
        ) as span:
            rewards = normalize_rewards(await runner.verify(ctx))
            span.set_output(rewards)
        return rewards

    def _write_log(self, ctx: VerifyContext, runner_cls: type[EvalRunner], last: str) -> None:
        """The human-readable log, where Harbor's verifier writes test.sh's output
        (the plugin shows it on the verifier span)."""
        lines = [f"graded by {runner_cls.__name__}.verify() in {self.runner}", *ctx.logger.lines]
        path = self.trial_paths.test_stdout_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _ = path.write_text("\n".join([*lines, last]) + "\n")
        except OSError as e:
            logger.warning(f"Could not write {path}: {e}")

    async def _write_rewards(self, ctx: VerifyContext, rewards: dict[str, float | int]) -> None:
        host_path = self.trial_paths.reward_json_path
        host_path.parent.mkdir(parents=True, exist_ok=True)
        _ = host_path.write_text(json.dumps(rewards))
        env_paths = EnvironmentPaths.for_os(self.environment.os)
        if not self.environment.is_path_mounted(
            env_paths.verifier_dir, self.trial_paths.verifier_dir
        ):
            await ctx.container.write_file(str(env_paths.reward_json_path), json.dumps(rewards))


async def _collect(
    runner: EvalRunner,
    ctx: AgentContext,
    mocks: RunningMocks | None,
    *,
    user_collect: bool = True,
) -> None:
    """`collect` (unless setup failed), then the services' snapshots and the
    mock call log (each also when an earlier one raised)."""
    try:
        if user_collect:
            await runner.collect(ctx)
    finally:
        try:
            await snapshot_services(type(runner).services, ctx)
        finally:
            if mocks is not None:
                await collect_mocks(ctx, mocks)


async def _collect_after_failure(
    runner: EvalRunner,
    ctx: AgentContext,
    mocks: RunningMocks | None,
    *,
    user_collect: bool,
    timeout: float | None = None,
) -> None:
    """`_collect` after `setup` or `run` raised. Its own error is logged, so
    the phase's error wins."""
    try:
        async with asyncio.timeout(timeout):
            await _collect(runner, ctx, mocks, user_collect=user_collect)
    except Exception:
        what = "run()" if user_collect else "setup()"
        logger.warning(f"collect failed after {what} failed:\n{traceback.format_exc()}")


def _trial_model(config_path: Path) -> str | None:
    try:
        config = TrialConfig.model_validate_json(config_path.read_text())
    except Exception:
        return None
    return config.agent.model_name
