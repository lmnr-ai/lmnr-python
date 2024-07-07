from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
from typing import Optional, Union

from lmnr.types import ChatMessage
from lmnr_engine.engine import Engine
{% set function_names = cookiecutter._tasks.values() | selectattr('node_type', '!=', 'Input') | map(attribute='function_name') | join(', ') %}
from .nodes.functions import {{ function_names }}
from lmnr_engine.engine.task import Task


logger = logging.getLogger(__name__)


class PipelineRunnerError(Exception):
    pass


@dataclass
class PipelineRunOutput:
    value: Union[str, list[ChatMessage]]


# This class is not imported in other files and can be renamed to desired name
class {{ cookiecutter.class_name }}:
    thread_pool_executor: ThreadPoolExecutor

    def __init__(
        self, thread_pool_executor: Optional[ThreadPoolExecutor] = None
    ) -> None:
        # Set max workers to hard-coded value for now
        self.thread_pool_executor = (
            ThreadPoolExecutor(max_workers=10)
            if thread_pool_executor is None
            else thread_pool_executor
        )

    def run(
        self,
        inputs: dict[str, Union[str, list]],
        env: dict[str, str] = {},
    ) -> dict[str, PipelineRunOutput]:
        """
        Run the pipeline with the given graph

        Raises:
            PipelineRunnerError: if there is an error running the pipeline
        """
        logger.info("Running pipeline {{ cookiecutter.pipeline_name }}, pipeline_version: {{ cookiecutter.pipeline_version_name }}")

        run_inputs = {}
        for inp_name, inp in inputs.items():
            if isinstance(inp, str):
                run_inputs[inp_name] = inp
            else:
                assert isinstance(inp, list), f"Invalid input type: {type(inp)}"
                run_inputs[inp_name] = [ChatMessage.model_validate(msg) for msg in inp]

        tasks = []
        {% for task in cookiecutter._tasks.values() %}
        tasks.append(
            Task(
                name="{{ task.name }}",
                value={{ "''" if task.node_type == "Input" else task.function_name }},
                handles_mapping={{ task.handles_mapping }},
                prev=[
                    {% for prev in task.prev %}
                    "{{ prev }}",
                    {% endfor %}
                ],
                next=[
                    {% for next in task.next %}
                    "{{ next }}",
                    {% endfor %}
                ],
            )
        )
        {% endfor %}
        engine = Engine.with_tasks(tasks, self.thread_pool_executor, env=env)

        # TODO: Raise PipelineRunnerError with node_errors
        run_res = engine.run(run_inputs)
        return {
            name: PipelineRunOutput(value=output.value)
            for name, output in run_res.items()
        }
