from concurrent.futures import ThreadPoolExecutor
import datetime
import logging
from typing import Optional
import uuid
from dataclasses import dataclass
import queue

from .task import Task
from .action import NodeRunError, RunOutput
from .state import State
from lmnr_engine.types import Message, NodeInput


logger = logging.getLogger(__name__)


@dataclass
class ScheduledTask:
    status: str  # "Task" | "Err"  TODO: Use an enum
    task_name: Optional[str]


class RunError(Exception):
    outputs: dict[str, Message]


@dataclass
class Engine:
    tasks: dict[str, Task]
    active_tasks: set[str]
    depths: dict[str, int]
    outputs: dict[str, Message]
    env: dict[str, str]
    thread_pool_executor: ThreadPoolExecutor
    # TODO: Store thread pool executor's Futures here to have control
    # over them (e.g. cancel them)

    @classmethod
    def new(
        cls, thread_pool_executor: ThreadPoolExecutor, env: dict[str, str] = {}
    ) -> "Engine":
        return cls(
            tasks={},
            active_tasks=set(),
            depths={},
            outputs={},
            env=env,
            thread_pool_executor=thread_pool_executor,
        )

    @classmethod
    def with_tasks(
        cls,
        tasks: list[Task],
        thread_pool_executor: ThreadPoolExecutor,
        env: dict[str, str] = {},
    ) -> "Engine":
        dag = cls.new(thread_pool_executor, env=env)

        for task in tasks:
            dag.tasks[task.name] = task
            dag.depths[task.name] = 0

        return dag

    def override_inputs(self, inputs: dict[str, NodeInput]) -> None:
        for task in self.tasks.values():
            # TODO: Check that it's the Input type task
            if not task.prev:
                task.value = inputs[task.name]

    def run(self, inputs: dict[str, NodeInput]) -> dict[str, Message]:
        self.override_inputs(inputs)

        q = queue.Queue()

        input_tasks = []
        for task in self.tasks.values():
            if len(task.prev) == 0:
                input_tasks.append(task.name)

        for task_id in input_tasks:
            q.put(ScheduledTask(status="Task", task_name=task_id))

        while True:
            logger.info("Waiting for task from queue")
            scheduled_task: ScheduledTask = q.get()
            logger.info(f"Got task from queue: {scheduled_task}")
            if scheduled_task.status == "Err":
                # TODO: Abort all other threads
                raise RunError(self.outputs)

            task: Task = self.tasks[scheduled_task.task_name]  # type: ignore
            logger.info(f"Task next: {task.next}")

            if not task.next:
                try:
                    fut = self.execute_task(task, q)
                    fut.result()
                    if not self.active_tasks:
                        return self.outputs
                except Exception:
                    raise RunError(self.outputs)
            else:
                self.execute_task(task, q)

    def execute_task_inner(
        self,
        task: Task,
        queue: queue.Queue,
    ) -> None:
        task_id = task.name
        next = task.next
        input_states = task.input_states
        active_tasks = self.active_tasks
        tasks = self.tasks
        depths = self.depths
        depth = depths[task.name]
        outputs = self.outputs

        inputs: dict[str, NodeInput] = {}
        input_messages = []

        # Wait for inputs for this task to be set
        for handle_name, input_state in input_states.items():
            logger.info(f"Task {task_id} waiting for semaphore for {handle_name}")
            input_state.semaphore.acquire()
            logger.info(f"Task {task_id} acquired semaphore for {handle_name}")

            # Set the outputs of predecessors as inputs of the current
            output = input_state.get_state()
            # If at least one of the inputs is termination,
            # also terminate this task early and set its state to termination
            if output.status == "Termination":
                return
            message = output.get_out()

            inputs[handle_name] = message.value
            input_messages.append(message)

        start_time = datetime.datetime.now()

        try:
            if callable(task.value):
                res = task.value(**inputs, _env=self.env)
            else:
                res = RunOutput(status="Success", output=task.value)

            if res.status == "Success":
                id = uuid.uuid4()
                state = State.new(
                    Message(
                        id=id,
                        value=res.output,  # type: ignore
                        start_time=start_time,
                        end_time=datetime.datetime.now(),
                    )
                )
            else:
                assert res.status == "Termination"
                state = State.termination()

            is_termination = state.is_termination()
            logger.info(f"Task {task_id} executed")

            # remove the task from active tasks once it's done
            if task_id in active_tasks:
                active_tasks.remove(task_id)

            if depth > 0:
                # propagate reset once we enter the loop
                # TODO: Implement this for cycles
                raise NotImplementedError()

            if depth == 10:
                # TODO: Implement this for cycles
                raise NotImplementedError()

            if not next:
                # if there are no next tasks, we can terminate the graph
                outputs[task.name] = state.get_out()

            # push next tasks to the channel only if
            # the current task is not a termination
            for next_task_name in next:
                # we set the inputs of the next tasks
                # to the outputs of the current task
                next_task = tasks[next_task_name]

                # in majority of cases there will be only one handle name
                # however we need to handle the case when single output
                # is mapped to multiple inputs on the next node
                handle_names = []
                for k, v in next_task.handles_mapping:
                    if v == task.name:
                        handle_names.append(k)

                for handle_name in handle_names:
                    next_state = next_task.input_states[handle_name]
                    next_state.set_state_and_permits(state, 1)

                # push next tasks to the channel only if the task is not active
                # and current task is not a termination
                if not (next_task_name in active_tasks) and not is_termination:
                    active_tasks.add(next_task_name)
                    queue.put(
                        ScheduledTask(
                            status="Task",
                            task_name=next_task_name,
                        )
                    )

            # increment depth of the finished task
            depths[task_id] = depth + 1
        except NodeRunError as e:
            logger.exception(f"Execution failed [id: {task_id}]")

            error = Message(
                id=uuid.uuid4(),
                value=str(e),
                start_time=start_time,
                end_time=datetime.datetime.now(),
            )

            outputs[task.name] = error

            # terminate entire graph by sending err task
            queue.put(
                ScheduledTask(
                    status="Err",
                    task_name=None,
                )
            )

        except Exception:
            logger.exception(f"Execution failed [id: {task_id}]")
            error = Message(
                id=uuid.uuid4(),
                value="Unexpected server error",
                start_time=start_time,
                end_time=datetime.datetime.now(),
            )
            outputs[task.name] = error
            queue.put(
                ScheduledTask(
                    status="Err",
                    task_name=None,
                )
            )

    def execute_task(
        self,
        task: Task,
        queue: queue.Queue,
    ):
        return self.thread_pool_executor.submit(
            self.execute_task_inner,
            task,
            queue,
        )
