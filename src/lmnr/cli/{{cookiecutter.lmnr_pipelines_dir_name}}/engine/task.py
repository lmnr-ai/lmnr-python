from typing import Callable, Union

from .action import RunOutput
from .state import ExecState
from lmnr_engine.types import NodeInput


class Task:
    # unique identifier
    name: str
    # mapping from current node's handle name to previous node's unique name
    # assumes nodes have only one output
    handles_mapping: list[tuple[str, str]]
    # Value or a function that returns a value
    # Usually a function which waits for inputs from previous nodes
    value: Union[NodeInput, Callable[..., RunOutput]]  # TODO: Type this fully
    # unique node names of previous nodes
    prev: list[str]
    # unique node names of next nodes
    next: list[str]
    input_states: dict[str, ExecState]

    def __init__(
        self,
        name: str,
        handles_mapping: list[tuple[str, str]],
        value: Union[NodeInput, Callable[..., RunOutput]],
        prev: list[str],
        next: list[str],
    ) -> None:
        self.name = name
        self.handles_mapping = handles_mapping
        self.value = value
        self.prev = prev
        self.next = next
        self.input_states = {
            handle_name: ExecState.new() for (handle_name, _) in self.handles_mapping
        }
