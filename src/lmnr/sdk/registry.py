from typing import Callable

from lmnr.types import NodeFunction, NodeInput


class Registry:
    """
    Class to register and resolve node functions based on their node names.

    Node names cannot have space in their name.
    """

    functions: dict[str, NodeFunction]

    def __init__(self):
        self.functions = {}

    def add(self, node_name: str, function: Callable[..., NodeInput]):
        self.functions[node_name] = NodeFunction(node_name, function)

    def get(self, node_name: str) -> Callable[..., NodeInput]:
        return self.functions[node_name].function
