from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional
import uuid


HandleType = str  # "String" | "ChatMessageList" | "Any"


@dataclass
class Handle:
    id: uuid.UUID
    name: Optional[str]
    type: HandleType

    @classmethod
    def from_dict(cls, dict: dict) -> "Handle":
        return cls(
            id=uuid.UUID(dict["id"]),
            name=(dict["name"] if "name" in dict else None),
            type=dict["type"],
        )


@abstractmethod
class NodeFunctions(metaclass=ABCMeta):
    @abstractmethod
    def handles_mapping(
        self, output_handle_id_to_node_name: dict[str, str]
    ) -> list[tuple[str, str]]:
        """
        Returns a list of tuples mapping from this node's input
        handle name to the unique name of the previous node.

        Assumes previous node has only one output.
        """
        pass

    @abstractmethod
    def node_type(self) -> str:
        pass

    @abstractmethod
    def config(self) -> dict:
        """
        Returns a dictionary of node-specific configuration.

        E.g. prompt and model name for LLM node.
        """
        pass
