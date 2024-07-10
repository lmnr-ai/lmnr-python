from dataclasses import dataclass
from typing import Optional
import uuid

from lmnr.cli.parser.nodes import Handle, HandleType, NodeFunctions
from lmnr.types import NodeInput


@dataclass
class InputNode(NodeFunctions):
    id: uuid.UUID
    name: str
    outputs: list[Handle]
    input: Optional[NodeInput]
    input_type: HandleType

    def handles_mapping(
        self, output_handle_id_to_node_name: dict[str, str]
    ) -> list[tuple[str, str]]:
        return []

    def node_type(self) -> str:
        return "Input"

    def config(self) -> dict:
        return {}
