from dataclasses import dataclass
import uuid

from lmnr.cli.parser.nodes import Handle, NodeFunctions
from lmnr.cli.parser.utils import map_handles


@dataclass
class Route:
    name: str


@dataclass
class RouterNode(NodeFunctions):
    id: uuid.UUID
    name: str
    inputs: list[Handle]
    outputs: list[Handle]
    inputs_mappings: dict[uuid.UUID, uuid.UUID]
    routes: list[Route]
    has_default_route: bool

    def handles_mapping(
        self, output_handle_id_to_node_name: dict[str, str]
    ) -> list[tuple[str, str]]:
        return map_handles(
            self.inputs, self.inputs_mappings, output_handle_id_to_node_name
        )

    def node_type(self) -> str:
        return "Router"

    def config(self) -> dict:
        return {
            "routes": str([route.name for route in self.routes]),
            "has_default_route": str(self.has_default_route),
        }
