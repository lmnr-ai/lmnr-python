from dataclasses import dataclass

import uuid

from lmnr.cli.parser.nodes import Handle, NodeFunctions
from lmnr.cli.parser.utils import map_handles


@dataclass
class Dataset:
    id: uuid.UUID
    # created_at: datetime
    # project_id: uuid.UUID
    # name: str
    # indexed_on: Optional[str]

    @classmethod
    def from_dict(cls, dataset_dict: dict) -> "Dataset":
        return cls(
            id=uuid.UUID(dataset_dict["id"]),
        )


@dataclass
class SemanticSearchNode(NodeFunctions):
    id: uuid.UUID
    name: str
    inputs: list[Handle]
    outputs: list[Handle]
    inputs_mappings: dict[uuid.UUID, uuid.UUID]
    limit: int
    threshold: float
    template: str
    datasets: list[Dataset]

    def handles_mapping(
        self, output_handle_id_to_node_name: dict[str, str]
    ) -> list[tuple[str, str]]:
        return map_handles(
            self.inputs, self.inputs_mappings, output_handle_id_to_node_name
        )

    def node_type(self) -> str:
        return "SemanticSearch"

    def config(self) -> dict:
        return {
            "limit": self.limit,
            "threshold": self.threshold,
            "template": self.template,
            "datasource_ids": [str(dataset.id) for dataset in self.datasets],
            "datasource_ids_list": str([str(dataset.id) for dataset in self.datasets]),
        }
