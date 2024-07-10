from dataclasses import dataclass
from datetime import datetime

import uuid

from lmnr.cli.parser.nodes import Handle, NodeFunctions
from lmnr.cli.parser.utils import map_handles


@dataclass
class FileMetadata:
    id: uuid.UUID
    created_at: datetime
    project_id: uuid.UUID
    filename: str


@dataclass
class Dataset:
    id: uuid.UUID
    created_at: datetime
    project_id: uuid.UUID
    name: str


@dataclass
class SemanticSearchDatasource:
    type: str
    id: uuid.UUID
    # TODO: Paste other fields here, use Union[FileMetadata, Dataset]

    @classmethod
    def from_dict(cls, datasource_dict: dict) -> "SemanticSearchDatasource":
        if datasource_dict["type"] == "File":
            return cls(
                type="File",
                id=uuid.UUID(datasource_dict["id"]),
            )
        elif datasource_dict["type"] == "Dataset":
            return cls(
                type="Dataset",
                id=uuid.UUID(datasource_dict["id"]),
            )
        else:
            raise ValueError(
                f"Invalid SemanticSearchDatasource type: {datasource_dict['type']}"
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
    datasources: list[SemanticSearchDatasource]

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
            "datasource_ids": [str(datasource.id) for datasource in self.datasources],
            "concatenated_datasource_ids": ",".join(
                str(datasource.id) for datasource in self.datasources
            ),
        }
