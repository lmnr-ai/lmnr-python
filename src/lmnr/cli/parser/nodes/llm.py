from dataclasses import dataclass
from typing import Optional
import uuid

from lmnr.cli.parser.nodes import Handle, NodeFunctions
from lmnr.cli.parser.utils import map_handles


@dataclass
class LLMNode(NodeFunctions):
    id: uuid.UUID
    name: str
    inputs: list[Handle]
    dynamic_inputs: list[Handle]
    outputs: list[Handle]
    inputs_mappings: dict[uuid.UUID, uuid.UUID]
    prompt: str
    model: str
    model_params: Optional[str]
    stream: bool
    structured_output_enabled: bool
    structured_output_max_retries: int
    structured_output_schema: Optional[str]
    structured_output_schema_target: Optional[str]

    def handles_mapping(
        self, output_handle_id_to_node_name: dict[str, str]
    ) -> list[tuple[str, str]]:
        combined_inputs = self.inputs + self.dynamic_inputs
        return map_handles(
            combined_inputs, self.inputs_mappings, output_handle_id_to_node_name
        )

    def node_type(self) -> str:
        return "LLM"

    def config(self) -> dict:
        # For easier access in the template separate the provider and model here
        provider, model = self.model.split(":", maxsplit=1)

        return {
            "prompt": self.prompt,
            "provider": provider,
            "model": model,
            "model_params": self.model_params,
            "stream": self.stream,
            "structured_output_enabled": self.structured_output_enabled,
            "structured_output_max_retries": self.structured_output_max_retries,
            "structured_output_schema": self.structured_output_schema,
            "structured_output_schema_target": self.structured_output_schema_target,
        }
