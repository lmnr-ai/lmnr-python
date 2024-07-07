from dataclasses import dataclass
from typing import Any, Optional, Union
import uuid
from lmnr.cli.parser.nodes import Handle, HandleType, NodeFunctions
from lmnr.cli.parser.utils import map_handles
from lmnr.types import NodeInput, ChatMessage


def node_input_from_json(json_val: Any) -> NodeInput:
    if isinstance(json_val, str):
        return json_val
    elif isinstance(json_val, list):
        return [ChatMessage.model_validate(msg) for msg in json_val]
    else:
        raise ValueError(f"Invalid NodeInput value: {json_val}")


# TODO: Convert to Pydantic
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


# TODO: Convert to Pydantic
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


# TODO: Convert to Pydantic
@dataclass
class OutputNode(NodeFunctions):
    id: uuid.UUID
    name: str
    inputs: list[Handle]
    outputs: list[Handle]
    inputs_mappings: dict[uuid.UUID, uuid.UUID]

    def handles_mapping(
        self, output_handle_id_to_node_name: dict[str, str]
    ) -> list[tuple[str, str]]:
        return map_handles(
            self.inputs, self.inputs_mappings, output_handle_id_to_node_name
        )

    def node_type(self) -> str:
        return "Output"

    def config(self) -> dict:
        return {}


Node = Union[InputNode, OutputNode, LLMNode]


def node_from_dict(node_dict: dict) -> Node:
    if node_dict["type"] == "Input":
        return InputNode(
            id=uuid.UUID(node_dict["id"]),
            name=node_dict["name"],
            outputs=[Handle.from_dict(handle) for handle in node_dict["outputs"]],
            input=node_input_from_json(node_dict["input"]),
            input_type=node_dict["inputType"],
        )
    elif node_dict["type"] == "Output":
        return OutputNode(
            id=uuid.UUID(node_dict["id"]),
            name=node_dict["name"],
            inputs=[Handle.from_dict(handle) for handle in node_dict["inputs"]],
            outputs=[Handle.from_dict(handle) for handle in node_dict["outputs"]],
            inputs_mappings={
                uuid.UUID(k): uuid.UUID(v)
                for k, v in node_dict["inputsMappings"].items()
            },
        )
    elif node_dict["type"] == "LLM":
        return LLMNode(
            id=uuid.UUID(node_dict["id"]),
            name=node_dict["name"],
            inputs=[Handle.from_dict(handle) for handle in node_dict["inputs"]],
            dynamic_inputs=[
                Handle.from_dict(handle) for handle in node_dict["dynamicInputs"]
            ],
            outputs=[Handle.from_dict(handle) for handle in node_dict["outputs"]],
            inputs_mappings={
                uuid.UUID(k): uuid.UUID(v)
                for k, v in node_dict["inputsMappings"].items()
            },
            prompt=node_dict["prompt"],
            model=node_dict["model"],
            model_params=(
                node_dict["modelParams"] if "modelParams" in node_dict else None
            ),
            stream=False,
            # TODO: Implement structured output
            structured_output_enabled=False,
            structured_output_max_retries=3,
            structured_output_schema=None,
            structured_output_schema_target=None,
        )
    else:
        raise ValueError(f"Node type {node_dict['type']} not supported")
