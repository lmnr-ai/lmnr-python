from typing import Any, Union
import uuid

from lmnr.cli.parser.nodes import Handle
from lmnr.cli.parser.nodes.code import CodeNode
from lmnr.cli.parser.nodes.condition import ConditionNode
from lmnr.cli.parser.nodes.input import InputNode
from lmnr.cli.parser.nodes.llm import LLMNode
from lmnr.cli.parser.nodes.output import OutputNode
from lmnr.cli.parser.nodes.router import Route, RouterNode
from lmnr.cli.parser.nodes.semantic_search import (
    Dataset,
    SemanticSearchNode,
)
from lmnr.types import NodeInput, ChatMessage


def node_input_from_json(json_val: Any) -> NodeInput:
    if isinstance(json_val, str):
        return json_val
    elif isinstance(json_val, list):
        return [ChatMessage.model_validate(msg) for msg in json_val]
    else:
        raise ValueError(f"Invalid NodeInput value: {json_val}")


Node = Union[
    InputNode,
    OutputNode,
    ConditionNode,
    LLMNode,
    RouterNode,
    SemanticSearchNode,
    CodeNode,
]


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
    elif node_dict["type"] == "Condition":
        return ConditionNode(
            id=uuid.UUID(node_dict["id"]),
            name=node_dict["name"],
            inputs=[Handle.from_dict(handle) for handle in node_dict["inputs"]],
            outputs=[Handle.from_dict(handle) for handle in node_dict["outputs"]],
            inputs_mappings={
                uuid.UUID(k): uuid.UUID(v)
                for k, v in node_dict["inputsMappings"].items()
            },
            condition=node_dict["condition"],
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
    elif node_dict["type"] == "Router":
        return RouterNode(
            id=uuid.UUID(node_dict["id"]),
            name=node_dict["name"],
            inputs=[Handle.from_dict(handle) for handle in node_dict["inputs"]],
            outputs=[Handle.from_dict(handle) for handle in node_dict["outputs"]],
            inputs_mappings={
                uuid.UUID(k): uuid.UUID(v)
                for k, v in node_dict["inputsMappings"].items()
            },
            routes=[Route(name=route["name"]) for route in node_dict["routes"]],
            has_default_route=node_dict["hasDefaultRoute"],
        )
    elif node_dict["type"] == "SemanticSearch":
        return SemanticSearchNode(
            id=uuid.UUID(node_dict["id"]),
            name=node_dict["name"],
            inputs=[Handle.from_dict(handle) for handle in node_dict["inputs"]],
            outputs=[Handle.from_dict(handle) for handle in node_dict["outputs"]],
            inputs_mappings={
                uuid.UUID(k): uuid.UUID(v)
                for k, v in node_dict["inputsMappings"].items()
            },
            limit=node_dict["limit"],
            threshold=node_dict["threshold"],
            template=node_dict["template"],
            datasets=[Dataset.from_dict(ds) for ds in node_dict["datasets"]],
        )
    elif node_dict["type"] == "Code":
        return CodeNode(
            id=uuid.UUID(node_dict["id"]),
            name=node_dict["name"],
            inputs=[Handle.from_dict(handle) for handle in node_dict["inputs"]],
            outputs=[Handle.from_dict(handle) for handle in node_dict["outputs"]],
            inputs_mappings={
                uuid.UUID(k): uuid.UUID(v)
                for k, v in node_dict["inputsMappings"].items()
            },
        )
    else:
        raise ValueError(f"Node type {node_dict['type']} not supported")
