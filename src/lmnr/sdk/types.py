import requests
import pydantic
import uuid
from typing import Optional, Union


class ChatMessage(pydantic.BaseModel):
    role: str
    content: str


class ConditionedValue(pydantic.BaseModel):
    condition: str
    value: "NodeInput"


NodeInput = Union[str, list[ChatMessage], ConditionedValue]  # TypeAlias


class PipelineRunRequest(pydantic.BaseModel):
    inputs: dict[str, NodeInput]
    pipeline: str
    env: dict[str, str] = pydantic.Field(default_factory=dict)
    metadata: dict[str, str] = pydantic.Field(default_factory=dict)
    stream: bool = pydantic.Field(default=False)
    parent_span_id: Optional[uuid.UUID] = pydantic.Field(default=None)
    trace_id: Optional[uuid.UUID] = pydantic.Field(default=None)

    # uuid is not serializable by default, so we need to convert it to a string
    def to_dict(self):
        return {
            "inputs": {
                k: v.model_dump() if isinstance(v, pydantic.BaseModel) else v
                for k, v in self.inputs.items()
            },
            "pipeline": self.pipeline,
            "env": self.env,
            "metadata": self.metadata,
            "stream": self.stream,
            "parentSpanId": str(self.parent_span_id) if self.parent_span_id else None,
            "traceId": str(self.trace_id) if self.trace_id else None,
        }


class PipelineRunResponse(pydantic.BaseModel):
    outputs: dict[str, dict[str, NodeInput]]
    run_id: str


class PipelineRunError(Exception):
    error_code: str
    error_message: str

    def __init__(self, response: requests.Response):
        try:
            resp_json = response.json()
            self.error_code = resp_json["error_code"]
            self.error_message = resp_json["error_message"]
            super().__init__(self.error_message)
        except Exception:
            super().__init__(response.text)

    def __str__(self) -> str:
        try:
            return str(
                {"error_code": self.error_code, "error_message": self.error_message}
            )
        except Exception:
            return super().__str__()
