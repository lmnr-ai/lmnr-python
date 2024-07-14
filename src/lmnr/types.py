import requests
import pydantic
import uuid
from typing import Union, Optional


class ChatMessage(pydantic.BaseModel):
    role: str
    content: str


class ConditionedValue(pydantic.BaseModel):
    condition: str
    value: "NodeInput"


NodeInput = Union[str, list[ChatMessage], ConditionedValue]  # TypeAlias


class EndpointRunRequest(pydantic.BaseModel):
    inputs: dict[str, NodeInput]
    endpoint: str
    env: dict[str, str] = pydantic.Field(default_factory=dict)
    metadata: dict[str, str] = pydantic.Field(default_factory=dict)


class EndpointRunResponse(pydantic.BaseModel):
    outputs: dict[str, dict[str, NodeInput]]
    run_id: str


class EndpointRunError(Exception):
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


class SDKError(Exception):
    def __init__(self, error_message: str):
        super().__init__(error_message)


class ToolCallFunction(pydantic.BaseModel):
    name: str
    arguments: str


class ToolCall(pydantic.BaseModel):
    id: Optional[str]
    type: Optional[str]
    function: ToolCallFunction


# TODO: allow snake_case and manually convert to camelCase
class ToolCallRequest(pydantic.BaseModel):
    reqId: uuid.UUID
    toolCall: ToolCall


class ToolCallResponse(pydantic.BaseModel):
    reqId: uuid.UUID
    response: NodeInput


class ToolCallError(pydantic.BaseModel):
    reqId: uuid.UUID
    error: str


class RegisterDebuggerRequest(pydantic.BaseModel):
    debuggerSessionId: str


class DeregisterDebuggerRequest(pydantic.BaseModel):
    debuggerSessionId: str
    deregister: bool
