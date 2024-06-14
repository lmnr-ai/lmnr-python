
import requests
import pydantic

class ChatMessage(pydantic.BaseModel):
    role: str
    content: str

type NodeInput = str | list[ChatMessage]

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
            self.error_code = resp_json['error_code']
            self.error_message = resp_json['error_message']
            super().__init__(self.error_message)
        except:
            super().__init__(response.text)
    
    def __str__(self) -> str:
        try:
            return str({'error_code': self.error_code, 'error_message': self.error_message})
        except:
            return super().__str__()
