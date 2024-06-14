import json
from pydantic.alias_generators import to_snake
import requests
from .model import EndpointRunError, EndpointRunResponse, NodeInput, EndpointRunRequest
from typing import Optional

class Laminar:
    project_api_key: Optional[str] = None
    def __init__(self, project_api_key: str):
        self.project_api_key = project_api_key
        self.url = 'https://api.lmnr.ai/v2/endpoint/run'
    
    def run(
        self,
        endpoint: str,
        inputs: dict[str, NodeInput],
        env: dict[str, str] = {},
        metadata: dict[str, str] = {}
    ) -> EndpointRunResponse:
        if self.project_api_key is None:
            raise ValueError('Please initialize the Laminar object with your project API key')
        try:
            request = EndpointRunRequest(inputs = inputs, endpoint = endpoint, env = env, metadata = metadata)
        except Exception as e:
            raise ValueError(f'Invalid request: {e}')
        response = requests.post(
            self.url, 
            json=json.loads(request.model_dump_json()),
            headers={'Authorization': f'Bearer {self.project_api_key}'}
        )
        if response.status_code != 200:
            raise EndpointRunError(response)
        try:
            resp_json = response.json()
            keys = list(resp_json.keys())
            for key in keys:
                value = resp_json[key]
                del resp_json[key]
                resp_json[to_snake(key)] = value
            return EndpointRunResponse(**resp_json)
        except:
            raise EndpointRunError(response)
        