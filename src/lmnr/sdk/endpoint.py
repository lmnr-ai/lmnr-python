import json
from pydantic.alias_generators import to_snake
import pydantic
import requests
from lmnr.types import (
    EndpointRunError, EndpointRunResponse, NodeInput, EndpointRunRequest,
    ToolCall, SDKError
)
from typing import Callable, Optional
from websockets.sync.client import connect

class Laminar:
    project_api_key: Optional[str] = None
    def __init__(self, project_api_key: str):
        """Initialize the Laminar object with your project API key

        Args:
            project_api_key (str):
                Project api key. Generate or view your keys
                in the project settings in the Laminar dashboard.
        """        
        self.project_api_key = project_api_key
        self.url = 'https://api.lmnr.ai/v2/endpoint/run'
        self.ws_url = 'wss://api.lmnr.ai/v2/endpoint/ws'
    
    def run (
        self,
        endpoint: str,
        inputs: dict[str, NodeInput],
        env: dict[str, str] = {},
        metadata: dict[str, str] = {},
        tools: list[Callable[..., NodeInput]] = [],
    ) -> EndpointRunResponse:
        """Runs the endpoint with the given inputs

        Args:
            endpoint (str): name of the Laminar endpoint
            inputs (dict[str, NodeInput]):
                inputs to the endpoint's target pipeline.
                Keys in the dictionary must match input node names
            env (dict[str, str], optional):
                Environment variables for the pipeline execution.
                Defaults to {}.
            metadata (dict[str, str], optional):
                any custom metadata to be stored
                with execution trace. Defaults to {}.
            tools (list[Callable[..., NodeInput]], optional):
                List of callable functions the execution can call as tools.
                If specified and non-empty, a bidirectional communication
                with Laminar API through websocket will be established.
                Defaults to [].

        Returns:
            EndpointRunResponse: response object containing the outputs
        
        Raises:
            ValueError: if project API key is not set
            EndpointRunError: if the endpoint run fails
            SDKError: if an error occurs on client side during the execution
        """        
        if self.project_api_key is None:
            raise ValueError(
                'Please initialize the Laminar object with' 
                ' your project API key'
            )
        if tools:
            return self._run_websocket(endpoint, inputs, env, metadata, tools)
        return self._run(endpoint, inputs, env, metadata)
    
    def _run(
        self,
        endpoint: str,
        inputs: dict[str, NodeInput],
        env: dict[str, str] = {},
        metadata: dict[str, str] = {}
    ) -> EndpointRunResponse:
        try:
            request = EndpointRunRequest(
                inputs = inputs,
                endpoint = endpoint,
                env = env,
                metadata = metadata
            )
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

    def _run_websocket(
        self,
        endpoint: str,
        inputs: dict[str, NodeInput],
        env: dict[str, str] = {},
        metadata: dict[str, str] = {},
        tools: list[Callable[..., NodeInput]] = [],
    ) -> EndpointRunResponse:
        try:
            request = EndpointRunRequest(
                inputs = inputs,
                endpoint = endpoint,
                env = env,
                metadata = metadata
            )
        except Exception as e:
            raise ValueError(f'Invalid request: {e}')

        with connect(
            self.ws_url,
            additional_headers={
                'Authorization': f'Bearer {self.project_api_key}'
            }
        ) as websocket:
            websocket.send(request.model_dump_json())

            while True:
                message = websocket.recv()
                try:
                    tool_call = ToolCall.model_validate_json(message)
                    matching_tools = [
                        tool for tool in tools
                        if tool.__name__ == tool_call.function.name
                    ]
                    if not matching_tools:
                        raise SDKError(
                            f'Tool {tool_call.function.name} not found.'
                            ' Registered tools: '
                            f'{", ".join([tool.__name__ for tool in tools])}'
                        )
                    tool = matching_tools[0]
                    if tool.__name__ == tool_call.function.name:
                        # default the arguments to an empty dictionary
                        arguments = {}
                        try:
                            arguments = json.loads(tool_call.function.arguments)
                        except:
                            pass
                        response = tool(**arguments)
                        websocket.send(json.dumps(response))
                except pydantic.ValidationError as e:
                    message_json = json.loads(message)
                    keys = list(message_json.keys())
                    for key in keys:
                        value = message_json[key]
                        del message_json[key]
                        message_json[to_snake(key)] = value
                    result = EndpointRunResponse.model_validate(message_json)
                    websocket.close()
                    return result
                except Exception:
                    websocket.close()
                    raise SDKError('Error communicating to backend through websocket')
