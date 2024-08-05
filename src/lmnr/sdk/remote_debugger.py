from typing import Callable, Optional, Union
from websockets.sync.client import connect
import pydantic
import websockets
from lmnr.types import (
    DeregisterDebuggerRequest,
    NodeFunction,
    NodeInput,
    RegisterDebuggerRequest,
    SDKError,
    ToolCallError,
    ToolCallRequest,
    ToolCallResponse,
)
import uuid
import json
from threading import Thread


class RemoteDebugger:
    def __init__(
        self,
        project_api_key: str,
        tools: Union[dict[str, NodeFunction], list[Callable[..., NodeInput]]] = [],
    ):
        # for simplicity and backwards compatibility, we allow the user to pass a list
        if isinstance(tools, list):
            tools = {f.__name__: NodeFunction(f.__name__, f) for f in tools}

        self.project_api_key = project_api_key
        self.url = "wss://api.lmnr.ai/v2/endpoint/ws"
        self.tools = tools
        self.thread = Thread(target=self._run)
        self.stop_flag = False
        self.session = None

    def start(self) -> Optional[str]:
        self.stop_flag = False
        self.session = self._generate_session_id()
        self.thread.start()
        return self.session

    def stop(self):
        self.stop_flag = True
        self.thread.join()
        self.session = None
        # python allows running threads only once, so we need to create
        # a new thread
        # in case the user wants to start the debugger again
        self.thread = Thread(target=self._run)

    def _run(self):
        assert self.session is not None, "Session ID not set"
        request = RegisterDebuggerRequest(debuggerSessionId=self.session)
        with connect(
            self.url,
            additional_headers={"Authorization": f"Bearer {self.project_api_key}"},
        ) as websocket:
            websocket.send(request.model_dump_json())
            print(self._format_session_id_and_registerd_functions())
            req_id = None

            while not self.stop_flag:
                try:
                    # blocks the thread until a message
                    # is received or a timeout (3 seconds) occurs
                    message = websocket.recv(3)
                except TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosedError:
                    print("Connection closed. Please restart the debugger.")
                    return
                try:
                    tool_call = ToolCallRequest.model_validate_json(message)
                    req_id = tool_call.reqId
                except Exception:
                    raise SDKError(f"Invalid message received:\n{message}")
                matching_tool = self.tools.get(tool_call.toolCall.function.name)
                if matching_tool is None:
                    error_message = (
                        f"Tool {tool_call.toolCall.function.name} not found"
                        + ". Registered tools: "
                        + ", ".join(self.tools.keys())
                    )
                    e = ToolCallError(error=error_message, reqId=req_id)
                    websocket.send(e.model_dump_json())
                    continue
                tool = matching_tool.function

                # default the arguments to an empty dictionary
                arguments = {}
                try:
                    arguments = json.loads(tool_call.toolCall.function.arguments)
                except Exception:
                    pass
                try:
                    response = tool(**arguments)
                except Exception as e:
                    error_message = (
                        "Error occurred while running tool" + f"{tool.__name__}: {e}"
                    )
                    e = ToolCallError(error=error_message, reqId=req_id)
                    websocket.send(e.model_dump_json())
                    continue
                formatted_response = None
                try:
                    formatted_response = ToolCallResponse(
                        reqId=tool_call.reqId, response=response
                    )
                except pydantic.ValidationError:
                    formatted_response = ToolCallResponse(
                        reqId=tool_call.reqId, response=str(response)
                    )
                websocket.send(formatted_response.model_dump_json())
            websocket.send(
                DeregisterDebuggerRequest(
                    debuggerSessionId=self.session, deregister=True
                ).model_dump_json()
            )

    def _generate_session_id(self) -> str:
        return uuid.uuid4().urn[9:]

    def _format_session_id_and_registerd_functions(self) -> str:
        registered_functions = ",\n".join(["- " + k for k in self.tools.keys()])
        return f"""
========================================
Debugger Session ID:
{self.session}
========================================

Registered functions:
{registered_functions}

========================================
"""
