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
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
import time


class RemoteDebugger:
    def __init__(
        self,
        project_api_key: str,
        dev_session_id: str,
        tools: Union[dict[str, NodeFunction], list[Callable[..., NodeInput]]] = [],
    ):
        # for simplicity and backwards compatibility, we allow the user to pass a list
        if isinstance(tools, list):
            tools = {f.__name__: NodeFunction(f.__name__, f) for f in tools}

        self.project_api_key = project_api_key
        self.url = "wss://api.lmnr.ai/v2/endpoint/ws"
        self.tools = tools
        self.stop_flag = False
        self.session = dev_session_id
        self.executor = ThreadPoolExecutor(5)
        self.running_tasks = {}  # dict[str, Future] from request_id to Future
    
    def start(self) -> Optional[str]:
        self.stop_flag = False
        self.executor.submit(self._run)
        return self.session

    def stop(self):
        self.stop_flag = True
        self.executor.shutdown()
        self.session = None

    def _run(self, backoff=1):
        assert self.session is not None, "Session ID not set"
        request = RegisterDebuggerRequest(debuggerSessionId=self.session)
        try:
            self._connect_and_run(request, backoff)
        except Exception as e:
            print(f"Could not connect to server. Retrying in {backoff} seconds...")
            time.sleep(backoff)
            self._run(min(backoff * 2, 60))
    
    def _connect_and_run(self, request: RegisterDebuggerRequest, backoff=1):
        with connect(
            self.url,
            additional_headers={"Authorization": f"Bearer {self.project_api_key}"},
        ) as websocket:
            websocket.send(request.model_dump_json())
            print(self._format_session_id_and_registerd_functions())
            req_id = None

            while not self.stop_flag:
                # first check if any of the running tasks are done
                done_tasks = []
                for req_id, future in self.running_tasks.items():
                    if not future.done():
                        continue
                    done_tasks.append(req_id)
                    try:
                        response = future.result()
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
                            reqId=req_id, response=response
                        )
                    except pydantic.ValidationError:
                        formatted_response = ToolCallResponse(
                            reqId=req_id, response=str(response)
                        )
                    websocket.send(formatted_response.model_dump_json())
                for req_id in done_tasks:
                    del self.running_tasks[req_id]
                try:
                    # blocks the thread until a message
                    # is received or a timeout (0.1 seconds) occurs
                    message = websocket.recv(0.1)
                except TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosedError:
                    print("Connection interrupted by server. Trying to reconnect...")
                    self._run()
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
                self.running_tasks[tool_call.reqId] = self.executor.submit(tool, **arguments)
            websocket.send(
                DeregisterDebuggerRequest(
                    debuggerSessionId=self.session, deregister=True
                ).model_dump_json()
            )

    def _format_session_id_and_registerd_functions(self) -> str:
        registered_functions = ",\n".join(["- " + k for k in self.tools.keys()])
        return f"""
========================================
Dev Session ID:
{self.session}
========================================

Registered functions:
{registered_functions}

========================================
"""
