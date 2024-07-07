from typing import Callable, Optional
from websockets.sync.client import connect
from lmnr.types import DeregisterDebuggerRequest, NodeInput, RegisterDebuggerRequest, SDKError, ToolCall
import uuid
import json
from threading import Thread

class RemoteDebugger:
    def __init__(self, project_api_key: str, tools: list[Callable[..., NodeInput]] = []):
        self.project_api_key = project_api_key
        self.url = 'wss://api.lmnr.ai/v2/endpoint/ws'
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
        # python allows running threads only once, so we need to create a new thread
        # in case the user wants to start the debugger again
        self.thread = Thread(target=self._run)
    
    def get_session_id(self) -> str:
        return self.session
    
    def _run(self):
        request = RegisterDebuggerRequest(debuggerSessionId=self.session)
        with connect(
            self.url,
            additional_headers={
                'Authorization': f'Bearer {self.project_api_key}'
            }
        ) as websocket:
            websocket.send(request.model_dump_json())
            print(self._format_session_id())
            while not self.stop_flag:
                try:
                    # blocks the thread until a message is received or a timeout (3 seconds) occurs
                    message = websocket.recv(3)
                except TimeoutError:
                    continue
                try:
                    tool_call = ToolCall.model_validate_json(message)
                except:
                    raise SDKError(f'Invalid message received:\n{message}')
                matching_tools = [
                    tool for tool in self.tools
                    if tool.__name__ == tool_call.function.name
                ]
                if not matching_tools:
                    raise SDKError(
                        f'Tool {tool_call.function.name} not found.'
                        ' Registered tools: '
                        f'{", ".join([tool.__name__ for tool in self.tools])}'
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
            websocket.send(DeregisterDebuggerRequest(debuggerSessionId=self.session, deregister=True).model_dump_json())

    def _generate_session_id(self) -> str:
        return uuid.uuid4().urn[9:]

    def _format_session_id(self) -> str:
        return \
f"""
========================================
Debugger Session ID:
{self.session}
========================================
"""
    