# import base64
import base64
from copy import deepcopy
from typing import Any

from lmnr.opentelemetry_lib.decorators import json_dumps
from pydantic import BaseModel


def screenshot_tool_output_formatter(output: Any) -> str:
    # output is of type BinaryAPIResponse, which implements
    # the iter_bytes method from httpx.Response

    return "<BINARY_BLOB_SCREENSHOT>"
    # The below implementation works, but it may consume the entire iterator,
    # making the response unusable after the formatter is called.
    # This is UNLESS somewhere in code output.read() (httpx.Response.read())
    # is called.
    # We cannot rely on that now, so we return a placeholder.
    # response_bytes = []
    # for chunk in output.iter_bytes():
    #     response_bytes.append(chunk)
    # response_base64 = base64.b64encode(response_bytes).decode("utf-8")
    # return f"data:image/png;base64,{response_base64}"


def process_tool_output_formatter(output: Any) -> str:
    if not isinstance(output, (dict, BaseModel)):
        return json_dumps(output)

    output = output.model_dump() if isinstance(output, BaseModel) else deepcopy(output)
    if "stderr_b64" in output:
        output["stderr"] = base64.b64decode(output["stderr_b64"]).decode("utf-8")
    if "stdout_b64" in output:
        output["stdout"] = base64.b64decode(output["stdout_b64"]).decode("utf-8")
    return json_dumps(output)
