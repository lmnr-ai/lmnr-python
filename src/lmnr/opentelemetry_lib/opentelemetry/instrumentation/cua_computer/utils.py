import base64
import orjson


def payload_to_base64url(payload_bytes: bytes) -> bytes:
    data = base64.b64encode(payload_bytes).decode("utf-8")
    url = f"data:image/png;base64,{data}"
    return orjson.dumps({"base64url": url})


def payload_to_placeholder(payload_bytes: bytes) -> str:
    return "<BINARY_BLOB_SCREENSHOT>"
