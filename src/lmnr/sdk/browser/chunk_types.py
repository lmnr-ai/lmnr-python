"""Wire format for chunked rrweb event messages sent from the browser.

This is a contract WE own on both ends: emitted by `inject_script.js`
(`createChunks` / `sendEvent`) and parsed here (`pw_utils.py`, `cdp_utils.py`) —
unlike the surrounding wrapt/CDP payloads, it is not a third-party shape, so it
is fully typed rather than `dict[str, Any]`.
"""

from typing import TypedDict


class ChunkMessage(TypedDict):
    """One `lmnrSendEvents` payload chunk, as emitted by `inject_script.js`.

    A batch of compressed rrweb events is JSON-stringified and, when too large
    for a single CDP/`expose_function` call, split into chunks of this shape
    by `createChunks`; a small batch is still sent as this shape with
    `chunkIndex=0, totalChunks=1`.
    """

    batchId: str
    chunkIndex: int
    totalChunks: int
    data: str
    isFinal: bool


class ChunkBuffer(TypedDict):
    """Reassembly state for one in-flight `batchId`, held in `chunk_buffers`."""

    chunks: dict[int, str]
    total: int
    timestamp: float
