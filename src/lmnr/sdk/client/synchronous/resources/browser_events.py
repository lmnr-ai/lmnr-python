"""Resource for sending browser events."""

import gzip
import json

from lmnr.sdk.client.synchronous.resources.base import BaseResource

from lmnr.version import PYTHON_VERSION, __version__


class BrowserEvents(BaseResource):
    """Resource for sending browser events."""

    def send(
        self,
        session_id: str,
        trace_id: str,
        events: list[dict],
    ):
        url = self._base_url + "/v1/browser-sessions/events"
        payload = {
            "sessionId": session_id,
            "traceId": trace_id,
            "events": events,
            "source": f"python@{PYTHON_VERSION}",
            "sdkVersion": __version__,
        }
        compressed_payload = gzip.compress(json.dumps(payload).encode("utf-8"))
        response = self._client.post(
            url,
            content=compressed_payload,
            headers={
                **self._headers(),
                "Content-Encoding": "gzip",
            },
        )
        if response.status_code != 200:
            raise ValueError(
                f"Failed to send events: [{response.status_code}] {response.text}"
            )
