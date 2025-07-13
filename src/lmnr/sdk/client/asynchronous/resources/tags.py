"""Resource for tagging traces."""

import json
import uuid

from lmnr.sdk.client.asynchronous.resources.base import BaseAsyncResource
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.utils import format_id

logger = get_default_logger(__name__)


class AsyncTags(BaseAsyncResource):
    """Resource for tagging traces."""

    async def tag(
        self,
        trace_id: str | int | uuid.UUID,
        tags: list[str] | str,
    ):
        """Tag a trace with a list of tags. Note that the trace must be ended
        before tagging it. You may want to call `Laminar.flush()` after the
        trace that you want to tag.

        Args:
            trace_id (str | int | uuid.UUID): The trace id to tag.
            tags (list[str] | str): The tag or list of tags to add to the trace.

        Raises:
            ValueError: If the trace id is not a valid UUID.

        Returns:
            list[dict]: The response from the server.

        Example:
        ```python
        from lmnr import Laminar, AsyncLaminarClient, observe

        Laminar.initialize()
        client = AsyncLaminarClient()
        trace_id = None

        @observe()
        def foo():
            trace_id = Laminar.get_trace_id()
            pass

        # make sure `foo` is called outside a trace context
        foo()

        # or make sure the trace is ended by this point
        Laminar.flush()

        await client.tags.tag(trace_id, "my_tag")
        ```
        """
        trace_tags = tags if isinstance(tags, list) else [tags]
        formatted_trace_id = format_id(trace_id)

        url = self._base_url + "/v1/tag"
        payload = {
            "traceId": formatted_trace_id,
            "names": trace_tags,
        }
        response = await self._client.post(
            url,
            content=json.dumps(payload),
            headers={
                **self._headers(),
            },
        )

        if response.status_code == 404:
            logger.warning(
                f"Trace {formatted_trace_id} not found. The trace may have not been ended yet."
            )
            return []

        if response.status_code != 200:
            raise ValueError(
                f"Failed to tag trace: [{response.status_code}] {response.text}"
            )
        return response.json()
