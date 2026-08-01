"""A 413 must surface the server's message in human-readable form (LAM-2050)."""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from lmnr.sdk.client.synchronous.sync_client import LaminarClient
from lmnr.sdk.types import PartialEvaluationDatapoint
from lmnr.sdk.utils import MAX_ERROR_BODY_CHARS, describe_response


@pytest.fixture
def sync_client():
    client = LaminarClient(base_url="http://test-api.com", project_api_key="test-key")
    yield client
    client.close()


@pytest.fixture
def sample_datapoints():
    return [
        PartialEvaluationDatapoint(
            id=uuid.uuid4(),
            data={"input": "large data " * 1000},
            target={"expected": "output"},
            index=0,
            trace_id=uuid.uuid4(),
            executor_span_id=uuid.uuid4(),
            metadata={"test": "metadata"},
        )
    ]

# What app-server actually returns for a payload-limit rejection: plain text, not JSON.
SERVER_413_BODY = (
    "Payload too large: request body is 60044 bytes, which exceeds the server's HTTP "
    "payload limit of 10000 bytes. Send fewer spans per batch, or reduce the size of "
    "individual span inputs and outputs. Self-hosted deployments can raise the "
    "HTTP_PAYLOAD_LIMIT environment variable."
)


def _response(status_code: int, text: str) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.text = text
    # A plain-text body is not JSON — anything that calls .json() on it must blow up,
    # which is what makes the "don't assume JSON" behaviour meaningful.
    response.json.side_effect = ValueError("not json")
    return response


class TestDescribeResponse:
    def test_plain_text_body_is_rendered_with_status(self):
        rendered = describe_response(_response(413, SERVER_413_BODY))
        assert rendered.startswith("[413] Payload too large:")
        assert "HTTP_PAYLOAD_LIMIT" in rendered

    def test_missing_response_does_not_raise(self):
        assert describe_response(None) == "no response received"

    def test_empty_body_is_labelled(self):
        assert describe_response(_response(413, "")) == "[413] <empty body>"

    def test_huge_body_is_truncated_so_it_cannot_flood_logs(self):
        rendered = describe_response(_response(500, "x" * (MAX_ERROR_BODY_CHARS + 5000)))
        assert rendered.endswith("... (truncated)")
        assert len(rendered) < MAX_ERROR_BODY_CHARS + 200


class TestSaveDatapoints413:
    def test_persistent_413_reports_status_and_server_message(
        self, sync_client, sample_datapoints
    ):
        """On a persistent 413 the user used to get a bare 'Error saving evaluation
        datapoints' with no status code and no server body."""
        eval_id = uuid.uuid4()
        response = _response(413, SERVER_413_BODY)

        with patch.object(sync_client.evals._client, "post", return_value=response):
            with pytest.raises(ValueError) as excinfo:
                sync_client.evals.save_datapoints(eval_id, sample_datapoints)

        message = str(excinfo.value)
        assert "[413]" in message
        assert "HTTP_PAYLOAD_LIMIT" in message

    def test_shrinking_to_nothing_reports_the_server_message(
        self, sync_client, sample_datapoints
    ):
        """The length==0 branch: data was truncated away and the server still says 413."""
        eval_id = uuid.uuid4()
        response = _response(413, SERVER_413_BODY)

        with patch.object(sync_client.evals._client, "post", return_value=response):
            with pytest.raises(ValueError) as excinfo:
                sync_client.evals._retry_save_datapoints(
                    eval_id, sample_datapoints, initial_length=2
                )

        message = str(excinfo.value)
        assert "truncating datapoint data to nothing" in message
        assert "[413]" in message
        assert "HTTP_PAYLOAD_LIMIT" in message

    def test_shrinking_to_nothing_before_any_request_does_not_crash(
        self, sync_client, sample_datapoints
    ):
        """Regression guard: the length==0 branch can be reached on the first iteration,
        when no response exists yet — it must report that, not raise UnboundLocalError."""
        with patch.object(sync_client.evals._client, "post") as mock_post:
            with pytest.raises(ValueError, match="no response received"):
                sync_client.evals._retry_save_datapoints(
                    uuid.uuid4(), sample_datapoints, initial_length=1
                )
            mock_post.assert_not_called()

    def test_non_413_failure_reports_status_and_body(
        self, sync_client, sample_datapoints
    ):
        eval_id = uuid.uuid4()
        response = _response(500, "internal error")

        with patch.object(sync_client.evals._client, "post", return_value=response):
            with pytest.raises(ValueError, match=r"\[500\] internal error"):
                sync_client.evals.save_datapoints(eval_id, sample_datapoints)


class TestInitEval413:
    def test_init_reports_status_code_and_plain_text_body(self, sync_client):
        response = _response(413, SERVER_413_BODY)

        with patch.object(sync_client.evals._client, "post", return_value=response):
            with pytest.raises(ValueError) as excinfo:
                sync_client.evals.init(name="too-big")

        message = str(excinfo.value)
        assert "[413]" in message
        assert "HTTP_PAYLOAD_LIMIT" in message
