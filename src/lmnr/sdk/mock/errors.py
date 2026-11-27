from __future__ import annotations

from typing_extensions import override

#: `MockedError.type` of every failure that isn't a handler's own error: an
#: unreachable server, a failed call, a function that can't be mocked.
MOCK_FAILURE = "MockFailure"


class MockedError(Exception):
    """Raised by an observed function whose call the eval's mock server
    answered with an error (`mock.Error(type, message)` in the handler), or
    whose mocked call failed (`type == "MockFailure"`)."""

    type: str
    message: str

    def __init__(self, type: str, message: str):
        super().__init__(f"{type}: {message}")
        self.type = type
        self.message = message

    @override
    def __reduce__(self) -> tuple[type[MockedError], tuple[str, str]]:
        return (MockedError, (self.type, self.message))
