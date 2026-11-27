"""Observe-level mocks for evals: when `LMNR_MOCK_URL` is set (an eval trial
with mocks), an observed function whose key is mocked is answered by the
trial's mock server instead of running. Protocol: `spec/lmnr-mock/`."""

from .errors import MOCK_FAILURE, MockedError
from .intercept import MockTarget, alookup, lookup, mocks_enabled

__all__ = [
    "MOCK_FAILURE",
    "MockTarget",
    "MockedError",
    "alookup",
    "lookup",
    "mocks_enabled",
]
