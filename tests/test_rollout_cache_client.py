"""
Tests for rollout cache client (sync HTTP client).
"""

import pytest
import httpx
from unittest.mock import Mock, patch

from lmnr.sdk.rollout.cache_client import CacheClient


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx.Client."""
    return Mock(spec=httpx.Client)


def test_cache_client_initialization():
    """Test cache client initialization."""
    client = CacheClient("http://localhost:1234", timeout=10.0)

    assert client.base_url == "http://localhost:1234"
    assert client.timeout == 10.0
    assert client._client is None
    assert client._path_to_count_cache is None
    assert client._overrides_cache is None


def test_cache_client_strips_trailing_slash():
    """Test that cache client strips trailing slash from URL."""
    client = CacheClient("http://localhost:1234/")
    assert client.base_url == "http://localhost:1234"


def test_cache_client_context_manager():
    """Test cache client as context manager."""
    with CacheClient("http://localhost:1234") as client:
        assert client._client is None  # Lazy initialization
    # Client should be closed after context


def test_get_cached_span_hit_with_metadata():
    """Test getting a cached span that exists."""
    client = CacheClient("http://localhost:1234")

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "span": {"name": "test", "output": "cached"},
        "path_to_count": {"root.llm": 3},
        "overrides": {"root.llm": {"system": "test"}},
    }

    with patch.object(client, "_get_client") as mock_get_client:
        mock_get_client.return_value.post.return_value = mock_response

        result = client.get_cached_span("root.llm", 0)

        assert result == {"name": "test", "output": "cached"}
        assert client._path_to_count_cache == {"root.llm": 3}
        assert client._overrides_cache == {"root.llm": {"system": "test"}}


def test_get_cached_span_miss():
    """Test getting a cached span that doesn't exist."""
    client = CacheClient("http://localhost:1234")

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "span": None,
        "path_to_count": {},
        "overrides": {},
    }

    with patch.object(client, "_get_client") as mock_get_client:
        mock_get_client.return_value.post.return_value = mock_response

        result = client.get_cached_span("nonexistent", 0)

        assert result is None


def test_get_cached_span_timeout():
    """Test timeout when fetching cached span."""
    client = CacheClient("http://localhost:1234", timeout=1.0)

    with patch.object(client, "_get_client") as mock_get_client:
        mock_get_client.return_value.post.side_effect = httpx.TimeoutException(
            "timeout"
        )

        result = client.get_cached_span("root.llm", 0)

        assert result is None


def test_get_path_to_count_cached():
    """Test that get_path_to_count() uses cache."""
    client = CacheClient("http://localhost:1234")

    # Set cache
    client._path_to_count_cache = {"root.llm": 5}

    # Should not make HTTP request
    result = client.get_path_to_count()

    assert result == {"root.llm": 5}


def test_get_path_to_count_fetch():
    """Test fetching path_to_count from server."""
    client = CacheClient("http://localhost:1234")

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"root.llm": 3}

    with patch.object(client, "_get_client") as mock_get_client:
        mock_get_client.return_value.get.return_value = mock_response

        result = client.get_path_to_count()

        assert result == {"root.llm": 3}
        assert client._path_to_count_cache == {"root.llm": 3}


def test_get_path_to_count_error():
    """Test error handling when fetching path_to_count."""
    client = CacheClient("http://localhost:1234")

    with patch.object(client, "_get_client") as mock_get_client:
        mock_get_client.return_value.get.side_effect = Exception("Network error")

        result = client.get_path_to_count()

        assert result == {}


def test_get_overrides_cached():
    """Test that get_overrides() uses cache."""
    client = CacheClient("http://localhost:1234")

    # Set cache
    client._overrides_cache = {"root.llm": {"system": "test"}}

    result = client.get_overrides()

    assert result == {"root.llm": {"system": "test"}}


def test_get_overrides_fetch():
    """Test fetching overrides from server."""
    client = CacheClient("http://localhost:1234")

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"root.llm": {"system": "test"}}

    with patch.object(client, "_get_client") as mock_get_client:
        mock_get_client.return_value.get.return_value = mock_response

        result = client.get_overrides()

        assert result == {"root.llm": {"system": "test"}}
        assert client._overrides_cache == {"root.llm": {"system": "test"}}


def test_should_use_cache_within_limit():
    """Test should_use_cache() when index is within cache limit."""
    client = CacheClient("http://localhost:1234")
    client._path_to_count_cache = {"root.llm": 3}

    assert client.should_use_cache("root.llm", 0) is True
    assert client.should_use_cache("root.llm", 1) is True
    assert client.should_use_cache("root.llm", 2) is True


def test_should_use_cache_exceeds_limit():
    """Test should_use_cache() when index exceeds cache limit."""
    client = CacheClient("http://localhost:1234")
    client._path_to_count_cache = {"root.llm": 3}

    assert client.should_use_cache("root.llm", 3) is False
    assert client.should_use_cache("root.llm", 4) is False


def test_should_use_cache_path_not_in_mapping():
    """Test should_use_cache() for path not in path_to_count."""
    client = CacheClient("http://localhost:1234")
    client._path_to_count_cache = {"root.llm": 3}

    assert client.should_use_cache("other.path", 0) is False


def test_invalidate_cache():
    """Test cache invalidation."""
    client = CacheClient("http://localhost:1234")

    # Set caches
    client._path_to_count_cache = {"root.llm": 3}
    client._overrides_cache = {"root.llm": {"system": "test"}}

    # Invalidate
    client.invalidate_cache()

    assert client._path_to_count_cache is None
    assert client._overrides_cache is None


def test_update_path_to_count_success():
    """Test updating path_to_count on server."""
    client = CacheClient("http://localhost:1234")

    mock_response = Mock()
    mock_response.status_code = 200

    with patch.object(client, "_get_client") as mock_get_client:
        mock_get_client.return_value.post.return_value = mock_response

        result = client.update_path_to_count({"root.llm": 5})

        assert result is True
        assert client._path_to_count_cache == {"root.llm": 5}


def test_update_path_to_count_failure():
    """Test error handling when updating path_to_count."""
    client = CacheClient("http://localhost:1234")

    mock_response = Mock()
    mock_response.status_code = 500

    with patch.object(client, "_get_client") as mock_get_client:
        mock_get_client.return_value.post.return_value = mock_response

        result = client.update_path_to_count({"root.llm": 5})

        assert result is False


def test_update_overrides_success():
    """Test updating overrides on server."""
    client = CacheClient("http://localhost:1234")

    mock_response = Mock()
    mock_response.status_code = 200

    with patch.object(client, "_get_client") as mock_get_client:
        mock_get_client.return_value.post.return_value = mock_response

        overrides = {"root.llm": {"system": "test"}}
        result = client.update_overrides(overrides)

        assert result is True
        assert client._overrides_cache == overrides


def test_update_spans_success():
    """Test bulk updating spans."""
    client = CacheClient("http://localhost:1234")

    mock_response = Mock()
    mock_response.status_code = 200

    with patch.object(client, "_get_client") as mock_get_client:
        mock_get_client.return_value.post.return_value = mock_response

        spans = {
            "root.llm:0": {"output": "cached"},
            "root.llm:1": {"output": "cached2"},
        }
        result = client.update_spans(spans)

        assert result is True


def test_cache_client_with_real_server_note():
    """
    Note: Integration test with real server would require async HTTP client.
    The CacheClient uses sync httpx.Client which can't be called from async context.
    Use the mock-based tests above for validation.
    """
    pass
