"""
Tests for rollout instrumentation wrapper.
"""

import pytest
from unittest.mock import Mock, patch

from lmnr.sdk.rollout.instrumentation import RolloutInstrumentationWrapper
from lmnr.sdk.rollout.cache_client import CacheClient


@pytest.fixture
def mock_cache_client():
    """Create a mock cache client."""
    client = Mock(spec=CacheClient)
    client.get_path_to_count.return_value = {}
    client.get_overrides.return_value = {}
    client.get_cached_span.return_value = None
    return client


def test_ensure_initialized_fetches_metadata_even_without_span():
    """
    Test the bug fix: _ensure_initialized() should fetch metadata
    even when no span exists at the dummy path.

    This was the bug - it was checking if cached_data exists before
    calling get_path_to_count()/get_overrides().
    """
    wrapper = RolloutInstrumentationWrapper()

    mock_client = Mock(spec=CacheClient)

    # Simulate no span exists, but metadata does
    mock_client.get_cached_span.return_value = None
    mock_client.get_path_to_count.return_value = {"root.llm": 4}
    mock_client.get_overrides.return_value = {"root.llm": {"system": "test"}}

    with patch.object(wrapper, "_get_cache_client", return_value=mock_client):
        wrapper._ensure_initialized()

        # Verify it still fetched metadata despite no span
        mock_client.get_cached_span.assert_called_once_with("", 0)
        mock_client.get_path_to_count.assert_called_once()
        mock_client.get_overrides.assert_called_once()

        # Verify metadata was stored
        assert wrapper._path_to_count == {"root.llm": 4}
        assert wrapper._overrides == {"root.llm": {"system": "test"}}


def test_ensure_initialized_only_runs_once():
    """Test that _ensure_initialized() only runs once."""
    wrapper = RolloutInstrumentationWrapper()

    mock_client = Mock(spec=CacheClient)
    mock_client.get_cached_span.return_value = None
    mock_client.get_path_to_count.return_value = {}
    mock_client.get_overrides.return_value = {}

    with patch.object(wrapper, "_get_cache_client", return_value=mock_client):
        wrapper._ensure_initialized()
        wrapper._ensure_initialized()
        wrapper._ensure_initialized()

        # Should only call cache client once
        assert mock_client.get_cached_span.call_count == 1


def test_should_use_cache_with_limit():
    """Test should_use_cache() with various indices."""
    wrapper = RolloutInstrumentationWrapper()
    wrapper._initialized = True
    wrapper._path_to_count = {"root.llm": 3}

    assert wrapper.should_use_cache("root.llm", 0) is True
    assert wrapper.should_use_cache("root.llm", 1) is True
    assert wrapper.should_use_cache("root.llm", 2) is True
    assert wrapper.should_use_cache("root.llm", 3) is False
    assert wrapper.should_use_cache("root.llm", 4) is False


def test_should_use_cache_path_not_configured():
    """Test should_use_cache() for path not in path_to_count."""
    wrapper = RolloutInstrumentationWrapper()
    wrapper._initialized = True
    wrapper._path_to_count = {"root.llm": 3}

    # Path not in mapping - should return False
    assert wrapper.should_use_cache("other.path", 0) is False


def test_should_use_cache_initializes_if_needed():
    """Test that should_use_cache() triggers initialization."""
    wrapper = RolloutInstrumentationWrapper()

    mock_client = Mock(spec=CacheClient)
    mock_client.get_cached_span.return_value = None
    mock_client.get_path_to_count.return_value = {"root.llm": 2}
    mock_client.get_overrides.return_value = {}

    with patch.object(wrapper, "_get_cache_client", return_value=mock_client):
        result = wrapper.should_use_cache("root.llm", 0)

        assert result is True
        assert wrapper._initialized is True


def test_get_cached_response_updates_metadata():
    """Test that get_cached_response() updates metadata on cache hit."""
    wrapper = RolloutInstrumentationWrapper()

    mock_client = Mock(spec=CacheClient)
    cached_span = {"name": "test", "output": "cached"}
    mock_client.get_cached_span.return_value = cached_span
    mock_client.get_path_to_count.return_value = {"root.llm": 5}
    mock_client.get_overrides.return_value = {"root.llm": {"system": "updated"}}

    with patch.object(wrapper, "_get_cache_client", return_value=mock_client):
        result = wrapper.get_cached_response("root.llm", 0)

        assert result == cached_span
        # Metadata should be updated
        assert wrapper._path_to_count == {"root.llm": 5}
        assert wrapper._overrides == {"root.llm": {"system": "updated"}}


def test_get_cached_response_miss():
    """Test get_cached_response() when span doesn't exist."""
    wrapper = RolloutInstrumentationWrapper()

    mock_client = Mock(spec=CacheClient)
    mock_client.get_cached_span.return_value = None
    mock_client.get_path_to_count.return_value = {}
    mock_client.get_overrides.return_value = {}

    with patch.object(wrapper, "_get_cache_client", return_value=mock_client):
        result = wrapper.get_cached_response("root.llm", 99)

        assert result is None


def test_get_overrides_for_specific_path():
    """Test getting overrides for a specific path."""
    wrapper = RolloutInstrumentationWrapper()
    wrapper._initialized = True
    wrapper._overrides = {
        "root.llm": {"system": "path-specific"},
        "root.other": {"system": "other-specific"},
    }

    overrides = wrapper.get_overrides("root.llm")

    assert overrides == {"system": "path-specific"}


def test_get_overrides_path_not_found():
    """Test getting overrides for non-existent path."""
    wrapper = RolloutInstrumentationWrapper()
    wrapper._initialized = True
    wrapper._overrides = {"root.llm": {"system": "test"}}

    overrides = wrapper.get_overrides("nonexistent.path")

    assert overrides == {}


def test_get_overrides_no_path():
    """Test getting all overrides when no path specified."""
    wrapper = RolloutInstrumentationWrapper()
    wrapper._initialized = True
    wrapper._overrides = {
        "root.llm": {"system": "test"},
        "root.other": {"system": "other"},
    }

    overrides = wrapper.get_overrides()

    assert overrides == {
        "root.llm": {"system": "test"},
        "root.other": {"system": "other"},
    }


def test_get_current_index_for_path_increments():
    """Test that get_current_index_for_path() increments counter."""
    wrapper = RolloutInstrumentationWrapper()

    # First call
    index = wrapper.get_current_index_for_path("root.llm")
    assert index == 0

    # Second call
    index = wrapper.get_current_index_for_path("root.llm")
    assert index == 1

    # Third call
    index = wrapper.get_current_index_for_path("root.llm")
    assert index == 2


def test_get_current_index_for_different_paths():
    """Test that different paths have independent counters."""
    wrapper = RolloutInstrumentationWrapper()

    # Path 1
    assert wrapper.get_current_index_for_path("root.llm") == 0
    assert wrapper.get_current_index_for_path("root.llm") == 1

    # Path 2 - starts at 0
    assert wrapper.get_current_index_for_path("root.other") == 0
    assert wrapper.get_current_index_for_path("root.other") == 1

    # Path 1 again - continues from where it left off
    assert wrapper.get_current_index_for_path("root.llm") == 2


def test_get_span_path_with_laminar_context():
    """Test get_span_path() with proper Laminar context."""
    wrapper = RolloutInstrumentationWrapper()

    # Mock Laminar.get_current_span()
    from lmnr.sdk.laminar import Laminar

    mock_span = Mock()
    mock_context = Mock()
    mock_context.span_path = ["root", "middle", "leaf"]
    mock_span.get_laminar_span_context.return_value = mock_context

    with patch.object(Laminar, "get_current_span", return_value=mock_span):
        path = wrapper.get_span_path()

        assert path == "root.middle.leaf"


def test_get_span_path_no_current_span():
    """Test get_span_path() when no span is active."""
    wrapper = RolloutInstrumentationWrapper()

    from lmnr.sdk.laminar import Laminar

    with patch.object(Laminar, "get_current_span", return_value=None):
        path = wrapper.get_span_path()

        assert path is None


def test_get_span_path_empty_path():
    """Test get_span_path() when span_path is empty."""
    wrapper = RolloutInstrumentationWrapper()

    from lmnr.sdk.laminar import Laminar

    mock_span = Mock()
    mock_context = Mock()
    mock_context.span_path = []
    mock_span.get_laminar_span_context.return_value = mock_context

    with patch.object(Laminar, "get_current_span", return_value=mock_span):
        path = wrapper.get_span_path()

        assert path is None


def test_should_use_rollout_checks_mode():
    """Test should_use_rollout() checks rollout mode."""
    wrapper = RolloutInstrumentationWrapper()

    # Patch where it's used (in the instrumentation module)
    with patch("lmnr.sdk.rollout.instrumentation.is_rollout_mode", return_value=True):
        assert wrapper.should_use_rollout() is True

    with patch("lmnr.sdk.rollout.instrumentation.is_rollout_mode", return_value=False):
        assert wrapper.should_use_rollout() is False
