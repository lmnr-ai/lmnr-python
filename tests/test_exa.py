import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode


@pytest.mark.vcr
def test_exa_search_basic(span_exporter: InMemorySpanExporter):
    """Test basic search functionality with Exa."""
    from exa_py import Exa
    
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    exa = Exa(api_key="")
    
    result = exa.search("hottest AI startups", num_results=2)
    
    # Verify the API call worked
    assert result is not None
    assert hasattr(result, 'results')
    assert len(result.results) <= 2
    
    # Check span creation
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    
    span = spans[0]
    assert span.name == "exa.search"
    assert span.status.status_code == StatusCode.OK
    
    # Check service attributes
    attributes = span.attributes
    assert attributes["service.name"] == "exa"
    assert attributes["service.operation"] == "search"
    assert attributes["service.method"] == "search"
    
    # Check input/output
    assert "lmnr.span.input" in attributes
    assert "lmnr.span.output" in attributes
    
    # Check cost tracking
    assert "service.cost.amount" in attributes
    # Cost might be 0.0 initially and updated later, so just check it exists
    assert "service.cost.unit" in attributes
    
    # Check metadata
    assert "service.metadata" in attributes


@pytest.mark.vcr
def test_exa_search_and_contents(span_exporter: InMemorySpanExporter):
    """Test search with content extraction."""
    from exa_py import Exa
    
    exa = Exa(api_key="")
    
    result = exa.search_and_contents(
        "AI in healthcare",
        text=True,
        num_results=2
    )
    
    # Verify the API call worked
    assert result is not None
    assert hasattr(result, 'results')
    
    # Check span creation
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    
    span = spans[0]
    assert span.name == "exa.search_and_contents"
    assert span.status.status_code == StatusCode.OK
    
    # Check service attributes
    attributes = span.attributes
    assert attributes["service.name"] == "exa"
    assert attributes["service.operation"] == "search"
    assert attributes["service.method"] == "search_and_contents"
    
    # Check cost tracking (should be higher due to content)
    assert "service.cost.amount" in attributes
    # Cost might be 0.0 initially and updated later, so just check it exists


@pytest.mark.vcr
def test_exa_find_similar(span_exporter: InMemorySpanExporter):
    """Test find similar functionality."""
    from exa_py import Exa
    
    exa = Exa(api_key="")
    
    result = exa.find_similar(
        "https://www.techcrunch.com/",
        num_results=2
    )
    
    # Verify the API call worked
    assert result is not None
    assert hasattr(result, 'results')
    
    # Check span creation
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    
    span = spans[0]
    assert span.name == "exa.find_similar"
    assert span.status.status_code == StatusCode.OK
    
    # Check service attributes
    attributes = span.attributes
    assert attributes["service.name"] == "exa"
    assert attributes["service.operation"] == "search"
    assert attributes["service.method"] == "find_similar"


@pytest.mark.vcr
def test_exa_find_similar_and_contents(span_exporter: InMemorySpanExporter):
    """Test find similar with content extraction."""
    from exa_py import Exa
    
    exa = Exa(api_key="")
    
    result = exa.find_similar_and_contents(
        "https://www.techcrunch.com/",
        text=True,
        num_results=2
    )
    
    # Verify the API call worked
    assert result is not None
    assert hasattr(result, 'results')
    
    # Check span creation
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    
    span = spans[0]
    assert span.name == "exa.find_similar_and_contents"
    assert span.status.status_code == StatusCode.OK
    
    # Check service attributes
    attributes = span.attributes
    assert attributes["service.name"] == "exa"
    assert attributes["service.operation"] == "search"
    assert attributes["service.method"] == "find_similar_and_contents"


@pytest.mark.vcr
def test_exa_answer(span_exporter: InMemorySpanExporter):
    """Test answer generation functionality."""
    from exa_py import Exa
    
    exa = Exa(api_key="")
    
    result = exa.answer("What is the capital of France?")
    
    # Verify the API call worked
    assert result is not None
    
    # Check span creation
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    
    span = spans[0]
    assert span.name == "exa.answer"
    assert span.status.status_code == StatusCode.OK
    
    # Check service attributes
    attributes = span.attributes
    assert attributes["service.name"] == "exa"
    assert attributes["service.operation"] == "answer"
    assert attributes["service.method"] == "answer"
    
    # Check input/output
    assert "lmnr.span.input" in attributes
    assert "lmnr.span.output" in attributes


@pytest.mark.vcr
def test_exa_stream_answer(span_exporter: InMemorySpanExporter):
    """Test streaming answer generation."""
    from exa_py import Exa
    
    exa = Exa(api_key="")
    
    # Collect streaming response
    chunks = []
    for chunk in exa.stream_answer("What is machine learning?"):
        chunks.append(chunk)
    
    # Verify we got streaming chunks
    assert len(chunks) > 0
    
    # Check span creation
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    
    span = spans[0]
    assert span.name == "exa.stream_answer"
    assert span.status.status_code == StatusCode.OK
    
    # Check service attributes
    attributes = span.attributes
    assert attributes["service.name"] == "exa"
    assert attributes["service.operation"] == "answer"
    assert attributes["service.method"] == "stream_answer"
    
    # Check streaming-specific attributes
    assert "service.response.chunks_total" in attributes
    assert attributes["service.response.chunks_total"] == len(chunks)


def test_exa_error_handling(span_exporter: InMemorySpanExporter):
    """Test error handling and span status on API errors."""
    from exa_py import Exa
    
    # Use invalid API key to trigger error
    exa = Exa(api_key="invalid-key")
    
    with pytest.raises(Exception):
        exa.search("test query")
    
    # Check span creation and error status
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    
    span = spans[0]
    assert span.name == "exa.search"
    assert span.status.status_code == StatusCode.ERROR
    
    # Check error attributes
    attributes = span.attributes
    assert attributes["service.response.status"] == "error"
    assert "error.type" in attributes


def test_exa_cost_extraction():
    """Test cost extraction from response metadata."""
    from src.lmnr.opentelemetry_lib.opentelemetry.instrumentation.exa import _extract_response_metadata
    
    # Mock Exa response with cost data (using correct field name from API)
    mock_response = {
        "results": [{"url": "https://example.com", "title": "Test"}],
        "costDollars": {"total": 0.005, "search": {"neural": 0.005}}
    }
    
    metadata = _extract_response_metadata(mock_response)
    
    assert "actual_cost_total" in metadata
    assert metadata["actual_cost_total"] == 0.005
    assert "cost_dollars" in metadata
    assert metadata["cost_dollars"]["total"] == 0.005


def test_exa_metadata_extraction():
    """Test metadata extraction from various request types."""
    from src.lmnr.opentelemetry_lib.opentelemetry.instrumentation.exa import _extract_service_metadata
    
    # Test search metadata
    search_metadata = _extract_service_metadata(
        "search", 
        "search_and_contents",
        ("AI startups",), 
        {"num_results": 5, "text": True, "include_domains": ["techcrunch.com"]}
    )
    
    assert search_metadata["query"] == "AI startups"
    assert search_metadata["num_results"] == 5
    assert search_metadata["include_text"] is True
    assert search_metadata["include_domains"] == ["techcrunch.com"]
    
    # Test answer metadata
    answer_metadata = _extract_service_metadata(
        "answer",
        "answer", 
        ("What is AI?",),
        {"text": True}
    )
    
    assert answer_metadata["query"] == "What is AI?"
    assert answer_metadata["include_text"] is True
    
    # Test research metadata
    research_metadata = _extract_service_metadata(
        "research",
        "create_task",
        ("Research AI trends",),
        {"model": "exa-research-pro", "infer_schema": True}
    )
    
    assert research_metadata["instructions"] == "Research AI trends"
    assert research_metadata["model"] == "exa-research-pro"
    assert research_metadata["infer_schema"] is True


@pytest.mark.vcr
def test_exa_service_attributes_structure(span_exporter: InMemorySpanExporter):
    """Test that service attributes follow the unified specification by checking recorded spans."""
    # This test uses the recorded spans from other tests to verify attribute structure
    from exa_py import Exa
    
    exa = Exa(api_key="")
    
    # Use the recorded cassette for a simple search
    result = exa.search("test query", num_results=1)
    
    # Check span creation
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    
    span = spans[0]
    attributes = span.attributes
    
    # Check unified service specification compliance
    assert attributes["service.name"] == "exa"
    assert attributes["service.operation"] in ["search", "answer", "research"]
    assert "service.method" in attributes
    assert "service.metadata" in attributes
    assert "lmnr.span.input" in attributes
    assert "lmnr.span.output" in attributes
    
    # Verify cost was captured (may be 0.0 initially)
    assert "service.cost.amount" in attributes
    assert "service.cost.unit" in attributes
