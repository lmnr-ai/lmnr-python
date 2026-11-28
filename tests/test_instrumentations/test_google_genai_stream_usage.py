from collections import defaultdict

from google.genai import types

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.google_genai.utils import (
    process_stream_chunk,
)


def _chunk(candidates_tokens, total_tokens, thoughts_tokens=None, text="x"):
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(content=types.Content(role="model", parts=[types.Part(text=text)]))
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=7,
            candidates_token_count=candidates_tokens,
            thoughts_token_count=thoughts_tokens,
            total_token_count=total_tokens,
        ),
        model_version="gemini-test",
    )


def test_stream_usage_metadata_is_not_summed_across_cumulative_chunks():
    # usage_metadata in streaming chunks is cumulative (2 -> 21 -> 68 -> 75),
    # so the final usage must be the last reported value, not the sum.
    chunks = [
        _chunk(2, 9),
        _chunk(21, 28),
        _chunk(68, 75),
        _chunk(75, 82),
    ]
    aggregated = defaultdict(int)
    final_parts = []
    role, model_version = "model", None
    for chunk in chunks:
        result = process_stream_chunk(chunk, role, model_version, aggregated, final_parts)
        role = result["role"]
        model_version = result["model_version"]

    assert aggregated["prompt_token_count"] == 7
    assert aggregated["candidates_token_count"] == 75
    assert aggregated["total_token_count"] == 82


def test_stream_usage_metadata_keeps_thoughts_token_count():
    chunks = [
        _chunk(2, 19, thoughts_tokens=10),
        _chunk(20, 47, thoughts_tokens=20),
    ]
    aggregated = defaultdict(int)
    final_parts = []
    role, model_version = "model", None
    for chunk in chunks:
        result = process_stream_chunk(chunk, role, model_version, aggregated, final_parts)
        role = result["role"]
        model_version = result["model_version"]

    assert aggregated.get("thoughts_token_count") == 20
