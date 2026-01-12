"""
Tests for rollout cache server (aiohttp server).
"""

import asyncio
import pytest
import httpx

from lmnr.sdk.rollout.cache_server import CacheServer


@pytest.mark.asyncio
async def test_cache_server_lifecycle():
    """Test cache server start and stop."""
    server = CacheServer(port=0)
    
    # Start server
    port = await server.start()
    assert port > 0
    assert server.actual_port == port
    assert server.get_url() == f"http://127.0.0.1:{port}"
    
    # Verify health endpoint
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{server.get_url()}/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    
    # Stop server
    await server.stop()


@pytest.mark.asyncio
async def test_cache_server_get_url_before_start():
    """Test that get_url() raises error before server is started."""
    server = CacheServer(port=0)
    
    with pytest.raises(RuntimeError, match="Server not started yet"):
        server.get_url()


@pytest.mark.asyncio
async def test_update_and_get_path_to_count():
    """Test updating and retrieving path_to_count mapping."""
    server = CacheServer(port=0)
    await server.start()
    
    try:
        path_to_count = {"root.llm_call": 2, "root.other": 1}
        
        async with httpx.AsyncClient() as client:
            # Update path_to_count
            response = await client.post(
                f"{server.get_url()}/path_to_count",
                json=path_to_count,
            )
            assert response.status_code == 200
            result = response.json()
            assert result["status"] == "ok"
            assert result["count"] == 2
            
            # Get path_to_count
            response = await client.get(f"{server.get_url()}/path_to_count")
            assert response.status_code == 200
            assert response.json() == path_to_count
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_update_and_get_overrides():
    """Test updating and retrieving overrides."""
    server = CacheServer(port=0)
    await server.start()
    
    try:
        overrides = {
            "root.llm": {"system": "test prompt", "tools": []},
            "root.other": {"system": "another prompt"},
        }
        
        async with httpx.AsyncClient() as client:
            # Update overrides
            response = await client.post(
                f"{server.get_url()}/overrides",
                json=overrides,
            )
            assert response.status_code == 200
            result = response.json()
            assert result["status"] == "ok"
            
            # Get overrides
            response = await client.get(f"{server.get_url()}/overrides")
            assert response.status_code == 200
            assert response.json() == overrides
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_update_and_get_cached_spans():
    """Test bulk updating spans and fetching individual spans."""
    server = CacheServer(port=0)
    await server.start()
    
    try:
        # Update multiple spans
        spans = {
            "root.llm:0": {
                "name": "llm_call",
                "input": {"prompt": "test"},
                "output": "response 1",
                "attributes": {"model": "gpt-4"},
            },
            "root.llm:1": {
                "name": "llm_call",
                "input": {"prompt": "test2"},
                "output": "response 2",
                "attributes": {"model": "gpt-4"},
            },
        }
        
        async with httpx.AsyncClient() as client:
            # Bulk update spans
            response = await client.post(
                f"{server.get_url()}/spans",
                json=spans,
            )
            assert response.status_code == 200
            result = response.json()
            assert result["status"] == "ok"
            assert result["count"] == 2
            
            # Fetch individual span
            response = await client.post(
                f"{server.get_url()}/cached",
                json={"path": "root.llm", "index": 0},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["span"] == spans["root.llm:0"]
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_get_cached_span_miss():
    """Test fetching a span that doesn't exist."""
    server = CacheServer(port=0)
    await server.start()
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{server.get_url()}/cached",
                json={"path": "nonexistent", "index": 0},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["span"] is None
            assert "path_to_count" in data
            assert "overrides" in data
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_cached_endpoint_returns_metadata():
    """Test that /cached endpoint always returns metadata."""
    server = CacheServer(port=0)
    await server.start()
    
    try:
        path_to_count = {"root.llm": 3}
        overrides = {"root.llm": {"system": "test"}}
        
        async with httpx.AsyncClient() as client:
            # Set metadata
            await client.post(f"{server.get_url()}/path_to_count", json=path_to_count)
            await client.post(f"{server.get_url()}/overrides", json=overrides)
            
            # Fetch non-existent span - should still return metadata
            response = await client.post(
                f"{server.get_url()}/cached",
                json={"path": "nonexistent", "index": 99},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["span"] is None
            assert data["path_to_count"] == path_to_count
            assert data["overrides"] == overrides
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_concurrent_access():
    """Test thread safety with concurrent requests."""
    server = CacheServer(port=0)
    await server.start()
    
    try:
        async def update_path_to_count(client, path, count):
            await client.post(
                f"{server.get_url()}/path_to_count",
                json={path: count},
            )
        
        async with httpx.AsyncClient() as client:
            # Make concurrent updates
            tasks = [
                update_path_to_count(client, f"path{i}", i)
                for i in range(10)
            ]
            await asyncio.gather(*tasks)
            
            # Final path_to_count should be the last update (due to lock)
            response = await client.get(f"{server.get_url()}/path_to_count")
            assert response.status_code == 200
            # Should have exactly one path (last update wins)
            data = response.json()
            assert len(data) == 1
    finally:
        await server.stop()
