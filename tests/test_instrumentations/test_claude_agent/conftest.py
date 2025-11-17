import pytest


@pytest.fixture(autouse=True)
def cleanup_claude_proxy():
    """Clean up Claude proxy server before and after each test."""
    # Clean up before test
    _cleanup_proxy()
    yield
    # Clean up after test
    _cleanup_proxy()


def start_claude_proxy():
    """Start the Claude proxy server if it's not running."""
    try:
        from lmnr_claude_code_proxy import run_server

        run_server()
    except Exception:
        # Ignore errors if the proxy couldn't be started
        pass


def _cleanup_proxy():
    """Stop the Claude proxy server if it's running."""
    try:
        from lmnr_claude_code_proxy import stop_server

        stop_server()
    except Exception:
        # Ignore errors if the proxy wasn't running or module not available
        pass
