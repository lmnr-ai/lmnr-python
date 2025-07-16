import os
import pytest
from unittest.mock import patch

from lmnr.sdk.laminar import Laminar
from lmnr.sdk.evaluations import Evaluation


@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Reset Laminar state before each test."""
    # Reset the initialized state
    Laminar._Laminar__initialized = False
    Laminar._Laminar__base_http_url = None
    Laminar._Laminar__project_api_key = None
    yield
    # Clean up after test
    Laminar._Laminar__initialized = False
    Laminar._Laminar__base_http_url = None
    Laminar._Laminar__project_api_key = None


def test_laminar_initialize_url_parsing():
    """Test various combinations of URL parameters for Laminar.initialize."""
    test_cases = [
        # Test case format: {
        #     'name': 'description',
        #     'params': {'base_url': ..., 'base_http_url': ..., 'http_port': ..., 'grpc_port': ...},
        #     'env_vars': {'LMNR_BASE_URL': ...},
        #     'expected': {'base_url': ..., 'http_port': ..., 'port': ..., 'base_http_url': ...}
        # }
        # Default case - no params, no env vars
        {
            "name": "default_case",
            "params": {},
            "env_vars": {},
            "expected": {
                "base_url": "https://api.lmnr.ai",
                "http_port": 443,
                "port": 8443,
                "base_http_url": "https://api.lmnr.ai:443",
            },
        },
        # Base URL with trailing slash
        {
            "name": "base_url_with_trailing_slash",
            "params": {"base_url": "https://example.com/"},
            "env_vars": {},
            "expected": {
                "base_url": "https://example.com",
                "http_port": 443,
                "port": 8443,
                "base_http_url": "https://example.com:443",
            },
        },
        # Base URL without https prefix
        {
            "name": "base_url_without_https",
            "params": {"base_url": "example.com"},
            "env_vars": {},
            "expected": {
                "base_url": "https://example.com",
                "http_port": 443,
                "port": 8443,
                "base_http_url": "https://example.com:443",
            },
        },
        # Base URL with port (should be stripped)
        {
            "name": "base_url_with_port",
            "params": {"base_url": "https://example.com:8080"},
            "env_vars": {},
            "expected": {
                "base_url": "https://example.com",
                "http_port": 443,
                "port": 8443,
                "base_http_url": "https://example.com:443",
            },
        },
        # Custom HTTP port
        {
            "name": "custom_http_port",
            "params": {"http_port": 8080},
            "env_vars": {},
            "expected": {
                "base_url": "https://api.lmnr.ai",
                "http_port": 8080,
                "port": 8443,
                "base_http_url": "https://api.lmnr.ai:8080",
            },
        },
        # Custom GRPC port
        {
            "name": "custom_grpc_port",
            "params": {"grpc_port": 9090},
            "env_vars": {},
            "expected": {
                "base_url": "https://api.lmnr.ai",
                "http_port": 443,
                "port": 9090,
                "base_http_url": "https://api.lmnr.ai:443",
            },
        },
        # Base HTTP URL different from base URL
        {
            "name": "different_base_http_url",
            "params": {
                "base_url": "https://api.example.com",
                "base_http_url": "https://http.example.com",
            },
            "env_vars": {},
            "expected": {
                "base_url": "https://api.example.com",
                "http_port": 443,
                "port": 8443,
                "base_http_url": "https://http.example.com:443",
            },
        },
        # Base HTTP URL with port, no http_port param (should extract port)
        {
            "name": "base_http_url_with_port_no_param",
            "params": {"base_http_url": "https://http.example.com:8080"},
            "env_vars": {},
            "expected": {
                "base_url": "https://api.lmnr.ai",
                "http_port": 8080,
                "port": 8443,
                "base_http_url": "https://http.example.com:8080",
            },
        },
        # Base HTTP URL with port, but http_port param provided (should use param)
        {
            "name": "base_http_url_with_port_and_param",
            "params": {
                "base_http_url": "https://http.example.com:8080",
                "http_port": 9090,
            },
            "env_vars": {},
            "expected": {
                "base_url": "https://api.lmnr.ai",
                "http_port": 9090,
                "port": 8443,
                "base_http_url": "https://http.example.com:9090",
            },
        },
        # Environment variable for base URL
        {
            "name": "env_var_base_url",
            "params": {},
            "env_vars": {"LMNR_BASE_URL": "https://env.example.com"},
            "expected": {
                "base_url": "https://env.example.com",
                "http_port": 443,
                "port": 8443,
                "base_http_url": "https://env.example.com:443",
            },
        },
        # Environment variable with port
        {
            "name": "env_var_with_port",
            "params": {},
            "env_vars": {"LMNR_BASE_URL": "https://env.example.com:8080"},
            "expected": {
                "base_url": "https://env.example.com",
                "http_port": 443,
                "port": 8443,
                "base_http_url": "https://env.example.com:443",
            },
        },
        # Base URL param overrides env var
        {
            "name": "param_overrides_env",
            "params": {"base_url": "https://param.example.com"},
            "env_vars": {"LMNR_BASE_URL": "https://env.example.com"},
            "expected": {
                "base_url": "https://param.example.com",
                "http_port": 443,
                "port": 8443,
                "base_http_url": "https://param.example.com:443",
            },
        },
        # Base HTTP URL without https prefix
        {
            "name": "base_http_url_without_https",
            "params": {"base_http_url": "http.example.com"},
            "env_vars": {},
            "expected": {
                "base_url": "https://api.lmnr.ai",
                "http_port": 443,
                "port": 8443,
                "base_http_url": "https://http.example.com:443",
            },
        },
    ]

    for test_case in test_cases:
        with patch.dict(os.environ, test_case["env_vars"], clear=True):
            with patch("lmnr.opentelemetry_lib.TracerManager.init") as mock_tracer_init:
                # Reset state for each test
                Laminar._Laminar__initialized = False
                Laminar._Laminar__base_http_url = None
                Laminar._Laminar__project_api_key = None

                # Add required project_api_key
                params = test_case["params"].copy()
                params["project_api_key"] = "test-key"

                # Call initialize
                Laminar.initialize(**params)

                # Verify TracerManager.init was called with expected args
                mock_tracer_init.assert_called_once()
                call_args = mock_tracer_init.call_args[1]

                assert (
                    call_args["base_url"] == test_case["expected"]["base_url"]
                ), f"Test '{test_case['name']}': base_url mismatch. Expected: {test_case['expected']['base_url']}, Got: {call_args['base_url']}"
                assert (
                    call_args["http_port"] == test_case["expected"]["http_port"]
                ), f"Test '{test_case['name']}': http_port mismatch. Expected: {test_case['expected']['http_port']}, Got: {call_args['http_port']}"
                assert (
                    call_args["port"] == test_case["expected"]["port"]
                ), f"Test '{test_case['name']}': port mismatch. Expected: {test_case['expected']['port']}, Got: {call_args['port']}"
                assert (
                    call_args["project_api_key"] == "test-key"
                ), f"Test '{test_case['name']}': project_api_key mismatch"

                # Verify internal state
                assert (
                    Laminar._Laminar__base_http_url
                    == test_case["expected"]["base_http_url"]
                ), f"Test '{test_case['name']}': __base_http_url mismatch. Expected: {test_case['expected']['base_http_url']}, Got: {Laminar._Laminar__base_http_url}"


def test_evaluation_initialize_url_parsing():
    """Test various combinations of URL parameters for Evaluation.__init__."""
    test_cases = [
        # Default case - no params, no env vars
        {
            "name": "default_case",
            "params": {},
            "env_vars": {},
            "expected": {
                "base_url": "https://api.lmnr.ai",
                "base_http_url": "https://api.lmnr.ai:443",
                "http_port": None,
                "grpc_port": None,
            },
        },
        # Custom base URL
        {
            "name": "custom_base_url",
            "params": {"base_url": "https://example.com"},
            "env_vars": {},
            "expected": {
                "base_url": "https://example.com",
                "base_http_url": "https://example.com:443",
                "http_port": None,
                "grpc_port": None,
            },
        },
        # Custom base HTTP URL
        {
            "name": "custom_base_http_url",
            "params": {"base_http_url": "https://http.example.com"},
            "env_vars": {},
            "expected": {
                "base_url": "https://api.lmnr.ai",
                "base_http_url": "https://http.example.com:443",
                "http_port": None,
                "grpc_port": None,
            },
        },
        # Custom HTTP port
        {
            "name": "custom_http_port",
            "params": {"http_port": 8080},
            "env_vars": {},
            "expected": {
                "base_url": "https://api.lmnr.ai",
                "base_http_url": "https://api.lmnr.ai:8080",
                "http_port": 8080,
                "grpc_port": None,
            },
        },
        # Custom GRPC port
        {
            "name": "custom_grpc_port",
            "params": {"grpc_port": 9090},
            "env_vars": {},
            "expected": {
                "base_url": "https://api.lmnr.ai",
                "base_http_url": "https://api.lmnr.ai:443",
                "http_port": None,
                "grpc_port": 9090,
            },
        },
        # Base HTTP URL overrides base URL
        {
            "name": "base_http_url_overrides_base_url",
            "params": {
                "base_url": "https://api.example.com",
                "base_http_url": "https://http.example.com",
            },
            "env_vars": {},
            "expected": {
                "base_url": "https://api.example.com",
                "base_http_url": "https://http.example.com:443",
                "http_port": None,
                "grpc_port": None,
            },
        },
        # Both base HTTP URL and custom HTTP port
        {
            "name": "base_http_url_with_custom_port",
            "params": {
                "base_http_url": "https://http.example.com",
                "http_port": 8080,
            },
            "env_vars": {},
            "expected": {
                "base_url": "https://api.lmnr.ai",
                "base_http_url": "https://http.example.com:8080",
                "http_port": 8080,
                "grpc_port": None,
            },
        },
        # Environment variable for base URL
        {
            "name": "env_var_base_url",
            "params": {},
            "env_vars": {"LMNR_BASE_URL": "https://env.example.com"},
            "expected": {
                "base_url": "https://env.example.com",
                "base_http_url": "https://env.example.com:443",
                "http_port": None,
                "grpc_port": None,
            },
        },
        # Base URL param overrides env var
        {
            "name": "param_overrides_env",
            "params": {"base_url": "https://param.example.com"},
            "env_vars": {"LMNR_BASE_URL": "https://env.example.com"},
            "expected": {
                "base_url": "https://param.example.com",
                "base_http_url": "https://param.example.com:443",
                "http_port": None,
                "grpc_port": None,
            },
        },
        # All parameters set
        {
            "name": "all_params_set",
            "params": {
                "base_url": "https://api.example.com",
                "base_http_url": "https://http.example.com",
                "http_port": 8080,
                "grpc_port": 9090,
            },
            "env_vars": {},
            "expected": {
                "base_url": "https://api.example.com",
                "base_http_url": "https://http.example.com:8080",
                "http_port": 8080,
                "grpc_port": 9090,
            },
        },
    ]

    for test_case in test_cases:
        with patch.dict(os.environ, test_case["env_vars"], clear=True):
            with patch("lmnr.sdk.laminar.Laminar.initialize") as mock_laminar_init:
                with patch(
                    "lmnr.sdk.laminar.Laminar.is_initialized", return_value=False
                ):
                    # Reset Laminar state
                    Laminar._Laminar__initialized = False

                    # Create minimal evaluation params
                    params = test_case["params"].copy()
                    params.update(
                        {
                            "data": [{"data": "test", "target": "test"}],
                            "executor": lambda x: x,
                            "evaluators": {"test": lambda x, y: 1.0},
                            "project_api_key": "test-key",
                        }
                    )

                    # Create Evaluation instance
                    evaluation = Evaluation(**params)

                    # Verify the base_http_url was set correctly
                    assert (
                        evaluation.base_http_url
                        == test_case["expected"]["base_http_url"]
                    ), f"Test '{test_case['name']}': base_http_url mismatch. Expected: {test_case['expected']['base_http_url']}, Got: {evaluation.base_http_url}"

                    # Verify Laminar.initialize was called with expected args
                    mock_laminar_init.assert_called_once()
                    call_args = mock_laminar_init.call_args[1]

                    assert (
                        call_args["base_url"] == test_case["expected"]["base_url"]
                    ), f"Test '{test_case['name']}': base_url mismatch. Expected: {test_case['expected']['base_url']}, Got: {call_args['base_url']}"
                    assert (
                        call_args["base_http_url"]
                        == test_case["expected"]["base_http_url"]
                    ), f"Test '{test_case['name']}': base_http_url mismatch. Expected: {test_case['expected']['base_http_url']}, Got: {call_args['base_http_url']}"
                    assert (
                        call_args["http_port"] == test_case["expected"]["http_port"]
                    ), f"Test '{test_case['name']}': http_port mismatch. Expected: {test_case['expected']['http_port']}, Got: {call_args['http_port']}"
                    assert (
                        call_args["grpc_port"] == test_case["expected"]["grpc_port"]
                    ), f"Test '{test_case['name']}': grpc_port mismatch. Expected: {test_case['expected']['grpc_port']}, Got: {call_args['grpc_port']}"
                    assert (
                        call_args["project_api_key"] == "test-key"
                    ), f"Test '{test_case['name']}': project_api_key mismatch"
