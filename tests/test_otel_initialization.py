import os
import pytest
from unittest.mock import patch

from lmnr import Laminar
from lmnr.sdk.utils import get_otel_env_var, parse_otel_headers
from lmnr.opentelemetry_lib.tracing.exporter import LaminarSpanExporter


class TestOtelEnvVarUtils:
    """Test OTEL environment variable utility functions."""

    def test_get_otel_env_var_priority_order(self):
        """Test that OTEL env vars are checked in correct priority order."""
        with patch.dict(
            os.environ,
            {
                "OTEL_ENDPOINT": "http://otel-endpoint",
                "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otlp-endpoint",
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://traces-endpoint",
            },
            clear=False,
        ):
            assert get_otel_env_var("ENDPOINT") == "http://traces-endpoint"

    def test_get_otel_env_var_fallback(self):
        """Test fallback through priority order."""
        with patch.dict(
            os.environ,
            {
                "OTEL_ENDPOINT": "http://otel-endpoint",
                "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otlp-endpoint",
            },
            clear=False,
        ):
            assert get_otel_env_var("ENDPOINT") == "http://otlp-endpoint"

        with patch.dict(
            os.environ, {"OTEL_ENDPOINT": "http://otel-endpoint"}, clear=False
        ):
            assert get_otel_env_var("ENDPOINT") == "http://otel-endpoint"

    def test_get_otel_env_var_not_found(self):
        """Test when no OTEL env var is found."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_otel_env_var("NONEXISTENT") is None

    def test_parse_otel_headers_valid(self):
        """Test parsing valid OTEL headers string."""
        headers_str = "Authorization=Bearer%20token,Content-Type=application/json"
        expected = {"Authorization": "Bearer token", "Content-Type": "application/json"}
        assert parse_otel_headers(headers_str) == expected

    def test_parse_otel_headers_empty(self):
        """Test parsing empty headers string."""
        assert parse_otel_headers("") == {}
        assert parse_otel_headers(None) == {}

    def test_parse_otel_headers_invalid(self):
        """Test parsing invalid headers string."""
        headers_str = "invalid,no-equals-sign"
        assert parse_otel_headers(headers_str) == {}


class TestLaminarSpanExporterOtel:
    """Test LaminarSpanExporter with OTEL configuration."""

    def test_otel_config_initialization(self):
        """Test exporter initialization with OTEL config."""
        with patch.dict(
            os.environ,
            {
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://custom-endpoint:4318",
                "OTEL_EXPORTER_OTLP_TRACES_HEADERS": "Authorization=Bearer%20test-token",
                "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "http/protobuf",
            },
            clear=True,
        ):
            exporter = LaminarSpanExporter()

            assert exporter.endpoint == "http://custom-endpoint:4318"
            assert exporter.headers == {"Authorization": "Bearer test-token"}
            assert exporter.force_http is True

    def test_otel_config_defaults(self):
        """Test OTEL config with defaults."""
        with patch.dict(
            os.environ, {"OTEL_ENDPOINT": "http://simple-endpoint"}, clear=True
        ):
            exporter = LaminarSpanExporter()

            assert exporter.endpoint == "http://simple-endpoint"
            assert exporter.headers == {}
            assert exporter.timeout == 30  # default timeout
            assert exporter.force_http is False  # default protocol is grpc/protobuf

    def test_otel_config_grpc_protocol(self):
        """Test OTEL config with gRPC protocol."""
        with patch.dict(
            os.environ,
            {
                "OTEL_EXPORTER_OTLP_ENDPOINT": "http://grpc-endpoint:4317",
                "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc",
            },
            clear=True,
        ):
            exporter = LaminarSpanExporter()

            assert exporter.force_http is False

    def test_laminar_config_initialization(self):
        """Test traditional Laminar config still works."""
        exporter = LaminarSpanExporter(
            base_url="https://custom.lmnr.ai",
            port=9443,
            api_key="test-key",
            timeout_seconds=60,
            force_http=True,
        )

        assert exporter.endpoint == "https://custom.lmnr.ai:9443"
        assert exporter.headers == {"Authorization": "Bearer test-key"}
        assert exporter.timeout == 60
        assert exporter.force_http is True


class TestLaminarOtelInitialization:
    """Test Laminar initialization with OTEL configuration."""

    def setup_method(self):
        """Reset Laminar state before each test."""
        if hasattr(Laminar, "_Laminar__initialized"):
            Laminar._Laminar__initialized = False
        if hasattr(Laminar, "_Laminar__project_api_key"):
            Laminar._Laminar__project_api_key = None

    def test_traditional_initialization_still_works(self):
        """Test that traditional initialization path is not broken."""
        with patch("lmnr.opentelemetry_lib.TracerManager.init") as mock_init:
            Laminar.initialize(project_api_key="test-key")

            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["project_api_key"] == "test-key"
            assert call_kwargs["base_url"] is None

    def test_laminar_config_takes_precedence(self):
        """Test Laminar config takes precedence over OTEL config."""
        with patch.dict(
            os.environ,
            {
                "LMNR_PROJECT_API_KEY": "laminar-key",
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://custom-endpoint:4318",
            },
            clear=True,
        ):
            with patch("lmnr.opentelemetry_lib.TracerManager.init") as mock_init:
                Laminar.initialize()

                mock_init.assert_called_once()
                call_kwargs = mock_init.call_args[1]
                assert call_kwargs["base_url"] is None

    def test_base_url_param_prevents_otel_config(self):
        """Test that providing base_url parameter prevents OTEL config."""
        with patch.dict(
            os.environ,
            {"OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://custom-endpoint:4318"},
            clear=True,
        ):
            with patch("lmnr.opentelemetry_lib.TracerManager.init") as mock_init:
                # Need to provide API key when base_url is specified
                Laminar.initialize(
                    base_url="https://custom.lmnr.ai", project_api_key="test-key"
                )

                mock_init.assert_called_once()
                call_kwargs = mock_init.call_args[1]
                assert call_kwargs["base_url"] == "https://custom.lmnr.ai"

    def test_error_when_no_config_available(self):
        """Test error when neither Laminar nor OTEL config is available."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="Please initialize the Laminar object"
            ):
                Laminar.initialize()

    def test_otel_config_with_explicit_api_key_fails(self):
        """Test that providing API key prevents OTEL config even with OTEL vars."""
        with patch.dict(
            os.environ,
            {"OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://custom-endpoint:4318"},
            clear=True,
        ):
            with patch("lmnr.opentelemetry_lib.TracerManager.init") as mock_init:
                Laminar.initialize(project_api_key="explicit-key")

                mock_init.assert_called_once()
                call_kwargs = mock_init.call_args[1]
                assert call_kwargs["project_api_key"] == "explicit-key"
                assert call_kwargs["base_url"] is None


class TestOtelConfigIntegration:
    """Integration tests for OTEL configuration."""

    def setup_method(self):
        """Reset Laminar state before each test."""
        if hasattr(Laminar, "_Laminar__initialized"):
            Laminar._Laminar__initialized = False
        if hasattr(Laminar, "_Laminar__project_api_key"):
            Laminar._Laminar__project_api_key = None

    def test_end_to_end_otel_initialization(self):
        """Test end-to-end OTEL initialization flow."""
        with patch.dict(
            os.environ,
            {
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://otel-collector:4318/v1/traces",
                "OTEL_EXPORTER_OTLP_TRACES_HEADERS": "Authorization=Bearer%20otel-token,x-custom=value",
                "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT": "60s",
                "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "http/protobuf",
            },
            clear=True,
        ):
            with patch("lmnr.opentelemetry_lib.TracerManager.init") as mock_init:
                Laminar.initialize()

                # Verify TracerManager.init was called with OTEL config
                mock_init.assert_called_once()
                call_kwargs = mock_init.call_args[1]
                assert call_kwargs["base_url"] is None
