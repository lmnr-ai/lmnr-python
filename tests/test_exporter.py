"""Unit tests for LaminarSpanExporter URL normalization."""

import pytest
from unittest.mock import patch, MagicMock
from lmnr.opentelemetry_lib.tracing.exporter import LaminarSpanExporter
from lmnr.opentelemetry_lib.tracing.exporter import _normalize_http_endpoint


class TestHttpEndpointNormalization:
    """Test cases for HTTP endpoint URL normalization."""

    @pytest.fixture
    def mock_http_exporter(self):
        """Mock HTTPOTLPSpanExporter to avoid actual initialization."""
        with patch(
            "lmnr.opentelemetry_lib.tracing.exporter.HTTPOTLPSpanExporter"
        ) as mock:
            mock.return_value = MagicMock()
            yield mock

    @pytest.fixture
    def mock_grpc_exporter(self):
        """Mock OTLPSpanExporter to avoid actual initialization."""
        with patch("lmnr.opentelemetry_lib.tracing.exporter.OTLPSpanExporter") as mock:
            mock.return_value = MagicMock()
            yield mock

    def test_http_endpoint_without_path_adds_v1_traces(
        self, mock_http_exporter, mock_grpc_exporter
    ):
        """Test that endpoint without path gets /v1/traces added."""
        exporter = LaminarSpanExporter(
            base_url="http://localhost:8080",
            api_key="test-key",
            force_http=True,
        )
        assert exporter is not None

        mock_http_exporter.assert_called_once()
        call_kwargs = mock_http_exporter.call_args[1]
        assert call_kwargs["endpoint"] == "http://localhost:8080/v1/traces"

    def test_http_endpoint_with_trailing_slash_adds_v1_traces(
        self, mock_http_exporter, mock_grpc_exporter
    ):
        """Test that endpoint with trailing slash gets /v1/traces added."""
        exporter = LaminarSpanExporter(
            base_url="http://localhost:8080/",
            api_key="test-key",
            force_http=True,
        )
        assert exporter is not None

        mock_http_exporter.assert_called_once()
        call_kwargs = mock_http_exporter.call_args[1]
        assert call_kwargs["endpoint"] == "http://localhost:8080/v1/traces"

    def test_grpc_endpoint_not_normalized(self, mock_http_exporter, mock_grpc_exporter):
        """Test that gRPC endpoints are not normalized (only HTTP)."""
        exporter = LaminarSpanExporter(
            base_url="http://localhost:8443",
            api_key="test-key",
            force_http=False,
        )
        assert exporter is not None

        mock_grpc_exporter.assert_called_once()
        mock_http_exporter.assert_not_called()
        call_kwargs = mock_grpc_exporter.call_args[1]
        assert call_kwargs["endpoint"] == "http://localhost:8443"

    def test_https_endpoint_without_path_adds_v1_traces(
        self, mock_http_exporter, mock_grpc_exporter
    ):
        """Test that HTTPS endpoint without path gets /v1/traces added."""
        exporter = LaminarSpanExporter(
            base_url="https://api.example.com:443",
            api_key="test-key",
            force_http=True,
        )
        assert exporter is not None

        mock_http_exporter.assert_called_once()
        call_kwargs = mock_http_exporter.call_args[1]
        assert call_kwargs["endpoint"] == "https://api.example.com:443/v1/traces"

    def test_info_logging_when_path_added(self, mock_http_exporter, mock_grpc_exporter):
        """Test that info message is logged when path is added."""
        with patch("lmnr.opentelemetry_lib.tracing.exporter.logger") as mock_logger:
            exporter = LaminarSpanExporter(
                base_url="http://localhost:8080",
                api_key="test-key",
                force_http=True,
            )
            assert exporter is not None

            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            assert "No path found in HTTP endpoint URL" in call_args
            assert "Adding default path /v1/traces" in call_args

    def test_no_logging_when_path_exists(
        self, mock_http_exporter, mock_grpc_exporter, caplog
    ):
        """Test that no info message is logged when path already exists."""
        import logging

        caplog.set_level(logging.INFO)

        exporter = LaminarSpanExporter(
            base_url="http://localhost:8080/custom/path",
            api_key="test-key",
            force_http=True,
        )
        assert exporter is not None

        assert not any(
            "No path found in HTTP endpoint URL" in record.message
            for record in caplog.records
        )

    def test_normalize_http_endpoint_directly(self):
        """Test the _normalize_http_endpoint method directly."""
        with (
            patch(
                "lmnr.opentelemetry_lib.tracing.exporter.HTTPOTLPSpanExporter"
            ) as mock_http,
            patch(
                "lmnr.opentelemetry_lib.tracing.exporter.OTLPSpanExporter"
            ) as mock_grpc,
        ):
            mock_http.return_value = MagicMock()
            mock_grpc.return_value = MagicMock()

            exporter = LaminarSpanExporter(
                base_url="http://localhost:8080",
                api_key="test-key",
                force_http=True,
            )

            test_cases = [
                ("http://example.com", "http://example.com/v1/traces"),
                ("http://example.com/", "http://example.com/v1/traces"),
                ("https://example.com:443", "https://example.com:443/v1/traces"),
                ("http://example.com/path", "http://example.com/path"),
                ("http://example.com/v1/traces", "http://example.com/v1/traces"),
            ]

            for input_url, expected_url in test_cases:
                result = _normalize_http_endpoint(input_url, "/v1/traces")
                assert (
                    result == expected_url
                ), f"Expected {expected_url}, got {result} for input {input_url}"
