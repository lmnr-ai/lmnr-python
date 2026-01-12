"""
Tests for rollout parameter parsing in CLI.
"""

import inspect
import pytest

from lmnr.cli.dev import DevCommandHandler


def test_parse_function_with_no_params():
    """Test parsing function with no parameters."""
    def test_func():
        return "result"
    
    handler = DevCommandHandler(
        file_path="/tmp/test.py",
        function_name="test",
        project_api_key="key",
        base_url="http://test",
    )
    
    params = handler._parse_function_params(test_func)
    
    assert params == []


def test_parse_function_with_required_params():
    """Test parsing function with required parameters."""
    def test_func(arg1, arg2, arg3):
        pass
    
    handler = DevCommandHandler(
        file_path="/tmp/test.py",
        function_name="test",
        project_api_key="key",
        base_url="http://test",
    )
    
    params = handler._parse_function_params(test_func)
    
    assert len(params) == 3
    assert params[0]["name"] == "arg1"
    assert params[0]["required"] is True
    assert "default" not in params[0]
    
    assert params[1]["name"] == "arg2"
    assert params[1]["required"] is True
    
    assert params[2]["name"] == "arg3"
    assert params[2]["required"] is True


def test_parse_function_with_optional_params():
    """Test parsing function with default values."""
    def test_func(required_arg, optional_arg="default", optional_int=42):
        pass
    
    handler = DevCommandHandler(
        file_path="/tmp/test.py",
        function_name="test",
        project_api_key="key",
        base_url="http://test",
    )
    
    params = handler._parse_function_params(test_func)
    
    assert len(params) == 3
    
    # Required param
    assert params[0]["name"] == "required_arg"
    assert params[0]["required"] is True
    assert "default" not in params[0]
    
    # Optional string param
    assert params[1]["name"] == "optional_arg"
    assert params[1]["required"] is False
    assert params[1]["default"] == "'default'"
    
    # Optional int param
    assert params[2]["name"] == "optional_int"
    assert params[2]["required"] is False
    assert params[2]["default"] == "42"


def test_parse_function_with_type_annotations():
    """Test parsing function with type annotations."""
    def test_func(name: str, count: int, data: dict):
        pass
    
    handler = DevCommandHandler(
        file_path="/tmp/test.py",
        function_name="test",
        project_api_key="key",
        base_url="http://test",
    )
    
    params = handler._parse_function_params(test_func)
    
    assert len(params) == 3
    assert params[0]["name"] == "name"
    assert params[0]["type"] == "<class 'str'>"
    
    assert params[1]["name"] == "count"
    assert params[1]["type"] == "<class 'int'>"
    
    assert params[2]["name"] == "data"
    assert params[2]["type"] == "<class 'dict'>"


def test_parse_method_skips_self():
    """Test that self parameter is skipped for methods."""
    class TestClass:
        def test_method(self, arg1, arg2):
            pass
    
    handler = DevCommandHandler(
        file_path="/tmp/test.py",
        function_name="test",
        project_api_key="key",
        base_url="http://test",
    )
    
    params = handler._parse_function_params(TestClass.test_method)
    
    assert len(params) == 2
    assert params[0]["name"] == "arg1"
    assert params[1]["name"] == "arg2"


def test_parse_classmethod_skips_cls():
    """Test that cls parameter is skipped for classmethods."""
    class TestClass:
        @classmethod
        def test_method(cls, arg1):
            pass
    
    handler = DevCommandHandler(
        file_path="/tmp/test.py",
        function_name="test",
        project_api_key="key",
        base_url="http://test",
    )
    
    params = handler._parse_function_params(TestClass.test_method)
    
    assert len(params) == 1
    assert params[0]["name"] == "arg1"


def test_parse_function_with_mixed_params():
    """Test parsing function with mix of required, optional, and typed params."""
    def test_func(
        required: str,
        optional_typed: int = 10,
        optional_untyped="default",
        another_required=None,
    ):
        pass
    
    handler = DevCommandHandler(
        file_path="/tmp/test.py",
        function_name="test",
        project_api_key="key",
        base_url="http://test",
    )
    
    params = handler._parse_function_params(test_func)
    
    assert len(params) == 4
    
    # Required with type
    assert params[0]["name"] == "required"
    assert params[0]["required"] is True
    assert params[0]["type"] == "<class 'str'>"
    
    # Optional with type
    assert params[1]["name"] == "optional_typed"
    assert params[1]["required"] is False
    assert params[1]["default"] == "10"
    assert params[1]["type"] == "<class 'int'>"
    
    # Optional without type
    assert params[2]["name"] == "optional_untyped"
    assert params[2]["required"] is False
    assert params[2]["default"] == "'default'"
    
    # Required (None is a default value)
    assert params[3]["name"] == "another_required"
    assert params[3]["required"] is False
    assert params[3]["default"] == "None"


def test_parse_function_with_kwargs():
    """Test parsing function with **kwargs."""
    def test_func(arg1, **kwargs):
        pass
    
    handler = DevCommandHandler(
        file_path="/tmp/test.py",
        function_name="test",
        project_api_key="key",
        base_url="http://test",
    )
    
    params = handler._parse_function_params(test_func)
    
    # Should include both arg1 and kwargs
    assert len(params) == 2
    assert params[0]["name"] == "arg1"
    assert params[1]["name"] == "kwargs"


def test_parse_function_error_handling():
    """Test error handling when parsing fails."""
    handler = DevCommandHandler(
        file_path="/tmp/test.py",
        function_name="test",
        project_api_key="key",
        base_url="http://test",
    )
    
    # Non-callable object
    params = handler._parse_function_params("not a function")
    
    # Should return empty list on error
    assert params == []
