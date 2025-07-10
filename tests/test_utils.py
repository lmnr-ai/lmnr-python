from lmnr.sdk.utils import is_otel_attribute_value_type, format_id
import uuid
import pytest


def test_is_otel_attribute_value_type():
    # primitive types
    assert is_otel_attribute_value_type(1)
    assert is_otel_attribute_value_type(1.0)
    assert is_otel_attribute_value_type(True)
    assert is_otel_attribute_value_type(False)
    assert is_otel_attribute_value_type("test")
    assert is_otel_attribute_value_type(b"test")

    assert not is_otel_attribute_value_type(None)

    # empty sequences
    assert is_otel_attribute_value_type([])
    assert is_otel_attribute_value_type(())
    assert is_otel_attribute_value_type(tuple())

    # non-empty sequences of same type
    assert is_otel_attribute_value_type([1, 2, 3])
    assert is_otel_attribute_value_type((1, 2, 3))
    assert is_otel_attribute_value_type(("a", "b", "c"))
    assert is_otel_attribute_value_type((True, False, True))

    # nested sequences
    assert not is_otel_attribute_value_type([[1, 2, 3], [4, 5, 6]])
    assert not is_otel_attribute_value_type([(1, 2, 3), (4, 5, 6)])
    assert not is_otel_attribute_value_type([("a", "b", "c"), ("d", "e", "f")])
    assert not is_otel_attribute_value_type([(True, False, True), (False, True, False)])

    # non-empty sequences of different types
    assert not is_otel_attribute_value_type([1, "a", True])
    assert not is_otel_attribute_value_type((1, "a", True))
    assert not is_otel_attribute_value_type(("a", 1, True))


def test_format_id_with_uuid():
    """Test format_id with UUID objects."""
    test_uuid = uuid.UUID("12345678-1234-5678-9abc-123456789abc")
    result = format_id(test_uuid)
    assert result == "12345678-1234-5678-9abc-123456789abc"
    assert isinstance(result, str)


def test_format_id_with_int():
    """Test format_id with integer values."""
    # Test with a valid integer that can be converted to UUID
    test_int = 123456789012345678901234567890123456
    result = format_id(test_int)
    
    # Verify it's a valid UUID string
    uuid.UUID(result)
    assert isinstance(result, str)
    
    # Test with zero
    result_zero = format_id(0)
    assert result_zero == "00000000-0000-0000-0000-000000000000"


def test_format_id_with_valid_uuid_string():
    """Test format_id with valid UUID strings."""
    test_uuid_str = "12345678-1234-5678-9abc-123456789abc"
    result = format_id(test_uuid_str)
    assert result == test_uuid_str
    
    # Test with uppercase UUID string
    test_uuid_upper = "12345678-1234-5678-9ABC-123456789ABC"
    result_upper = format_id(test_uuid_upper)
    assert result_upper == test_uuid_upper


def test_format_id_with_uuid_string_no_hyphens():
    """Test format_id with UUID string without hyphens."""
    test_uuid_no_hyphens = "123456781234567890ab123456789abc"
    result = format_id(test_uuid_no_hyphens)
    assert result == test_uuid_no_hyphens


def test_format_id_with_invalid_string():
    """Test format_id with invalid string values."""
    with pytest.raises(ValueError):
        format_id("not-a-valid-uuid")
    
    with pytest.raises(ValueError):
        format_id("12345")  # Too short
    
    with pytest.raises(ValueError):
        format_id("invalid-uuid-string-format")
    
    # String that's too long for UUID
    with pytest.raises(ValueError):
        format_id("12345678901234567890123456789012345678901234567890")
    
    # String with invalid characters for UUID
    with pytest.raises(ValueError):
        format_id("gggggggg-1234-5678-9abc-123456789abc")
    
    # Decimal number as string (no longer supported)
    with pytest.raises(ValueError):
        format_id("123456789012345678901234567890123456")


def test_format_id_with_invalid_types():
    """Test format_id with invalid input types."""
    with pytest.raises(ValueError, match="Invalid ID type"):
        format_id(None)
    
    with pytest.raises(ValueError, match="Invalid ID type"):
        format_id([])
    
    with pytest.raises(ValueError, match="Invalid ID type"):
        format_id({})
    
    with pytest.raises(ValueError, match="Invalid ID type"):
        format_id(1.5)


def test_format_id_consistency():
    """Test that format_id is consistent with round-trip conversions."""
    # Test UUID -> string -> UUID consistency
    original_uuid = uuid.UUID("12345678-1234-5678-9abc-123456789abc")
    formatted = format_id(original_uuid)
    parsed_back = uuid.UUID(formatted)
    assert original_uuid == parsed_back
    
    # Test int -> UUID -> string -> UUID consistency
    original_int = 123456789012345678901234567890123456
    formatted_from_int = format_id(original_int)
    parsed_uuid = uuid.UUID(formatted_from_int)
    assert parsed_uuid.int == original_int


def test_format_id_clear_behavior():
    """Test that format_id has clear, predictable behavior for each input type."""
    # UUID objects -> string representation
    test_uuid = uuid.uuid4()
    assert format_id(test_uuid) == str(test_uuid)
    
    # Integers -> UUID from integer -> string
    test_int = 42
    result = format_id(test_int)
    expected = str(uuid.UUID(int=test_int))
    assert result == expected
    
    # Valid UUID strings -> returned as-is (after validation)
    valid_uuid_str = "12345678-1234-5678-9abc-123456789abc"
    assert format_id(valid_uuid_str) == valid_uuid_str
    
    # Invalid strings -> ValueError (no guessing)
    with pytest.raises(ValueError):
        format_id("not a uuid at all")
