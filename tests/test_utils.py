import base64
import dataclasses
import json
import pytest
import uuid

from typing import Any, Dict, List
from pydantic import BaseModel, ConfigDict

from lmnr.sdk.utils import is_otel_attribute_value_type, format_id, json_dumps


class SimplePydanticModel(BaseModel):
    name: str
    value: int
    active: bool = True


class NestedPydanticModel(BaseModel):
    simple: SimplePydanticModel
    items: List[str]
    metadata: Dict[str, Any] = {}


class PydanticModelWithCustomMethods(BaseModel):
    name: str

    def to_json(self):
        return f'{{"custom_name": "{self.name}_custom"}}'

    def json(self):
        return f'{{"json_name": "{self.name}_json"}}'


class ComplexPydanticModel(BaseModel):
    model_config = ConfigDict(validate_by_name=True)
    id: int
    user: SimplePydanticModel
    tags: List[str]
    settings: Dict[str, Any]


# Test dataclasses
@dataclasses.dataclass
class SimpleDataClass:
    name: str
    value: int


@dataclasses.dataclass
class NestedDataClass:
    simple: SimpleDataClass
    items: List[str]


class CircularRef:
    def __init__(self, name: str):
        self.name = name
        self.ref = None


class FailingStr:
    """Class that raises an exception in __str__ method"""

    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        raise RuntimeError("__str__ method failed")

    def __repr__(self):
        return f"FailingStr(name='{self.name}')"


class FailingRepr:
    """Class that raises an exception in __repr__ method"""

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        raise RuntimeError("__repr__ method failed")

    def __str__(self):
        return f"FailingRepr with name: {self.name}"


class FailingBoth:
    """Class that raises exceptions in both __str__ and __repr__ methods"""

    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        raise RuntimeError("__str__ method failed")

    def __repr__(self):
        raise RuntimeError("__repr__ method failed")


class ComplexFailingObject:
    """Class that fails in str() but has other attributes"""

    def __init__(self, value: int):
        self.value = value
        self.data = {"key": "value"}

    def __str__(self):
        raise ValueError("Cannot convert to string")

    def __repr__(self):
        raise ValueError("Cannot create repr")


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


def test_json_dumps_basic():
    """Test basic serialization"""
    result = json_dumps({"a": 1, "b": "test"})
    parsed = json.loads(result)
    assert parsed == {"a": 1, "b": "test"}

    result = json_dumps({"a": 1, "b": "test", "c": [1, 2, 3]})
    parsed = json.loads(result)
    assert parsed == {"a": 1, "b": "test", "c": [1, 2, 3]}


def test_json_dumps_sequence_types():
    """Test different sequence types"""
    # Lists
    result = json_dumps([1, 2, 3])
    parsed = json.loads(result)
    assert parsed == [1, 2, 3]

    # Tuples
    result = json_dumps((1, 2, 3))
    parsed = json.loads(result)
    assert parsed == [1, 2, 3]  # Tuples become lists in JSON

    # Sets (order may vary, so check content)
    result = json_dumps({1, 2, 3})
    parsed = json.loads(result)
    assert set(parsed) == {1, 2, 3}

    # Nested sequences
    result = json_dumps({"list": [1, 2], "tuple": (3, 4)})
    parsed = json.loads(result)
    assert parsed == {"list": [1, 2], "tuple": [3, 4]}


def test_json_dumps_dataclass():
    """Test dataclass serialization"""
    simple = SimpleDataClass(name="test", value=42)
    result = json_dumps(simple)
    parsed = json.loads(result)
    assert parsed == {"name": "test", "value": 42}

    # Nested dataclass
    nested = NestedDataClass(simple=simple, items=["a", "b"])
    result = json_dumps(nested)
    parsed = json.loads(result)
    assert parsed["simple"]["name"] == "test"
    assert parsed["simple"]["value"] == 42
    assert parsed["items"] == ["a", "b"]


def test_json_dumps_circular_reference():
    """Test circular reference handling"""
    obj1 = CircularRef("obj1")
    obj2 = CircularRef("obj2")
    obj1.ref = obj2
    obj2.ref = obj1

    # This should not cause infinite recursion
    result = json_dumps(obj1)
    assert isinstance(result, str)
    # Should fallback to string representation or handle gracefully


def test_json_dumps_mixed_nested_types():
    """Test complex nested structures with different types"""
    dataclass_obj = SimpleDataClass(name="dataclass", value=2)

    complex_data = {
        "list": [1, 2, 3],
        "tuple": (4, 5, 6),
        "set": {7, 8, 9},
        "nested_dict": {"inner_list": [10, 11, 12], "inner_tuple": (13, 14, 15)},
        "dataclass": dataclass_obj,
    }

    result = json_dumps(complex_data)
    parsed = json.loads(result)

    # Verify structure
    assert parsed["list"] == [1, 2, 3]
    assert parsed["tuple"] == [4, 5, 6]
    assert set(parsed["set"]) == {7, 8, 9}
    assert parsed["nested_dict"]["inner_list"] == [10, 11, 12]
    assert parsed["nested_dict"]["inner_tuple"] == [13, 14, 15]
    assert parsed["dataclass"]["name"] == "dataclass"
    assert parsed["dataclass"]["value"] == 2


def test_json_dumps_unsupported_types():
    """Test handling of unsupported types"""

    # Function object
    def sample_func():
        pass

    result = json_dumps(sample_func)
    parsed = json.loads(result)
    assert isinstance(parsed, str)  # Should fallback to string representation

    # Lambda
    result = json_dumps(lambda x: x)
    parsed = json.loads(result)
    assert isinstance(parsed, str)


def test_json_dumps_generators_and_iterators():
    """Test that generators and iterators are handled properly"""

    # Generator
    def gen():
        yield 1
        yield 2
        yield 3

    result = json_dumps(gen())
    parsed = json.loads(result)
    assert isinstance(
        parsed, str
    )  # Should fallback to string, not consume the generator

    # Iterator
    iterator = iter([1, 2, 3])
    result = json_dumps(iterator)
    parsed = json.loads(result)
    assert isinstance(
        parsed, str
    )  # Should fallback to string, not consume the iterator


def test_json_dumps_deeply_nested():
    """Test deeply nested structures"""
    nested = {"level1": {"level2": {"level3": {"level4": {"value": "deep"}}}}}
    result = json_dumps(nested)
    parsed = json.loads(result)
    assert parsed["level1"]["level2"]["level3"]["level4"]["value"] == "deep"


def test_json_dumps_empty_containers():
    """Test empty containers"""
    result = json_dumps([])
    parsed = json.loads(result)
    assert parsed == []

    result = json_dumps({})
    parsed = json.loads(result)
    assert parsed == {}

    result = json_dumps(())
    parsed = json.loads(result)
    assert parsed == []  # Tuples become lists in JSON

    result = json_dumps(set())
    parsed = json.loads(result)
    assert parsed == []  # Sets become lists in JSON


def test_json_dumps_none_values():
    """Test None values"""
    result = json_dumps(None)
    parsed = json.loads(result)
    assert parsed is None

    result = json_dumps({"key": None})
    parsed = json.loads(result)
    assert parsed == {"key": None}


# Original test updated to use parsed comparison
def test_original_json_dumps():
    result = json_dumps({"a": 1, "b": "test"})
    parsed = json.loads(result)
    assert parsed == {"a": 1, "b": "test"}

    result = json_dumps({"a": 1, "b": "test", "c": [1, 2, 3]})
    parsed = json.loads(result)
    assert parsed == {"a": 1, "b": "test", "c": [1, 2, 3]}

    result = json_dumps({"a": 1, "b": "test", "c": [1, 2, 3]})
    parsed = json.loads(result)
    assert parsed == {"a": 1, "b": "test", "c": [1, 2, 3]}


def test_json_dumps_simple_pydantic():
    """Test basic pydantic model serialization"""
    model = SimplePydanticModel(name="test", value=42)
    result = json_dumps(model)

    # Check the parsed structure instead of exact bytes
    parsed = json.loads(result)
    expected = {"name": "test", "value": 42, "active": True}
    assert parsed == expected


def test_json_dumps_pydantic_with_defaults():
    """Test pydantic model with default values"""
    model = SimplePydanticModel(name="test", value=42, active=False)
    result = json_dumps(model)

    parsed = json.loads(result)
    assert parsed["name"] == "test"
    assert parsed["value"] == 42
    assert parsed["active"] is False


def test_json_dumps_nested_pydantic():
    """Test nested pydantic models"""
    simple = SimplePydanticModel(name="nested", value=100)
    model = NestedPydanticModel(
        simple=simple, items=["a", "b", "c"], metadata={"key": "value", "count": 3}
    )
    result = json_dumps(model)

    parsed = json.loads(result)
    assert parsed["simple"]["name"] == "nested"
    assert parsed["simple"]["value"] == 100
    assert parsed["simple"]["active"] is True
    assert parsed["items"] == ["a", "b", "c"]
    assert parsed["metadata"]["key"] == "value"
    assert parsed["metadata"]["count"] == 3


def test_json_dumps_complex_pydantic():
    """Test complex pydantic model with various data types"""
    user = SimplePydanticModel(name="alice", value=30)
    model = ComplexPydanticModel(
        id=123,
        user=user,
        tags=["admin", "user", "active"],
        settings={
            "theme": "dark",
            "notifications": True,
            "limits": {"max_uploads": 10, "timeout": 30.5},
        },
    )
    result = json_dumps(model)

    parsed = json.loads(result)
    assert parsed["id"] == 123
    assert parsed["user"]["name"] == "alice"
    assert parsed["user"]["value"] == 30
    assert parsed["tags"] == ["admin", "user", "active"]
    assert parsed["settings"]["theme"] == "dark"
    assert parsed["settings"]["notifications"] is True
    assert parsed["settings"]["limits"]["max_uploads"] == 10
    assert parsed["settings"]["limits"]["timeout"] == 30.5


def test_json_dumps_pydantic_with_custom_methods():
    """Test pydantic model with custom to_json/json methods"""
    model = PydanticModelWithCustomMethods(name="test")
    result = json_dumps(model)

    parsed = json.loads(result)
    assert parsed["name"] == "test"


def test_json_dumps_mixed_pydantic_and_dataclass():
    """Test mix of pydantic models, dataclasses, and other types"""
    pydantic_model = SimplePydanticModel(name="pydantic", value=1)
    dataclass_obj = SimpleDataClass(name="dataclass", value=2)

    mixed_data = {
        "pydantic": pydantic_model,
        "dataclass": dataclass_obj,
        "list": [1, 2, 3],
        "tuple": (4, 5, 6),
        "set": {7, 8, 9},
        "nested": {"inner_pydantic": pydantic_model, "inner_dataclass": dataclass_obj},
    }

    result = json_dumps(mixed_data)
    parsed = json.loads(result)

    # Check pydantic serialization
    assert parsed["pydantic"]["name"] == "pydantic"
    assert parsed["pydantic"]["value"] == 1
    assert parsed["pydantic"]["active"] is True

    # Check dataclass serialization
    assert parsed["dataclass"]["name"] == "dataclass"
    assert parsed["dataclass"]["value"] == 2

    # Check other types
    assert parsed["list"] == [1, 2, 3]
    assert parsed["tuple"] == [4, 5, 6]  # tuple becomes list in JSON
    assert set(parsed["set"]) == {7, 8, 9}  # set becomes list, order may vary

    # Check nested structures
    assert parsed["nested"]["inner_pydantic"]["name"] == "pydantic"
    assert parsed["nested"]["inner_dataclass"]["name"] == "dataclass"


def test_json_dumps_pydantic_list():
    """Test list of pydantic models"""
    models = [
        SimplePydanticModel(name="first", value=1),
        SimplePydanticModel(name="second", value=2, active=False),
        SimplePydanticModel(name="third", value=3),
    ]

    result = json_dumps(models)
    parsed = json.loads(result)

    assert len(parsed) == 3
    assert parsed[0]["name"] == "first"
    assert parsed[0]["value"] == 1
    assert parsed[0]["active"] is True

    assert parsed[1]["name"] == "second"
    assert parsed[1]["value"] == 2
    assert parsed[1]["active"] is False

    assert parsed[2]["name"] == "third"
    assert parsed[2]["value"] == 3
    assert parsed[2]["active"] is True


def test_json_dumps_deeply_nested_pydantic():
    """Test deeply nested pydantic structures"""
    level3 = SimplePydanticModel(name="level3", value=3)
    level2 = NestedPydanticModel(simple=level3, items=["c", "d"])
    level1 = ComplexPydanticModel(
        id=1,
        user=level3,
        tags=["nested"],
        settings={
            "nested_model": level2.model_dump()
        },  # Manually serialize for deep nesting
    )

    result = json_dumps(level1)
    parsed = json.loads(result)

    assert parsed["id"] == 1
    assert parsed["user"]["name"] == "level3"
    assert parsed["tags"] == ["nested"]
    assert parsed["settings"]["nested_model"]["simple"]["name"] == "level3"
    assert parsed["settings"]["nested_model"]["items"] == ["c", "d"]


def test_json_dumps_pydantic_with_none_values():
    """Test pydantic model with None values"""
    from typing import Optional

    class ModelWithOptional(BaseModel):
        name: str
        optional_value: Optional[int] = None
        optional_string: Optional[str] = None

    model = ModelWithOptional(name="test")
    result = json_dumps(model)
    parsed = json.loads(result)

    assert parsed["name"] == "test"
    assert parsed["optional_value"] is None
    assert parsed["optional_string"] is None


def test_json_dumps_pydantic_edge_cases():
    """Test pydantic models with edge cases"""
    from datetime import datetime, date
    import uuid

    class EdgeCaseModel(BaseModel):
        text: str
        number: int
        date_val: date
        datetime_val: datetime
        uuid_val: uuid.UUID
        empty_list: List[str] = []
        empty_dict: Dict[str, Any] = {}

    test_uuid = uuid.uuid4()
    test_date = date(2024, 1, 15)
    test_datetime = datetime(2024, 1, 15, 10, 30, 45)

    model = EdgeCaseModel(
        text="test",
        number=42,
        date_val=test_date,
        datetime_val=test_datetime,
        uuid_val=test_uuid,
    )

    result = json_dumps(model)
    parsed = json.loads(result)

    assert parsed["text"] == "test"
    assert parsed["number"] == 42
    # Dates should be serialized as strings
    assert parsed["date_val"] == "2024-01-15"
    assert parsed["datetime_val"] == "2024-01-15T10:30:45"
    assert parsed["uuid_val"] == str(test_uuid)
    assert parsed["empty_list"] == []
    assert parsed["empty_dict"] == {}


def test_json_dumps_failing_str():
    """Test object that fails in __str__ but works in __repr__"""
    obj = FailingStr("test")
    result = json_dumps(obj)

    # When serialization completely fails, json_dumps returns "{}"
    assert result == "{}"


def test_json_dumps_failing_repr():
    """Test object that fails in __repr__ but works in __str__"""
    obj = FailingRepr("test")
    result = json_dumps(obj)

    # Should succeed using __str__ method as fallback
    parsed = json.loads(result)
    assert isinstance(parsed, str)
    assert "FailingRepr with name: test" in parsed


def test_json_dumps_failing_both():
    """Test object that fails in both __str__ and __repr__"""
    obj = FailingBoth("test")
    result = json_dumps(obj)

    # When serialization completely fails, json_dumps returns "{}"
    assert result == "{}"


def test_json_dumps_complex_failing_object():
    """Test complex object that fails in string conversion"""
    obj = ComplexFailingObject(42)
    result = json_dumps(obj)

    # When serialization completely fails, json_dumps returns "{}"
    assert result == "{}"


def test_json_dumps_failing_objects_in_nested_structures():
    """Test failing objects within nested data structures"""
    failing_str = FailingStr("nested")
    failing_both = FailingBoth("deeply_nested")

    complex_data = {
        "normal": "value",
        "failing_str": failing_str,
        "list_with_failing": [1, 2, failing_both, 4],
        "nested_dict": {"inner": failing_str, "normal": "still_works"},
    }

    result = json_dumps(complex_data)

    # When any object in the structure fails to serialize, the entire thing fails
    assert json.loads(result) == {
        "normal": "value",
        "failing_str": {},
        "list_with_failing": [1, 2, {}, 4],
        "nested_dict": {"inner": {}, "normal": "still_works"},
    }


def test_json_dumps_fallback_hierarchy():
    """Test the complete fallback hierarchy"""
    # Test with different types of objects to verify fallback behavior

    # These should fail completely and return "{}"
    failing_cases = [
        FailingStr(
            "test1"
        ),  # __str__ fails, __repr__ works but encoder might not use it
        FailingBoth("test3"),  # Both __str__ and __repr__ fail
        ComplexFailingObject(99),  # Both __str__ and __repr__ fail
    ]

    for obj in failing_cases:
        result = json_dumps(obj)
        # Should produce the fallback empty JSON object
        assert result == "{}"

    # This should succeed using __str__ method
    succeeding_case = FailingRepr("test2")  # __repr__ fails but __str__ works
    result = json_dumps(succeeding_case)
    parsed = json.loads(result)
    assert isinstance(parsed, str)
    assert "FailingRepr with name: test2" in parsed


def test_json_dumps_working_objects_with_embedded_failing():
    """Test that working objects serialize correctly even when some fail"""
    # Create a structure with mostly working objects
    simple_dc = SimpleDataClass(name="working", value=100)

    # Test with only working objects first
    working_data = {"dataclass": simple_dc, "normal_list": [1, 2, 3], "string": "test"}

    result = json_dumps(working_data)
    parsed = json.loads(result)

    # Should work normally
    assert parsed["dataclass"]["name"] == "working"
    assert parsed["dataclass"]["value"] == 100
    assert parsed["normal_list"] == [1, 2, 3]
    assert parsed["string"] == "test"


def test_json_dumps_mixed_failing_and_working():
    """Test mix of working and failing objects"""
    simple_dc = SimpleDataClass(name="working", value=100)
    failing_obj = FailingBoth("mixed_test")

    # Any failing object in the structure causes the entire serialization to fail
    mixed_data = {
        "dataclass": simple_dc,
        "failing": failing_obj,
        "normal_list": [1, 2, 3],
    }

    result = json_dumps(mixed_data)
    # Should return the fallback empty object
    assert json.loads(result) == {
        "dataclass": {"name": "working", "value": 100},
        "failing": {},
        "normal_list": [1, 2, 3],
    }


def test_json_dumps_encoder_fallback_success():
    """Test cases where the encoder fallback succeeds"""

    # Test with a custom object that has a good __str__ method
    class GoodCustomObject:
        def __init__(self, name):
            self.name = name

        def __str__(self):
            return f"GoodCustomObject({self.name})"

    obj = GoodCustomObject("test")
    result = json_dumps(obj)
    parsed = json.loads(result)

    # Should successfully serialize to a string
    assert isinstance(parsed, str)
    assert "GoodCustomObject(test)" == parsed


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
    with pytest.raises(TypeError, match="Invalid ID type"):
        format_id(None)

    with pytest.raises(TypeError, match="Invalid ID type"):
        format_id([])

    with pytest.raises(TypeError, match="Invalid ID type"):
        format_id({})

    with pytest.raises(TypeError, match="Invalid ID type"):
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


# Tests for merge_text_parts function
def test_merge_text_parts_empty_list():
    """Test that empty list returns empty list"""
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.google_genai.utils import (
        merge_text_parts,
    )

    result = merge_text_parts([])
    assert result == []


def test_merge_text_parts_consecutive_strings():
    """Test merging consecutive string inputs"""
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.google_genai.utils import (
        merge_text_parts,
    )

    parts = ["Hello ", "world", "!"]
    result = merge_text_parts(parts)

    assert len(result) == 1
    assert result[0].text == "Hello world!"


def test_merge_text_parts_consecutive_part_objects():
    """Test merging consecutive Part objects with text"""
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.google_genai.utils import (
        merge_text_parts,
    )
    from google.genai import types

    parts = [
        types.Part(text="abc"),
        types.Part(text="def"),
        types.Part(text="ghi"),
    ]
    result = merge_text_parts(parts)

    assert len(result) == 1
    assert result[0].text == "abcdefghi"


def test_merge_text_parts_consecutive_part_dicts():
    """Test merging consecutive PartDict (dict) inputs"""
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.google_genai.utils import (
        merge_text_parts,
    )

    parts = [
        {"text": "First "},
        {"text": "second "},
        {"text": "third"},
    ]
    result = merge_text_parts(parts)

    assert len(result) == 1
    assert result[0].text == "First second third"


def test_merge_text_parts_mixed_types():
    """Test merging with mixed input types (str, Part, dict)"""
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.google_genai.utils import (
        merge_text_parts,
    )
    from google.genai import types

    parts = [
        "Start ",
        types.Part(text="middle "),
        {"text": "end"},
    ]
    result = merge_text_parts(parts)

    assert len(result) == 1
    assert result[0].text == "Start middle end"


def test_merge_text_parts_with_non_text_part():
    """Test that non-text parts break the merge sequence"""
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.google_genai.utils import (
        merge_text_parts,
    )
    from google.genai import types

    # Create an inline_data part (e.g., image)
    inline_data_part = types.Part(
        inline_data={"mime_type": "image/png", "data": b"fake_image_data"}
    )

    parts = [
        types.Part(text="abc"),
        types.Part(text="def"),
        inline_data_part,
        types.Part(text="xyz"),
    ]
    result = merge_text_parts(parts)

    # Should result in 3 parts: merged text "abcdef", inline_data, text "xyz"
    assert len(result) == 3
    assert result[0].text == "abcdef"
    assert result[1].inline_data is not None
    assert result[2].text == "xyz"


def test_merge_text_parts_with_function_call():
    """Test that function call parts break the merge sequence"""
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.google_genai.utils import (
        merge_text_parts,
    )
    from google.genai import types

    # Create a function call part
    function_call_part = types.Part(
        function_call={"name": "get_weather", "args": {"location": "Tokyo"}}
    )

    parts = [
        types.Part(text="The weather is "),
        function_call_part,
        types.Part(text=" degrees."),
    ]
    result = merge_text_parts(parts)

    # Should result in 3 parts: text, function_call, text
    assert len(result) == 3
    assert result[0].text == "The weather is "
    assert result[1].function_call is not None
    assert result[2].text == " degrees."


def test_merge_text_parts_multiple_non_text_parts():
    """Test multiple non-text parts with text in between"""
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.google_genai.utils import (
        merge_text_parts,
    )
    from google.genai import types

    inline_data_part1 = types.Part(
        inline_data={"mime_type": "image/png", "data": b"image1"}
    )
    inline_data_part2 = types.Part(
        inline_data={"mime_type": "image/png", "data": b"image2"}
    )

    parts = [
        types.Part(text="Text1 "),
        types.Part(text="Text2"),
        inline_data_part1,
        types.Part(text="Text3"),
        inline_data_part2,
        types.Part(text="Text4 "),
        types.Part(text="Text5"),
    ]
    result = merge_text_parts(parts)

    # Should result in 5 parts:
    # merged "Text1 Text2", image1, "Text3", image2, merged "Text4 Text5"
    assert len(result) == 5
    assert result[0].text == "Text1 Text2"
    assert result[1].inline_data is not None
    assert result[2].text == "Text3"
    assert result[3].inline_data is not None
    assert result[4].text == "Text4 Text5"


def test_merge_text_parts_only_non_text_parts():
    """Test that only non-text parts are preserved as-is"""
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.google_genai.utils import (
        merge_text_parts,
    )
    from google.genai import types

    inline_data_part1 = types.Part(
        inline_data={"mime_type": "image/png", "data": b"image1"}
    )
    inline_data_part2 = types.Part(
        inline_data={"mime_type": "image/jpeg", "data": b"image2"}
    )

    parts = [inline_data_part1, inline_data_part2]
    result = merge_text_parts(parts)

    # Should result in 2 parts unchanged
    assert len(result) == 2
    assert result[0].inline_data is not None
    assert result[1].inline_data is not None


def test_merge_text_parts_single_text_part():
    """Test that a single text part is returned as-is"""
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.google_genai.utils import (
        merge_text_parts,
    )
    from google.genai import types

    parts = [types.Part(text="Single text")]
    result = merge_text_parts(parts)

    assert len(result) == 1
    assert result[0].text == "Single text"


def test_merge_text_parts_with_file_object():
    """Test that File objects break the merge sequence"""
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.google_genai.utils import (
        merge_text_parts,
    )
    from google.genai import types

    # Create a File object
    file_obj = types.File(name="document.pdf", uri="gs://bucket/document.pdf")

    parts = [
        types.Part(text="Before file "),
        types.Part(text="part"),
        file_obj,
        types.Part(text="After "),
        types.Part(text="file"),
    ]
    result = merge_text_parts(parts)

    # Should result in 3 parts: merged text, file, merged text
    assert len(result) == 3
    assert result[0].text == "Before file part"
    assert isinstance(result[1], types.File)
    assert result[2].text == "After file"


def test_merge_text_parts_trailing_text_only():
    """Test parts ending with text (no non-text parts)"""
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.google_genai.utils import (
        merge_text_parts,
    )
    from google.genai import types

    parts = [
        types.Part(text="Part1 "),
        types.Part(text="Part2 "),
        types.Part(text="Part3"),
    ]
    result = merge_text_parts(parts)

    assert len(result) == 1
    assert result[0].text == "Part1 Part2 Part3"


def test_merge_text_parts_leading_non_text():
    """Test parts starting with non-text part"""
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.google_genai.utils import (
        merge_text_parts,
    )
    from google.genai import types

    inline_data_part = types.Part(
        inline_data={"mime_type": "image/png", "data": b"image"}
    )

    parts = [
        inline_data_part,
        types.Part(text="Text1 "),
        types.Part(text="Text2"),
    ]
    result = merge_text_parts(parts)

    assert len(result) == 2
    assert result[0].inline_data is not None
    assert result[1].text == "Text1 Text2"


def test_json_dumps_bytes_use_standard_base64():
    """Bytes must serialize as STANDARD base64, not a Python repr.

    Without a bytes branch they fall through to `str(o)` and land in the span
    as the useless literal "b'\\xfb\\xef'". Standard rather than URL-safe
    because consumers decode with `base64.b64decode`, which defaults to
    `validate=False` and silently DROPS `-`/`_` instead of raising, shifting
    every following byte.
    """
    raw = bytes([0xFB, 0xEF, 0xBE, 0xFF, 0xFF, 0xFF])

    parsed = json.loads(json_dumps({"data": raw}))

    assert parsed["data"] == "++++////"
    assert base64.b64decode(parsed["data"]) == raw

    # bytearray takes the same path
    parsed = json.loads(json_dumps({"data": bytearray(raw)}))
    assert parsed["data"] == "++++////"

    # and nested inside containers the orjson hook has to recurse through
    parsed = json.loads(json_dumps({"outer": [{"inner": raw}, {raw}]}))
    assert parsed["outer"][0]["inner"] == "++++////"
    assert parsed["outer"][1] == ["++++////"]


def test_json_dumps_non_string_dict_keys_do_not_discard_the_payload():
    """An unencodable mapping key must not collapse the whole document.

    orjson's `OPT_NON_STR_KEYS` covers only a fixed set of scalar key types and
    raises on the rest, which would drop every sibling value with it.
    """
    raw = bytes([0xFB, 0xEF, 0xBE, 0xFF, 0xFF, 0xFF])

    parsed = json.loads(json_dumps({"keep": "me", "grid": {(0, 1): "hit"}}))
    assert parsed["keep"] == "me"
    assert parsed["grid"] == {"(0, 1)": "hit"}

    # bytes keys become standard base64 rather than a repr
    parsed = json.loads(json_dumps({"keep": "me", "by_hash": {raw: "hit"}}))
    assert parsed["keep"] == "me"
    assert parsed["by_hash"] == {"++++////": "hit"}

    # nested under a list, too
    parsed = json.loads(json_dumps({"rows": [{"keep": 1, "k": {(2, 3): "v"}}]}))
    assert parsed["rows"][0]["keep"] == 1

    # keys orjson already handles natively are left exactly as they were
    parsed = json.loads(json_dumps({1: "a", "b": 2, None: "c"}))
    assert parsed == {"1": "a", "b": 2, "null": "c"}


def test_json_dumps_key_names_do_not_depend_on_the_retry_path():
    """The retry must not rename keys orjson already encodes itself.

    `OPT_NON_STR_KEYS` renders these differently from `str()` (`None` -> "null",
    `True` -> "true", datetimes -> RFC 3339, enums -> their value), so coercing
    them would make a key's name depend on whether some unrelated sibling key
    happened to force the retry.
    """
    import datetime
    import enum
    import uuid

    class Color(enum.Enum):
        RED = "red"

    native_keys = {
        None: 1,
        True: 2,
        False: 3,
        7: 4,
        2.5: 5,
        uuid.UUID(int=1): 6,
        datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc): 7,
        datetime.date(2026, 1, 1): 8,
        datetime.time(1, 2, 3): 9,
        Color.RED: 10,
        "plain": 11,
    }

    first_pass = json.loads(json_dumps({"m": dict(native_keys)}))["m"]
    # A tuple key elsewhere in the document forces the retry.
    retry_pass = json.loads(
        json_dumps({"m": dict(native_keys), "other": {(0, 1): "x"}})
    )["m"]

    assert first_pass == retry_pass
    assert "null" in first_pass and "true" in first_pass
    assert "2026-01-01T00:00:00Z" in first_pass
    assert "red" in first_pass


def test_json_dumps_dict_like_objects_do_not_stringify_to_python_reprs():
    """A Mapping / non-builtin Sequence must serialize structurally.

    Falling through to `str(o)` yields a Python repr with SINGLE quotes —
    "{'a': 1}" — which is not JSON and no consumer can parse it.
    """
    import collections
    import collections.abc

    class ReadOnlyMapping(collections.abc.Mapping):
        def __init__(self, data):
            self._data = data

        def __getitem__(self, key):
            return self._data[key]

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

    class CustomSequence(collections.abc.Sequence):
        def __init__(self, items):
            self._items = items

        def __getitem__(self, index):
            return self._items[index]

        def __len__(self):
            return len(self._items)

    mapping = ReadOnlyMapping({"a": 1, "b": [1, {"c": 2}]})
    assert json.loads(json_dumps({"m": mapping}))["m"] == {"a": 1, "b": [1, {"c": 2}]}

    parsed = json.loads(json_dumps({"s": CustomSequence([{"a": 1}, 2])}))
    assert parsed["s"] == [{"a": 1}, 2]

    # No Python repr anywhere in the output.
    assert "'" not in json_dumps({"m": ReadOnlyMapping({"a": 1})})

    # str/bytes are Sequences too — they must NOT be exploded into char lists.
    assert json.loads(json_dumps({"t": "hi"}))["t"] == "hi"
    assert json.loads(json_dumps({"b": b"\xfb\xef\xbe\xff\xff\xff"}))["b"] == "++++////"

    # A genuinely opaque object still degrades to its repr, as before.
    class Opaque:
        def __repr__(self):
            return "Opaque()"

    assert json.loads(json_dumps({"o": Opaque()}))["o"] == "Opaque()"

    # dict subclasses were already fine; pin them so the new branches can't
    # accidentally reorder or lose them.
    assert json.loads(json_dumps({"d": collections.OrderedDict(a=1)}))["d"] == {"a": 1}


def test_json_dumps_recovers_bad_keys_nested_in_non_builtin_containers():
    """The retry must walk every container `default_json` unwraps.

    A bad key inside a `Mapping` or a custom `Sequence` — rather than a plain
    dict/list — otherwise survives the retry, fails the second dump too, and
    takes every sibling field down with it. Reachable via google-genai's
    `FunctionResponse.response`, typed `dict[str, Any]`.
    """
    import collections.abc

    class ReadOnlyMapping(collections.abc.Mapping):
        def __init__(self, data):
            self._data = data

        def __getitem__(self, key):
            return self._data[key]

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

    class CustomSequence(collections.abc.Sequence):
        def __init__(self, items):
            self._items = items

        def __getitem__(self, index):
            return self._items[index]

        def __len__(self):
            return len(self._items)

    # Bad key directly inside a Mapping.
    parsed = json.loads(json_dumps({"keep": "me", "v": ReadOnlyMapping({(0, 1): "x"})}))
    assert parsed["keep"] == "me"
    assert parsed["v"] == {"(0, 1)": "x"}

    # Bad key in a dict nested inside a Mapping, and vice versa.
    nested = ReadOnlyMapping({"inner": {(0, 1): "x"}})
    assert json.loads(json_dumps({"keep": "me", "v": nested}))["keep"] == "me"
    assert json.loads(json_dumps({"keep": "me", "v": {"i": nested}}))["keep"] == "me"

    # Bad key inside a custom Sequence.
    sequence = CustomSequence([{(0, 1): "x"}])
    parsed = json.loads(json_dumps({"keep": "me", "v": sequence}))
    assert parsed["keep"] == "me"
    assert parsed["v"] == [{"(0, 1)": "x"}]

    # A bytes key inside a Mapping still becomes standard base64.
    by_hash = ReadOnlyMapping({b"\xfb\xef\xbe\xff\xff\xff": 1})
    parsed = json.loads(json_dumps({"keep": "me", "v": by_hash}))
    assert parsed["v"] == {"++++////": 1}

    # str/bytes are Sequences but must not explode into char lists on the retry.
    parsed = json.loads(
        json_dumps({"t": "hello", "b": b"\xfb\xef\xbe\xff\xff\xff", "bad": {(0, 1): 1}})
    )
    assert parsed["t"] == "hello"
    assert parsed["b"] == "++++////"


def test_json_dumps_recovers_bad_keys_nested_in_pydantic_models():
    """`default_json` opens pydantic models, so the retry walker must too.

    `part_to_dict` deliberately leaves a nested model un-dumped inside a dict
    part, so a bad key under e.g. a `FunctionResponse` reaches `json_dumps`
    still wrapped in the model.
    """

    class Inner(BaseModel):
        payload: Dict[Any, Any] = {}

    bad = {(0, 1): "hit"}

    # Model directly, and nested under each container kind.
    for value in (
        Inner(payload=bad),
        {"k": Inner(payload=bad)},
        [Inner(payload=bad)],
        (Inner(payload=bad),),
        Inner(payload={"inner": Inner(payload=bad)}),
    ):
        parsed = json.loads(json_dumps({"keep": "me", "v": value}))
        assert parsed["keep"] == "me", value
        assert "(0, 1)" in json_dumps(parsed), value

    # A model on the happy path (no retry) is unchanged.
    assert json.loads(json_dumps({"m": Inner(payload={"a": 1})})) == {
        "m": {"payload": {"a": 1}}
    }


def test_unwrap_container_is_the_single_source_of_truth_for_containers():
    """`default_json` and `_stringify_dict_keys` must not diverge.

    Both route through `_unwrap_container`; every past divergence (builtins only,
    then a missing `Mapping` branch, then a missing `BaseModel` branch) silently
    collapsed a whole span attribute to "{}". This pins the contract rather than
    re-testing each type: anything the unwrapper opens must survive a retry with
    its bad keys repaired.
    """
    import collections.abc

    from lmnr.sdk.utils import _UNWRAP_MISS, _unwrap_container

    class Mapped(collections.abc.Mapping):
        def __init__(self, d):
            self._d = d

        def __getitem__(self, key):
            return self._d[key]

        def __iter__(self):
            return iter(self._d)

        def __len__(self):
            return len(self._d)

    class Model(BaseModel):
        payload: Dict[Any, Any] = {}

    # Opened -> a bad key inside is repairable.
    openable = [{"a": 1}, [1], (1,), {1}, frozenset([1]), Mapped({"a": 1}), Model()]
    for value in openable:
        assert _unwrap_container(value) is not _UNWRAP_MISS, value

    # NOT opened -> str/bytes must stay scalar, or they'd become char lists.
    for value in ("hi", b"ab", bytearray(b"ab"), 1, None, object()):
        assert _unwrap_container(value) is _UNWRAP_MISS, value

    # The contract that matters: for every openable container holding a bad key,
    # siblings survive and the key is coerced.
    for wrap in (
        lambda bad: {"v": bad},
        lambda bad: [bad],
        lambda bad: Mapped({"v": bad}),
        lambda bad: Model(payload=bad),
    ):
        out = json_dumps({"keep": "me", "c": wrap({(0, 1): "x"})})
        assert '"keep":"me"' in out.replace(", ", ",")
        assert "(0, 1)" in out
