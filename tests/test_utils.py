import dataclasses
import json
import pytest
import uuid

from typing import Any, Dict, List
from pydantic import BaseModel

from lmnr.opentelemetry_lib.decorators import json_dumps
from lmnr.sdk.utils import is_otel_attribute_value_type, format_id


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
    id: int
    user: SimplePydanticModel
    tags: List[str]
    settings: Dict[str, Any]

    class Config:
        # Test with pydantic config
        allow_population_by_field_name = True


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
