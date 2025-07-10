import json
import dataclasses
from typing import List

from lmnr.opentelemetry_lib.decorators import json_dumps, CustomJSONEncoder
from lmnr.sdk.utils import is_otel_attribute_value_type


# Test dataclasses
@dataclasses.dataclass
class SimpleDataClass:
    name: str
    value: int


@dataclasses.dataclass
class NestedDataClass:
    simple: SimpleDataClass
    items: List[str]


class ObjectWithToJson:
    def __init__(self, name: str):
        self.name = name

    def to_json(self):
        return f'{{"to_json_name": "{self.name}"}}'


class ObjectWithJson:
    def __init__(self, name: str):
        self.name = name

    def json(self):
        return f'{{"json_name": "{self.name}"}}'


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
    assert json_dumps({"a": 1, "b": "test"}) == '{"a": 1, "b": "test"}'
    assert (
        json_dumps({"a": 1, "b": "test", "c": [1, 2, 3]})
        == '{"a": 1, "b": "test", "c": [1, 2, 3]}'
    )


def test_json_dumps_sequence_types():
    """Test different sequence types"""
    # Lists
    assert json_dumps([1, 2, 3]) == "[1, 2, 3]"

    # Tuples
    assert json_dumps((1, 2, 3)) == "[1, 2, 3]"

    # Sets (order may vary, so check content)
    result = json_dumps({1, 2, 3})
    parsed = json.loads(result)
    assert set(parsed) == {1, 2, 3}

    # Nested sequences
    assert (
        json_dumps({"list": [1, 2], "tuple": (3, 4)})
        == '{"list": [1, 2], "tuple": [3, 4]}'
    )


def test_json_dumps_dataclass():
    """Test dataclass serialization"""
    simple = SimpleDataClass(name="test", value=42)
    expected = '{"name": "test", "value": 42}'
    assert json_dumps(simple) == expected

    # Nested dataclass
    nested = NestedDataClass(simple=simple, items=["a", "b"])
    result = json_dumps(nested)
    parsed = json.loads(result)
    assert parsed["simple"]["name"] == "test"
    assert parsed["simple"]["value"] == 42
    assert parsed["items"] == ["a", "b"]


def test_json_dumps_object_with_to_json():
    """Test object with to_json method - potential double serialization"""
    obj = ObjectWithToJson("test")
    result = json_dumps(obj)

    # This should NOT result in double serialization
    parsed = json.loads(result)
    assert isinstance(parsed, dict)


def test_json_dumps_object_with_json():
    """Test object with json method - potential double serialization"""
    obj = ObjectWithJson("test")
    result = json_dumps(obj)

    # This should NOT result in double serialization
    parsed = json.loads(result)
    assert isinstance(parsed, dict)


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
    assert json_dumps([]) == "[]"
    assert json_dumps({}) == "{}"
    assert json_dumps(()) == "[]"
    assert json_dumps(set()) == "[]"


def test_json_dumps_none_values():
    """Test None values"""
    assert json_dumps(None) == "null"
    assert json_dumps({"key": None}) == '{"key": null}'


def test_custom_json_encoder_direct():
    """Test the CustomJSONEncoder directly"""
    encoder = CustomJSONEncoder()

    # Test default method with dataclass
    simple_dc = SimpleDataClass(name="test", value=42)
    serialized = encoder.default(simple_dc)
    assert serialized == {"name": "test", "value": 42}

    # Test with list containing dataclass
    list_with_dc = [simple_dc, {"key": "value"}]
    serialized = encoder.default(list_with_dc)
    assert serialized == [{"name": "test", "value": 42}, {"key": "value"}]

    # Test with primitive types (should pass through unchanged)
    assert encoder.default(42) == 42
    assert encoder.default("test") == "test"
    assert encoder.default(True) is True
    assert encoder.default(None) is None


# Original test renamed to avoid conflict
def test_original_json_dumps():
    assert json_dumps({"a": 1, "b": "test"}) == '{"a": 1, "b": "test"}'
    assert (
        json_dumps({"a": 1, "b": "test", "c": [1, 2, 3]})
        == '{"a": 1, "b": "test", "c": [1, 2, 3]}'
    )
    assert (
        json_dumps({"a": 1, "b": "test", "c": [1, 2, 3]})
        == '{"a": 1, "b": "test", "c": [1, 2, 3]}'
    )


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
    assert result == "{}"


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


def test_custom_json_encoder_fallback_direct():
    """Test the CustomJSONEncoder fallback behavior directly"""
    encoder = CustomJSONEncoder()

    # Test with failing __str__ - should use super().default() which may work
    failing_str = FailingStr("direct_test")
    result = encoder.default(failing_str)
    # The result depends on what super().default() returns
    assert result is not None

    # Test with failing __repr__ - should use super().default() which may work
    failing_repr = FailingRepr("direct_test")
    result = encoder.default(failing_repr)
    # The result depends on what super().default() returns
    assert result is not None

    # Test with both failing - should use the final fallback
    failing_both = FailingBoth("direct_test")
    result = encoder.default(failing_both)
    # Should return the object itself as final fallback
    assert result is not None


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
    assert result == "{}"


def test_custom_json_encoder_fallback_order():
    """Test that the fallback order works correctly in the encoder"""
    encoder = CustomJSONEncoder()

    # Test with a simple non-serializable object that doesn't fail in str()
    class SimpleNonSerializable:
        def __init__(self, value):
            self.value = value

    obj = SimpleNonSerializable(42)
    result = encoder.default(obj)

    # Should return a string representation
    assert isinstance(result, str)
    assert "SimpleNonSerializable" in result

    # Test that this works when used with json_dumps
    result_json = json_dumps(obj)
    parsed = json.loads(result_json)
    assert isinstance(parsed, str)
    assert "SimpleNonSerializable" in parsed


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
