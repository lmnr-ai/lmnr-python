from lmnr.sdk.utils import is_otel_attribute_value_type


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
