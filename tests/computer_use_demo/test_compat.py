"""Tests for computer_use_demo.compat module."""

import sys
from enum import Enum

import pytest


def test_strenum_import():
    """Test that StrEnum is properly imported based on Python version."""
    from computer_use_demo.compat import StrEnum

    # StrEnum should be available regardless of Python version
    assert StrEnum is not None
    assert issubclass(StrEnum, str)
    assert issubclass(StrEnum, Enum)


def test_strenum_behavior():
    """Test that StrEnum behaves correctly as a string enum."""
    from computer_use_demo.compat import StrEnum

    class TestEnum(StrEnum):
        OPTION_A = "option_a"
        OPTION_B = "option_b"

    # Test enum member values
    assert TestEnum.OPTION_A.value == "option_a"
    assert TestEnum.OPTION_B.value == "option_b"

    # Test string representation
    assert str(TestEnum.OPTION_A) == "option_a"
    assert str(TestEnum.OPTION_B) == "option_b"

    # Test that members are strings
    assert isinstance(TestEnum.OPTION_A, str)
    assert isinstance(TestEnum.OPTION_B, str)


def test_strenum_comparison():
    """Test that StrEnum members can be compared with strings."""
    from computer_use_demo.compat import StrEnum

    class TestEnum(StrEnum):
        VALUE = "test_value"

    # Test string comparison
    assert TestEnum.VALUE == "test_value"
    assert "test_value" == TestEnum.VALUE
    assert TestEnum.VALUE != "other_value"


def test_strenum_in_dict_keys():
    """Test that StrEnum members can be used as dictionary keys."""
    from computer_use_demo.compat import StrEnum

    class TestEnum(StrEnum):
        KEY1 = "key1"
        KEY2 = "key2"

    test_dict = {
        TestEnum.KEY1: "value1",
        TestEnum.KEY2: "value2"
    }

    assert test_dict[TestEnum.KEY1] == "value1"
    assert test_dict[TestEnum.KEY2] == "value2"


def test_strenum_iteration():
    """Test that StrEnum members can be iterated."""
    from computer_use_demo.compat import StrEnum

    class TestEnum(StrEnum):
        A = "a"
        B = "b"
        C = "c"

    members = list(TestEnum)
    assert len(members) == 3
    assert TestEnum.A in members
    assert TestEnum.B in members
    assert TestEnum.C in members


def test_strenum_string_operations():
    """Test that StrEnum supports string operations."""
    from computer_use_demo.compat import StrEnum

    class TestEnum(StrEnum):
        PREFIX = "test"

    # Test string concatenation
    result = TestEnum.PREFIX + "_suffix"
    assert result == "test_suffix"

    # Test upper/lower
    assert TestEnum.PREFIX.upper() == "TEST"
    assert TestEnum.PREFIX.lower() == "test"

    # Test in string
    assert "est" in TestEnum.PREFIX


def test_strenum_format():
    """Test that StrEnum members can be formatted."""
    from computer_use_demo.compat import StrEnum

    class TestEnum(StrEnum):
        VALUE = "world"

    formatted = f"Hello {TestEnum.VALUE}"
    assert formatted == "Hello world"


def test_strenum_repr():
    """Test string representation of StrEnum members."""
    from computer_use_demo.compat import StrEnum

    class TestEnum(StrEnum):
        MEMBER = "member_value"

    # The repr might vary between implementations, but str should be consistent
    str_repr = str(TestEnum.MEMBER)
    assert str_repr == "member_value"


def test_strenum_value_access():
    """Test accessing the value attribute of StrEnum members."""
    from computer_use_demo.compat import StrEnum

    class TestEnum(StrEnum):
        ITEM = "item_value"

    assert TestEnum.ITEM.value == "item_value"
    assert hasattr(TestEnum.ITEM, "value")


def test_strenum_hash():
    """Test that StrEnum members are hashable."""
    from computer_use_demo.compat import StrEnum

    class TestEnum(StrEnum):
        A = "a"
        B = "b"

    # Should be hashable and usable in sets
    enum_set = {TestEnum.A, TestEnum.B}
    assert len(enum_set) == 2
    assert TestEnum.A in enum_set


def test_strenum_name_attribute():
    """Test that StrEnum members have a name attribute."""
    from computer_use_demo.compat import StrEnum

    class TestEnum(StrEnum):
        MY_VALUE = "my_value"

    assert TestEnum.MY_VALUE.name == "MY_VALUE"


def test_strenum_case_sensitivity():
    """Test that StrEnum preserves case."""
    from computer_use_demo.compat import StrEnum

    class TestEnum(StrEnum):
        MixedCase = "MixedCase"

    assert str(TestEnum.MixedCase) == "MixedCase"
    assert TestEnum.MixedCase.value == "MixedCase"


def test_strenum_empty_string():
    """Test that StrEnum can handle empty string values."""
    from computer_use_demo.compat import StrEnum

    class TestEnum(StrEnum):
        EMPTY = ""

    assert TestEnum.EMPTY.value == ""
    assert str(TestEnum.EMPTY) == ""


def test_strenum_special_characters():
    """Test that StrEnum can handle special characters."""
    from computer_use_demo.compat import StrEnum

    class TestEnum(StrEnum):
        SPECIAL = "value-with_special.chars"

    assert str(TestEnum.SPECIAL) == "value-with_special.chars"


def test_strenum_multiple_inheritance():
    """Test creating multiple StrEnum classes."""
    from computer_use_demo.compat import StrEnum

    class EnumA(StrEnum):
        A = "a"

    class EnumB(StrEnum):
        B = "b"

    assert str(EnumA.A) == "a"
    assert str(EnumB.B) == "b"
    assert EnumA.A != EnumB.B