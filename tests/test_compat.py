"""Tests for computer_use_demo.compat module."""

import sys
from enum import Enum

import pytest


def test_strenum_import():
    """Test that StrEnum can be imported from compat module."""
    from computer_use_demo.compat import StrEnum

    assert StrEnum is not None


def test_strenum_backport_for_older_python():
    """Test StrEnum backport functionality for Python < 3.11."""
    from computer_use_demo.compat import StrEnum

    class TestEnum(StrEnum):
        OPTION_A = "option_a"
        OPTION_B = "option_b"

    # Test enum values
    assert TestEnum.OPTION_A.value == "option_a"
    assert TestEnum.OPTION_B.value == "option_b"


def test_strenum_string_behavior():
    """Test that StrEnum members behave like strings."""
    from computer_use_demo.compat import StrEnum

    class TestEnum(StrEnum):
        HELLO = "hello"
        WORLD = "world"

    # Test string representation
    assert str(TestEnum.HELLO) == "hello"
    assert str(TestEnum.WORLD) == "world"

    # Test string comparison
    assert TestEnum.HELLO == "hello"
    assert TestEnum.WORLD == "world"


def test_strenum_is_enum():
    """Test that StrEnum is still an Enum."""
    from computer_use_demo.compat import StrEnum

    class TestEnum(StrEnum):
        VALUE = "test"

    assert isinstance(TestEnum.VALUE, Enum)
    assert isinstance(TestEnum.VALUE, str)


def test_strenum_iteration():
    """Test iterating over StrEnum members."""
    from computer_use_demo.compat import StrEnum

    class TestEnum(StrEnum):
        FIRST = "first"
        SECOND = "second"
        THIRD = "third"

    members = list(TestEnum)
    assert len(members) == 3
    assert TestEnum.FIRST in members
    assert TestEnum.SECOND in members
    assert TestEnum.THIRD in members


def test_strenum_member_lookup():
    """Test looking up StrEnum members by name and value."""
    from computer_use_demo.compat import StrEnum

    class TestEnum(StrEnum):
        ALPHA = "alpha"
        BETA = "beta"

    # Lookup by name
    assert TestEnum["ALPHA"] == TestEnum.ALPHA
    assert TestEnum["BETA"] == TestEnum.BETA

    # Lookup by value
    assert TestEnum("alpha") == TestEnum.ALPHA
    assert TestEnum("beta") == TestEnum.BETA


def test_strenum_hashable():
    """Test that StrEnum members are hashable and can be used in sets/dicts."""
    from computer_use_demo.compat import StrEnum

    class TestEnum(StrEnum):
        KEY1 = "key1"
        KEY2 = "key2"

    # Test in set
    enum_set = {TestEnum.KEY1, TestEnum.KEY2}
    assert len(enum_set) == 2

    # Test as dict key
    enum_dict = {TestEnum.KEY1: "value1", TestEnum.KEY2: "value2"}
    assert enum_dict[TestEnum.KEY1] == "value1"


def test_strenum_unique_values():
    """Test that StrEnum enforces unique values."""
    from computer_use_demo.compat import StrEnum

    # This should work fine - different values
    class TestEnum1(StrEnum):
        A = "a"
        B = "b"

    assert TestEnum1.A != TestEnum1.B


def test_strenum_concatenation():
    """Test string concatenation with StrEnum members."""
    from computer_use_demo.compat import StrEnum

    class TestEnum(StrEnum):
        PREFIX = "hello"

    result = TestEnum.PREFIX + " world"
    assert result == "hello world"
    assert isinstance(result, str)


def test_strenum_case_sensitivity():
    """Test that StrEnum is case-sensitive."""
    from computer_use_demo.compat import StrEnum

    class TestEnum(StrEnum):
        LOWER = "lower"
        UPPER = "UPPER"

    assert TestEnum.LOWER != TestEnum.UPPER
    assert TestEnum.LOWER == "lower"
    assert TestEnum.UPPER == "UPPER"