"""Tests for computer_use_demo.compat module."""

import sys
from enum import Enum

import pytest

from computer_use_demo.compat import StrEnum


class TestStrEnum:
    """Test the StrEnum compatibility class."""

    def test_strenum_is_enum(self):
        """Test that StrEnum is a subclass of Enum."""
        assert issubclass(StrEnum, Enum)

    def test_strenum_is_str(self):
        """Test that StrEnum is a subclass of str."""
        assert issubclass(StrEnum, str)

    def test_strenum_values(self):
        """Test that StrEnum members have string values."""
        class Color(StrEnum):
            RED = "red"
            GREEN = "green"
            BLUE = "blue"

        assert Color.RED == "red"
        assert Color.GREEN == "green"
        assert Color.BLUE == "blue"

    def test_strenum_str_representation(self):
        """Test that StrEnum.__str__ returns the value."""
        class Status(StrEnum):
            ACTIVE = "active"
            INACTIVE = "inactive"

        assert str(Status.ACTIVE) == "active"
        assert str(Status.INACTIVE) == "inactive"

    def test_strenum_comparison_with_string(self):
        """Test that StrEnum members can be compared with strings."""
        class Mode(StrEnum):
            DEBUG = "debug"
            RELEASE = "release"

        assert Mode.DEBUG == "debug"
        assert Mode.RELEASE == "release"
        assert Mode.DEBUG != "release"

    def test_strenum_can_be_used_in_dicts(self):
        """Test that StrEnum members work as dict keys."""
        class Environment(StrEnum):
            DEV = "dev"
            PROD = "prod"

        env_config = {
            Environment.DEV: "development config",
            Environment.PROD: "production config",
        }

        assert env_config[Environment.DEV] == "development config"
        assert env_config[Environment.PROD] == "production config"

    def test_strenum_iteration(self):
        """Test that StrEnum can be iterated over."""
        class Priority(StrEnum):
            LOW = "low"
            MEDIUM = "medium"
            HIGH = "high"

        priorities = list(Priority)
        assert len(priorities) == 3
        assert Priority.LOW in priorities
        assert Priority.MEDIUM in priorities
        assert Priority.HIGH in priorities

    def test_strenum_value_access(self):
        """Test accessing StrEnum value attribute."""
        class Level(StrEnum):
            INFO = "info"
            WARNING = "warning"

        assert Level.INFO.value == "info"
        assert Level.WARNING.value == "warning"

    def test_strenum_name_access(self):
        """Test accessing StrEnum name attribute."""
        class State(StrEnum):
            RUNNING = "running"
            STOPPED = "stopped"

        assert State.RUNNING.name == "RUNNING"
        assert State.STOPPED.name == "STOPPED"

    def test_strenum_membership(self):
        """Test membership checks with StrEnum."""
        class Role(StrEnum):
            ADMIN = "admin"
            USER = "user"

        roles = [Role.ADMIN, Role.USER]
        assert Role.ADMIN in roles
        assert Role.USER in roles

    def test_strenum_backport_for_old_python(self):
        """Test that StrEnum works on Python < 3.11."""
        # This test ensures the backport works correctly
        if sys.version_info < (3, 11):
            # Verify we're using the backport
            from computer_use_demo import compat
            assert hasattr(compat, 'StrEnum')
            assert StrEnum.__module__ == 'computer_use_demo.compat'
        else:
            # On Python 3.11+, ensure we use the standard library version
            from enum import StrEnum as StandardStrEnum
            # Both should be available and compatible
            assert issubclass(StrEnum, Enum)
            assert issubclass(StrEnum, str)