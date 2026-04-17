"""Tests for computer_use_demo.tools.base module."""

from computer_use_demo.tools.base import (
    CLIResult,
    ToolError,
    ToolFailure,
    ToolResult,
)


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_empty_result_is_falsy(self):
        """ToolResult with no fields set should be falsy."""
        result = ToolResult()
        assert not result

    def test_result_with_output_is_truthy(self):
        """ToolResult with output should be truthy."""
        result = ToolResult(output="hello")
        assert result

    def test_result_with_error_is_truthy(self):
        """ToolResult with error should be truthy."""
        result = ToolResult(error="oops")
        assert result

    def test_result_with_base64_image_is_truthy(self):
        """ToolResult with base64_image should be truthy."""
        result = ToolResult(base64_image="data")
        assert result

    def test_result_with_system_is_truthy(self):
        """ToolResult with system should be truthy."""
        result = ToolResult(system="info")
        assert result

    def test_add_combines_outputs(self):
        """Adding two ToolResults should concatenate their outputs."""
        a = ToolResult(output="hello ")
        b = ToolResult(output="world")
        combined = a + b
        assert combined.output == "hello world"

    def test_add_combines_errors(self):
        """Adding two ToolResults should concatenate their errors."""
        a = ToolResult(error="err1\n")
        b = ToolResult(error="err2")
        combined = a + b
        assert combined.error == "err1\nerr2"

    def test_add_combines_system(self):
        """Adding two ToolResults should concatenate their system fields."""
        a = ToolResult(system="sys1\n")
        b = ToolResult(system="sys2")
        combined = a + b
        assert combined.system == "sys1\nsys2"

    def test_add_uses_first_non_none_output(self):
        """When only one side has output, add should use that."""
        a = ToolResult()
        b = ToolResult(output="only")
        combined = a + b
        assert combined.output == "only"

    def test_add_raises_for_conflicting_base64_images(self):
        """Adding two ToolResults both with base64_image should raise ValueError."""
        a = ToolResult(base64_image="img1")
        b = ToolResult(base64_image="img2")
        try:
            _ = a + b
            raise AssertionError("Expected ValueError")
        except ValueError as e:
            assert "Cannot combine" in str(e)

    def test_replace_returns_new_instance(self):
        """replace() should return a new ToolResult with updated fields."""
        original = ToolResult(output="old")
        updated = original.replace(output="new")
        assert updated.output == "new"
        assert original.output == "old"

    def test_frozen_result_cannot_be_assigned(self):
        """ToolResult is frozen, so direct attribute assignment should raise."""
        result = ToolResult(output="hello")
        try:
            result.output = "changed"
            raise AssertionError("Expected FrozenInstanceError")
        except AttributeError:
            pass


class TestCLIResult:
    """Tests for CLIResult subclass."""

    def test_cli_result_inherits_tool_result(self):
        """CLIResult should be a subclass of ToolResult."""
        result = CLIResult(output="output")
        assert isinstance(result, ToolResult)

    def test_cli_result_bool(self):
        """CLIResult should follow the same truthiness rules as ToolResult."""
        assert not CLIResult()
        assert CLIResult(output="data")


class TestToolFailure:
    """Tests for ToolFailure subclass."""

    def test_tool_failure_inherits_tool_result(self):
        """ToolFailure should be a subclass of ToolResult."""
        failure = ToolFailure(error="bad")
        assert isinstance(failure, ToolResult)

    def test_tool_failure_is_truthy(self):
        """ToolFailure with error should be truthy."""
        assert ToolFailure(error="something went wrong")


class TestToolError:
    """Tests for ToolError exception."""

    def test_tool_error_stores_message(self):
        """ToolError should store the message attribute."""
        err = ToolError("test message")
        assert err.message == "test message"

    def test_tool_error_is_exception(self):
        """ToolError should be an Exception subclass."""
        err = ToolError("msg")
        assert isinstance(err, Exception)

    def test_tool_error_can_be_raised_and_caught(self):
        """ToolError should be raisable and catchable."""
        try:
            raise ToolError("boom")
        except ToolError as e:
            assert e.message == "boom"
