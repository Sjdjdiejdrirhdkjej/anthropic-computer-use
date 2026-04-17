"""Tests for computer_use_demo.tools.collection module."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from computer_use_demo.tools.base import BaseAnthropicTool, ToolError, ToolFailure, ToolResult
from computer_use_demo.tools.collection import ToolCollection


class TestToolCollection:
    """Tests for ToolCollection."""

    def _make_mock_tool(self, name: str) -> MagicMock:
        """Create a mock tool with a given name."""
        tool = MagicMock(spec=BaseAnthropicTool)
        tool.to_params.return_value = {"name": name, "type": f"{name}_type"}
        tool.__call__ = AsyncMock(return_value=ToolResult(output=f"{name} result"))
        return tool

    def test_to_params_returns_list(self):
        """to_params should return a list of tool parameter dicts."""
        tool_a = self._make_mock_tool("tool_a")
        tool_b = self._make_mock_tool("tool_b")
        collection = ToolCollection(tool_a, tool_b)
        params = collection.to_params()

        assert len(params) == 2
        assert params[0] == {"name": "tool_a", "type": "tool_a_type"}
        assert params[1] == {"name": "tool_b", "type": "tool_b_type"}

    def test_tool_map_populated_from_tools(self):
        """tool_map should map tool names to tool instances."""
        tool_a = self._make_mock_tool("tool_a")
        tool_b = self._make_mock_tool("tool_b")
        collection = ToolCollection(tool_a, tool_b)

        assert "tool_a" in collection.tool_map
        assert "tool_b" in collection.tool_map
        assert collection.tool_map["tool_a"] is tool_a

    @pytest.mark.asyncio
    async def test_run_dispatches_to_correct_tool(self):
        """run() should call the tool matching the given name."""
        tool_a = self._make_mock_tool("tool_a")
        tool_b = self._make_mock_tool("tool_b")
        collection = ToolCollection(tool_a, tool_b)

        result = await collection.run(name="tool_a", tool_input={"key": "val"})

        tool_a.__call__.assert_awaited_once_with(key="val")
        tool_b.__call__.assert_not_awaited()
        assert result.output == "tool_a result"

    @pytest.mark.asyncio
    async def test_run_with_invalid_tool_name_returns_failure(self):
        """run() with an unknown tool name should return ToolFailure."""
        tool_a = self._make_mock_tool("tool_a")
        collection = ToolCollection(tool_a)

        result = await collection.run(name="nonexistent", tool_input={})

        assert isinstance(result, ToolFailure)
        assert "nonexistent" in result.error
        assert "invalid" in result.error

    @pytest.mark.asyncio
    async def test_run_catches_tool_error(self):
        """run() should catch ToolError and return ToolFailure."""
        tool = self._make_mock_tool("failing_tool")
        tool.__call__ = AsyncMock(side_effect=ToolError("something broke"))
        collection = ToolCollection(tool)

        result = await collection.run(name="failing_tool", tool_input={})

        assert isinstance(result, ToolFailure)
        assert result.error == "something broke"

    @pytest.mark.asyncio
    async def test_run_does_not_catch_non_tool_errors(self):
        """run() should not catch exceptions other than ToolError."""
        tool = self._make_mock_tool("crashy_tool")
        tool.__call__ = AsyncMock(side_effect=RuntimeError("unexpected"))
        collection = ToolCollection(tool)

        with pytest.raises(RuntimeError, match="unexpected"):
            await collection.run(name="crashy_tool", tool_input={})
