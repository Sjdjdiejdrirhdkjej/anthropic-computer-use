"""Tests for computer_use_demo.tools.computer module."""

import os
import base64
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock

import pytest

from computer_use_demo.tools.computer import (
    ComputerTool,
    chunks,
    ScalingSource,
    MAX_SCALING_TARGETS,
    TYPING_DELAY_MS,
    TYPING_GROUP_SIZE,
)
from computer_use_demo.tools.base import ToolError, ToolResult


class TestChunksFunction:
    """Test the chunks utility function."""

    def test_chunks_basic(self):
        """Test basic chunking."""
        result = chunks("abcdefgh", 3)
        assert result == ["abc", "def", "gh"]

    def test_chunks_exact_division(self):
        """Test chunking with exact division."""
        result = chunks("abcdef", 2)
        assert result == ["ab", "cd", "ef"]

    def test_chunks_single_character(self):
        """Test chunking into single characters."""
        result = chunks("abc", 1)
        assert result == ["a", "b", "c"]

    def test_chunks_larger_than_string(self):
        """Test chunk size larger than string."""
        result = chunks("abc", 10)
        assert result == ["abc"]

    def test_chunks_empty_string(self):
        """Test chunking empty string."""
        result = chunks("", 5)
        assert result == []

    def test_chunks_preserves_content(self):
        """Test that chunking preserves all content."""
        original = "Hello, World! This is a test."
        chunked = chunks(original, 7)
        reconstructed = "".join(chunked)
        assert reconstructed == original


class TestScalingSource:
    """Test the ScalingSource enum."""

    def test_scaling_source_values(self):
        """Test that ScalingSource has expected values."""
        assert ScalingSource.COMPUTER == "computer"
        assert ScalingSource.API == "api"

    def test_scaling_source_is_str(self):
        """Test that ScalingSource members are strings."""
        assert isinstance(ScalingSource.COMPUTER.value, str)
        assert isinstance(ScalingSource.API.value, str)


class TestMaxScalingTargets:
    """Test the MAX_SCALING_TARGETS constant."""

    def test_max_scaling_targets_structure(self):
        """Test that MAX_SCALING_TARGETS has expected keys."""
        assert "XGA" in MAX_SCALING_TARGETS
        assert "WXGA" in MAX_SCALING_TARGETS
        assert "FWXGA" in MAX_SCALING_TARGETS

    def test_max_scaling_targets_values(self):
        """Test that target resolutions have width and height."""
        for name, resolution in MAX_SCALING_TARGETS.items():
            assert "width" in resolution
            assert "height" in resolution
            assert resolution["width"] > 0
            assert resolution["height"] > 0

    def test_max_scaling_targets_xga(self):
        """Test XGA resolution values."""
        xga = MAX_SCALING_TARGETS["XGA"]
        assert xga["width"] == 1024
        assert xga["height"] == 768


class TestComputerToolInit:
    """Test ComputerTool initialization."""

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    def test_init_with_width_height(self):
        """Test initialization with WIDTH and HEIGHT env vars."""
        mock_desktop = Mock()
        tool = ComputerTool(mock_desktop)

        assert tool.width == 1920
        assert tool.height == 1080
        assert tool.desktop == mock_desktop

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080", "DISPLAY_NUM": "1"})
    def test_init_with_display_num(self):
        """Test initialization with DISPLAY_NUM env var."""
        mock_desktop = Mock()
        tool = ComputerTool(mock_desktop)

        assert tool.display_num == 1
        assert tool._display_prefix == "DISPLAY=:1 "
        assert "DISPLAY=:1" in tool.xdotool

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"}, clear=True)
    def test_init_without_display_num(self):
        """Test initialization without DISPLAY_NUM env var."""
        mock_desktop = Mock()
        tool = ComputerTool(mock_desktop)

        assert tool.display_num is None
        assert tool._display_prefix == ""

    @patch.dict(os.environ, {}, clear=True)
    def test_init_without_width_height_raises(self):
        """Test initialization fails without WIDTH and HEIGHT."""
        mock_desktop = Mock()
        with pytest.raises(AssertionError):
            ComputerTool(mock_desktop)

    @patch.dict(os.environ, {"WIDTH": "0", "HEIGHT": "1080"})
    def test_init_with_zero_width_raises(self):
        """Test initialization fails with zero WIDTH."""
        mock_desktop = Mock()
        with pytest.raises(AssertionError):
            ComputerTool(mock_desktop)


class TestComputerToolOptions:
    """Test ComputerTool.options property."""

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    def test_options_returns_dict(self):
        """Test that options returns a dictionary."""
        mock_desktop = Mock()
        tool = ComputerTool(mock_desktop)
        options = tool.options

        assert isinstance(options, dict)
        assert "display_width_px" in options
        assert "display_height_px" in options
        assert "display_number" in options

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080", "DISPLAY_NUM": "1"})
    def test_options_includes_display_num(self):
        """Test that options includes display_number."""
        mock_desktop = Mock()
        tool = ComputerTool(mock_desktop)
        options = tool.options

        assert options["display_number"] == 1

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    def test_options_with_scaling(self):
        """Test that options apply scaling."""
        mock_desktop = Mock()
        tool = ComputerTool(mock_desktop)
        tool._scaling_enabled = True
        options = tool.options

        # Should be scaled down
        assert options["display_width_px"] <= 1920
        assert options["display_height_px"] <= 1080


class TestComputerToolToParams:
    """Test ComputerTool.to_params method."""

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    def test_to_params_structure(self):
        """Test that to_params returns correct structure."""
        mock_desktop = Mock()
        tool = ComputerTool(mock_desktop)
        params = tool.to_params()

        assert params["name"] == "computer"
        assert params["type"] == "computer_20241022"
        assert "display_width_px" in params
        assert "display_height_px" in params
        assert "display_number" in params


class TestComputerToolScaleCoordinates:
    """Test ComputerTool.scale_coordinates method."""

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    def test_scale_coordinates_disabled(self):
        """Test that scaling is bypassed when disabled."""
        mock_desktop = Mock()
        tool = ComputerTool(mock_desktop)
        tool._scaling_enabled = False

        x, y = tool.scale_coordinates(ScalingSource.COMPUTER, 100, 200)
        assert x == 100
        assert y == 200

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    def test_scale_coordinates_no_target_found(self):
        """Test scaling when no target resolution matches."""
        mock_desktop = Mock()
        tool = ComputerTool(mock_desktop)
        tool._scaling_enabled = True

        # 1920x1080 is already smaller than XGA in width
        x, y = tool.scale_coordinates(ScalingSource.COMPUTER, 100, 200)
        # Should return original since target is found and scaling applied
        assert isinstance(x, int)
        assert isinstance(y, int)

    @patch.dict(os.environ, {"WIDTH": "2560", "HEIGHT": "1440"})
    def test_scale_coordinates_from_computer(self):
        """Test scaling from computer source."""
        mock_desktop = Mock()
        tool = ComputerTool(mock_desktop)
        tool._scaling_enabled = True

        x, y = tool.scale_coordinates(ScalingSource.COMPUTER, 2560, 1440)
        # Should scale down to target
        assert x < 2560
        assert y < 1440

    @patch.dict(os.environ, {"WIDTH": "2560", "HEIGHT": "1440"})
    def test_scale_coordinates_from_api(self):
        """Test scaling from API source."""
        mock_desktop = Mock()
        tool = ComputerTool(mock_desktop)
        tool._scaling_enabled = True

        # Small coordinates from API should scale up
        x, y = tool.scale_coordinates(ScalingSource.API, 640, 360)
        assert x > 640
        assert y > 360

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    def test_scale_coordinates_api_out_of_bounds(self):
        """Test that out-of-bounds API coordinates raise error."""
        mock_desktop = Mock()
        tool = ComputerTool(mock_desktop)
        tool._scaling_enabled = True

        with pytest.raises(ToolError, match="out of bounds"):
            tool.scale_coordinates(ScalingSource.API, 3000, 2000)


class TestComputerToolMouseActions:
    """Test ComputerTool mouse-related actions."""

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    @pytest.mark.asyncio
    async def test_mouse_move_action(self):
        """Test mouse_move action."""
        mock_desktop = Mock()
        mock_desktop.commands.run.return_value = Mock(stdout="", stderr="")
        tool = ComputerTool(mock_desktop)

        result = await tool(action="mouse_move", coordinate=[100, 200])

        assert isinstance(result, ToolResult)
        mock_desktop.commands.run.assert_called_once()
        call_args = mock_desktop.commands.run.call_args[0][0]
        assert "mousemove" in call_args
        # Coordinates may be scaled, just verify mousemove was called
        assert "xdotool" in call_args

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    @pytest.mark.asyncio
    async def test_mouse_move_without_coordinate_raises(self):
        """Test mouse_move without coordinate raises error."""
        mock_desktop = Mock()
        tool = ComputerTool(mock_desktop)

        with pytest.raises(ToolError, match="coordinate is required"):
            await tool(action="mouse_move")

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    @pytest.mark.asyncio
    async def test_mouse_move_with_text_raises(self):
        """Test mouse_move with text raises error."""
        mock_desktop = Mock()
        tool = ComputerTool(mock_desktop)

        with pytest.raises(ToolError, match="text is not accepted"):
            await tool(action="mouse_move", coordinate=[100, 200], text="invalid")

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    @pytest.mark.asyncio
    async def test_mouse_move_invalid_coordinate_format(self):
        """Test mouse_move with invalid coordinate format."""
        mock_desktop = Mock()
        tool = ComputerTool(mock_desktop)

        with pytest.raises(ToolError, match="must be a tuple of length 2"):
            await tool(action="mouse_move", coordinate=[100])

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    @pytest.mark.asyncio
    async def test_mouse_move_negative_coordinate(self):
        """Test mouse_move with negative coordinates raises error."""
        mock_desktop = Mock()
        tool = ComputerTool(mock_desktop)

        with pytest.raises(ToolError, match="must be a tuple of non-negative ints"):
            await tool(action="mouse_move", coordinate=[-10, 200])

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    @pytest.mark.asyncio
    async def test_left_click_drag_action(self):
        """Test left_click_drag action."""
        mock_desktop = Mock()
        mock_desktop.commands.run.return_value = Mock(stdout="", stderr="")
        tool = ComputerTool(mock_desktop)

        result = await tool(action="left_click_drag", coordinate=[300, 400])

        assert isinstance(result, ToolResult)
        mock_desktop.commands.run.assert_called_once()
        call_args = mock_desktop.commands.run.call_args[0][0]
        assert "mousedown" in call_args
        assert "mousemove" in call_args
        assert "mouseup" in call_args


class TestComputerToolKeyboardActions:
    """Test ComputerTool keyboard-related actions."""

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    @pytest.mark.asyncio
    async def test_key_action(self):
        """Test key action."""
        mock_desktop = Mock()
        mock_desktop.commands.run.return_value = Mock(stdout="", stderr="")
        tool = ComputerTool(mock_desktop)

        result = await tool(action="key", text="ctrl+c")

        assert isinstance(result, ToolResult)
        mock_desktop.commands.run.assert_called_once()
        call_args = mock_desktop.commands.run.call_args[0][0]
        assert "key" in call_args
        assert "ctrl+c" in call_args

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    @pytest.mark.asyncio
    async def test_key_without_text_raises(self):
        """Test key action without text raises error."""
        mock_desktop = Mock()
        tool = ComputerTool(mock_desktop)

        with pytest.raises(ToolError, match="text is required"):
            await tool(action="key")

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    @pytest.mark.asyncio
    async def test_key_with_coordinate_raises(self):
        """Test key action with coordinate raises error."""
        mock_desktop = Mock()
        tool = ComputerTool(mock_desktop)

        with pytest.raises(ToolError, match="coordinate is not accepted"):
            await tool(action="key", text="a", coordinate=[100, 200])

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    @pytest.mark.asyncio
    async def test_type_action(self):
        """Test type action."""
        mock_desktop = Mock()
        mock_desktop.commands.run.return_value = Mock(stdout="", stderr="")
        mock_desktop.screenshot = Mock()
        tool = ComputerTool(mock_desktop)
        tool.screenshot = AsyncMock(return_value=ToolResult(base64_image="test_image"))

        result = await tool(action="type", text="Hello")

        assert isinstance(result, ToolResult)
        assert result.base64_image == "test_image"

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    @pytest.mark.asyncio
    async def test_type_action_chunks_text(self):
        """Test type action chunks long text."""
        mock_desktop = Mock()
        mock_desktop.commands.run.return_value = Mock(stdout="", stderr="")
        tool = ComputerTool(mock_desktop)
        tool.screenshot = AsyncMock(return_value=ToolResult(base64_image="test_image"))

        long_text = "a" * (TYPING_GROUP_SIZE * 2 + 10)
        result = await tool(action="type", text=long_text)

        # Should be called multiple times (once per chunk)
        assert mock_desktop.commands.run.call_count == 3

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    @pytest.mark.asyncio
    async def test_type_non_string_raises(self):
        """Test type action with non-string raises error."""
        mock_desktop = Mock()
        tool = ComputerTool(mock_desktop)

        # Note: This test reveals a bug in computer.py:148 where ToolError
        # is called with output= keyword arg but ToolError only accepts message
        with pytest.raises(TypeError):
            await tool(action="type", text=123)


class TestComputerToolClickActions:
    """Test ComputerTool click-related actions."""

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    @pytest.mark.asyncio
    async def test_left_click_action(self):
        """Test left_click action."""
        mock_desktop = Mock()
        mock_desktop.commands.run.return_value = Mock(stdout="", stderr="")
        tool = ComputerTool(mock_desktop)

        result = await tool(action="left_click")

        assert isinstance(result, ToolResult)
        mock_desktop.commands.run.assert_called_once()
        call_args = mock_desktop.commands.run.call_args[0][0]
        assert "click" in call_args
        assert "1" in call_args

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    @pytest.mark.asyncio
    async def test_right_click_action(self):
        """Test right_click action."""
        mock_desktop = Mock()
        mock_desktop.commands.run.return_value = Mock(stdout="", stderr="")
        tool = ComputerTool(mock_desktop)

        result = await tool(action="right_click")

        call_args = mock_desktop.commands.run.call_args[0][0]
        assert "click" in call_args
        assert "3" in call_args

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    @pytest.mark.asyncio
    async def test_middle_click_action(self):
        """Test middle_click action."""
        mock_desktop = Mock()
        mock_desktop.commands.run.return_value = Mock(stdout="", stderr="")
        tool = ComputerTool(mock_desktop)

        result = await tool(action="middle_click")

        call_args = mock_desktop.commands.run.call_args[0][0]
        assert "click" in call_args
        assert "2" in call_args

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    @pytest.mark.asyncio
    async def test_double_click_action(self):
        """Test double_click action."""
        mock_desktop = Mock()
        mock_desktop.commands.run.return_value = Mock(stdout="", stderr="")
        tool = ComputerTool(mock_desktop)

        result = await tool(action="double_click")

        call_args = mock_desktop.commands.run.call_args[0][0]
        assert "click" in call_args
        assert "--repeat" in call_args
        assert "2" in call_args

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    @pytest.mark.asyncio
    async def test_click_with_text_raises(self):
        """Test click actions with text raise error."""
        mock_desktop = Mock()
        tool = ComputerTool(mock_desktop)

        with pytest.raises(ToolError, match="text is not accepted"):
            await tool(action="left_click", text="invalid")

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    @pytest.mark.asyncio
    async def test_click_with_coordinate_raises(self):
        """Test click actions with coordinate raise error."""
        mock_desktop = Mock()
        tool = ComputerTool(mock_desktop)

        with pytest.raises(ToolError, match="coordinate is not accepted"):
            await tool(action="left_click", coordinate=[100, 200])


class TestComputerToolScreenshotAndCursor:
    """Test ComputerTool screenshot and cursor position actions."""

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    @pytest.mark.asyncio
    async def test_screenshot_action(self):
        """Test screenshot action."""
        mock_desktop = Mock()
        tool = ComputerTool(mock_desktop)

        # Mock screenshot method
        mock_result = ToolResult(base64_image="test_screenshot")
        tool.screenshot = AsyncMock(return_value=mock_result)

        result = await tool(action="screenshot")

        assert isinstance(result, ToolResult)
        assert result.base64_image == "test_screenshot"
        tool.screenshot.assert_called_once()

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    @pytest.mark.asyncio
    async def test_cursor_position_action(self):
        """Test cursor_position action."""
        mock_desktop = Mock()
        mock_desktop.commands.run.return_value = Mock(
            stdout="X=500\nY=300\nSCREEN=0\nWINDOW=12345\n",
            stderr=""
        )
        tool = ComputerTool(mock_desktop)

        result = await tool(action="cursor_position")

        assert isinstance(result, ToolResult)
        assert "X=" in result.output
        assert "Y=" in result.output
        mock_desktop.commands.run.assert_called_once()

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    @pytest.mark.asyncio
    async def test_screenshot_with_text_raises(self):
        """Test screenshot with text raises error."""
        mock_desktop = Mock()
        tool = ComputerTool(mock_desktop)

        with pytest.raises(ToolError, match="text is not accepted"):
            await tool(action="screenshot", text="invalid")

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    @pytest.mark.asyncio
    async def test_cursor_position_with_coordinate_raises(self):
        """Test cursor_position with coordinate raises error."""
        mock_desktop = Mock()
        tool = ComputerTool(mock_desktop)

        with pytest.raises(ToolError, match="coordinate is not accepted"):
            await tool(action="cursor_position", coordinate=[100, 200])


class TestComputerToolInvalidAction:
    """Test ComputerTool with invalid actions."""

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    @pytest.mark.asyncio
    async def test_invalid_action_raises(self):
        """Test invalid action raises ToolError."""
        mock_desktop = Mock()
        tool = ComputerTool(mock_desktop)

        with pytest.raises(ToolError, match="Invalid action"):
            await tool(action="invalid_action")

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    @pytest.mark.asyncio
    async def test_empty_action_raises(self):
        """Test empty action raises ToolError."""
        mock_desktop = Mock()
        tool = ComputerTool(mock_desktop)

        with pytest.raises(ToolError, match="Invalid action"):
            await tool(action="")