"""Tests for computer_use_demo.tools.computer module."""

import base64
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import tempfile

import pytest

from computer_use_demo.tools.computer import (
    ComputerTool,
    ScalingSource,
    chunks,
    MAX_SCALING_TARGETS,
    TYPING_DELAY_MS,
    TYPING_GROUP_SIZE,
)
from computer_use_demo.tools.base import ToolError, ToolResult


class TestChunksFunction:
    """Tests for chunks utility function."""

    def test_chunks_basic(self):
        """Test basic chunking of a string."""
        result = chunks("abcdefgh", 3)
        assert result == ["abc", "def", "gh"]

    def test_chunks_exact_division(self):
        """Test chunking when string length divides evenly."""
        result = chunks("abcdef", 2)
        assert result == ["ab", "cd", "ef"]

    def test_chunks_single_chunk(self):
        """Test chunking when chunk size equals string length."""
        result = chunks("hello", 5)
        assert result == ["hello"]

    def test_chunks_larger_chunk_size(self):
        """Test chunking when chunk size is larger than string."""
        result = chunks("hi", 10)
        assert result == ["hi"]

    def test_chunks_single_char(self):
        """Test chunking with single character chunks."""
        result = chunks("abc", 1)
        assert result == ["a", "b", "c"]

    def test_chunks_empty_string(self):
        """Test chunking an empty string."""
        result = chunks("", 5)
        assert result == []

    def test_chunks_with_unicode(self):
        """Test chunking strings with unicode characters."""
        result = chunks("hello😀world", 6)
        assert len(result) == 2
        assert result[0] == "hello😀"


class TestScalingSource:
    """Tests for ScalingSource enum."""

    def test_scaling_source_values(self):
        """Test that ScalingSource has expected values."""
        assert ScalingSource.COMPUTER == "computer"
        assert ScalingSource.API == "api"

    def test_scaling_source_string_behavior(self):
        """Test that ScalingSource members behave like strings."""
        assert str(ScalingSource.COMPUTER) == "computer"
        assert ScalingSource.API == "api"


class TestMaxScalingTargets:
    """Tests for MAX_SCALING_TARGETS constant."""

    def test_max_scaling_targets_structure(self):
        """Test that MAX_SCALING_TARGETS has expected structure."""
        assert "XGA" in MAX_SCALING_TARGETS
        assert "WXGA" in MAX_SCALING_TARGETS
        assert "FWXGA" in MAX_SCALING_TARGETS

    def test_max_scaling_targets_dimensions(self):
        """Test that scaling targets have correct dimensions."""
        xga = MAX_SCALING_TARGETS["XGA"]
        assert xga["width"] == 1024
        assert xga["height"] == 768

        wxga = MAX_SCALING_TARGETS["WXGA"]
        assert wxga["width"] == 1280
        assert wxga["height"] == 800

        fwxga = MAX_SCALING_TARGETS["FWXGA"]
        assert fwxga["width"] == 1366
        assert fwxga["height"] == 768


class TestComputerToolInit:
    """Tests for ComputerTool initialization."""

    def test_computer_tool_init_with_env_vars(self):
        """Test ComputerTool initialization with environment variables."""
        mock_desktop = MagicMock()

        with patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"}):
            tool = ComputerTool(mock_desktop)

            assert tool.width == 1920
            assert tool.height == 1080
            assert tool.desktop == mock_desktop

    def test_computer_tool_init_with_display_num(self):
        """Test ComputerTool initialization with DISPLAY_NUM."""
        mock_desktop = MagicMock()

        with patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080", "DISPLAY_NUM": "1"}):
            tool = ComputerTool(mock_desktop)

            assert tool.display_num == 1
            assert "DISPLAY=:1" in tool._display_prefix

    def test_computer_tool_init_without_display_num(self):
        """Test ComputerTool initialization without DISPLAY_NUM."""
        mock_desktop = MagicMock()

        with patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"}, clear=True):
            if "DISPLAY_NUM" in os.environ:
                del os.environ["DISPLAY_NUM"]
            tool = ComputerTool(mock_desktop)

            assert tool.display_num is None
            assert tool._display_prefix == ""

    def test_computer_tool_init_missing_width_height(self):
        """Test ComputerTool initialization fails without WIDTH and HEIGHT."""
        mock_desktop = MagicMock()

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(AssertionError):
                ComputerTool(mock_desktop)


class TestComputerToolOptions:
    """Tests for ComputerTool options property."""

    def test_options_returns_correct_structure(self):
        """Test that options property returns correct structure."""
        mock_desktop = MagicMock()

        with patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"}):
            tool = ComputerTool(mock_desktop)
            options = tool.options

            assert "display_width_px" in options
            assert "display_height_px" in options
            assert "display_number" in options

    def test_to_params_returns_tool_definition(self):
        """Test that to_params returns proper tool definition."""
        mock_desktop = MagicMock()

        with patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"}):
            tool = ComputerTool(mock_desktop)
            params = tool.to_params()

            assert params["name"] == "computer"
            assert params["type"] == "computer_20241022"
            assert "display_width_px" in params
            assert "display_height_px" in params


class TestComputerToolScaleCoordinates:
    """Tests for scale_coordinates method."""

    def test_scale_coordinates_disabled(self):
        """Test that coordinates are not scaled when scaling is disabled."""
        mock_desktop = MagicMock()

        with patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"}):
            tool = ComputerTool(mock_desktop)
            tool._scaling_enabled = False

            x, y = tool.scale_coordinates(ScalingSource.API, 100, 200)
            assert x == 100
            assert y == 200

    def test_scale_coordinates_no_target_dimension(self):
        """Test scaling when no target dimension matches."""
        mock_desktop = MagicMock()

        # Use dimensions that don't match any target
        with patch.dict(os.environ, {"WIDTH": "800", "HEIGHT": "600"}):
            tool = ComputerTool(mock_desktop)

            x, y = tool.scale_coordinates(ScalingSource.API, 100, 200)
            # Should return original coordinates when no scaling target matches
            assert x == 100
            assert y == 200

    def test_scale_coordinates_from_api(self):
        """Test scaling coordinates from API (scale up)."""
        mock_desktop = MagicMock()

        with patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"}):
            tool = ComputerTool(mock_desktop)

            # Assuming scaling target would be ~1366x768
            x, y = tool.scale_coordinates(ScalingSource.API, 683, 384)
            # Coordinates should be scaled up
            assert x != 683 or y != 384

    def test_scale_coordinates_from_computer(self):
        """Test scaling coordinates from computer (scale down)."""
        mock_desktop = MagicMock()

        with patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"}):
            tool = ComputerTool(mock_desktop)

            x, y = tool.scale_coordinates(ScalingSource.COMPUTER, 1920, 1080)
            # Coordinates should be scaled down or stay same
            assert isinstance(x, int)
            assert isinstance(y, int)

    def test_scale_coordinates_api_out_of_bounds(self):
        """Test that out of bounds coordinates raise error."""
        mock_desktop = MagicMock()

        with patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"}):
            tool = ComputerTool(mock_desktop)

            with pytest.raises(ToolError, match="out of bounds"):
                tool.scale_coordinates(ScalingSource.API, 5000, 5000)


class TestComputerToolActions:
    """Tests for ComputerTool action methods."""

    @pytest.mark.asyncio
    async def test_mouse_move_action(self):
        """Test mouse_move action."""
        mock_desktop = MagicMock()
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_desktop.commands.run.return_value = mock_result

        with patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"}):
            tool = ComputerTool(mock_desktop)
            tool._scaling_enabled = False

            result = await tool(action="mouse_move", coordinate=[100, 200])

            assert isinstance(result, ToolResult)
            mock_desktop.commands.run.assert_called_once()
            call_args = mock_desktop.commands.run.call_args[0][0]
            assert "xdotool" in call_args
            assert "mousemove" in call_args

    @pytest.mark.asyncio
    async def test_mouse_move_requires_coordinate(self):
        """Test that mouse_move requires coordinate."""
        mock_desktop = MagicMock()

        with patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"}):
            tool = ComputerTool(mock_desktop)

            with pytest.raises(ToolError, match="coordinate is required"):
                await tool(action="mouse_move")

    @pytest.mark.asyncio
    async def test_mouse_move_rejects_text(self):
        """Test that mouse_move rejects text parameter."""
        mock_desktop = MagicMock()

        with patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"}):
            tool = ComputerTool(mock_desktop)

            with pytest.raises(ToolError, match="text is not accepted"):
                await tool(action="mouse_move", coordinate=[100, 200], text="invalid")

    @pytest.mark.asyncio
    async def test_key_action(self):
        """Test key action."""
        mock_desktop = MagicMock()
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_desktop.commands.run.return_value = mock_result

        with patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"}):
            tool = ComputerTool(mock_desktop)

            result = await tool(action="key", text="Return")

            assert isinstance(result, ToolResult)
            mock_desktop.commands.run.assert_called_once()
            call_args = mock_desktop.commands.run.call_args[0][0]
            assert "xdotool" in call_args
            assert "key" in call_args

    @pytest.mark.asyncio
    async def test_key_requires_text(self):
        """Test that key action requires text."""
        mock_desktop = MagicMock()

        with patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"}):
            tool = ComputerTool(mock_desktop)

            with pytest.raises(ToolError, match="text is required"):
                await tool(action="key")

    @pytest.mark.asyncio
    async def test_key_rejects_coordinate(self):
        """Test that key action rejects coordinate parameter."""
        mock_desktop = MagicMock()

        with patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"}):
            tool = ComputerTool(mock_desktop)

            with pytest.raises(ToolError, match="coordinate is not accepted"):
                await tool(action="key", text="Return", coordinate=[100, 200])

    @pytest.mark.asyncio
    async def test_left_click_action(self):
        """Test left_click action."""
        mock_desktop = MagicMock()
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_desktop.commands.run.return_value = mock_result

        with patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"}):
            tool = ComputerTool(mock_desktop)

            result = await tool(action="left_click")

            assert isinstance(result, ToolResult)
            call_args = mock_desktop.commands.run.call_args[0][0]
            assert "click" in call_args
            assert "1" in call_args

    @pytest.mark.asyncio
    async def test_right_click_action(self):
        """Test right_click action."""
        mock_desktop = MagicMock()
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_desktop.commands.run.return_value = mock_result

        with patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"}):
            tool = ComputerTool(mock_desktop)

            result = await tool(action="right_click")

            call_args = mock_desktop.commands.run.call_args[0][0]
            assert "click" in call_args
            assert "3" in call_args

    @pytest.mark.asyncio
    async def test_double_click_action(self):
        """Test double_click action."""
        mock_desktop = MagicMock()
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_desktop.commands.run.return_value = mock_result

        with patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"}):
            tool = ComputerTool(mock_desktop)

            result = await tool(action="double_click")

            call_args = mock_desktop.commands.run.call_args[0][0]
            assert "click" in call_args
            assert "repeat" in call_args

    @pytest.mark.asyncio
    async def test_invalid_action(self):
        """Test that invalid action raises error."""
        mock_desktop = MagicMock()

        with patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"}):
            tool = ComputerTool(mock_desktop)

            with pytest.raises(ToolError, match="Invalid action"):
                await tool(action="invalid_action")

    @pytest.mark.asyncio
    async def test_cursor_position_action(self):
        """Test cursor_position action."""
        mock_desktop = MagicMock()
        mock_result = MagicMock()
        mock_result.stdout = "X=100\nY=200\nSCREEN=0\nWINDOW=123"
        mock_result.stderr = ""
        mock_desktop.commands.run.return_value = mock_result

        with patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"}):
            tool = ComputerTool(mock_desktop)
            tool._scaling_enabled = False

            result = await tool(action="cursor_position")

            assert isinstance(result, ToolResult)
            assert "X=100" in result.output
            assert "Y=200" in result.output


class TestComputerToolScreenshot:
    """Tests for screenshot functionality."""

    @pytest.mark.asyncio
    async def test_screenshot_action(self):
        """Test screenshot action."""
        mock_desktop = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("computer_use_demo.tools.computer.OUTPUT_DIR", tmpdir):
                with patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"}):
                    tool = ComputerTool(mock_desktop)
                    tool._scaling_enabled = False

                    # Mock screenshot method to create a file
                    def mock_screenshot(path):
                        Path(path).write_bytes(b"fake_png_data")

                    mock_desktop.screenshot = mock_screenshot

                    result = await tool(action="screenshot")

                    assert isinstance(result, ToolResult)
                    assert result.base64_image is not None
                    assert len(result.base64_image) > 0

    @pytest.mark.asyncio
    async def test_screenshot_creates_output_dir(self):
        """Test that screenshot creates output directory."""
        mock_desktop = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs"
            with patch("computer_use_demo.tools.computer.OUTPUT_DIR", str(output_dir)):
                with patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"}):
                    tool = ComputerTool(mock_desktop)
                    tool._scaling_enabled = False

                    def mock_screenshot(path):
                        Path(path).write_bytes(b"fake_png_data")

                    mock_desktop.screenshot = mock_screenshot

                    await tool.screenshot()

                    assert output_dir.exists()


class TestComputerToolShell:
    """Tests for shell command execution."""

    @pytest.mark.asyncio
    async def test_shell_command_execution(self):
        """Test executing shell commands."""
        mock_desktop = MagicMock()
        mock_result = MagicMock()
        mock_result.stdout = "command output"
        mock_result.stderr = ""
        mock_desktop.commands.run.return_value = mock_result

        with patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"}):
            tool = ComputerTool(mock_desktop)

            result = await tool.shell("echo hello")

            assert isinstance(result, ToolResult)
            assert result.output == "command output"
            mock_desktop.commands.run.assert_called_once_with("echo hello")

    @pytest.mark.asyncio
    async def test_shell_command_with_error(self):
        """Test shell command execution with error."""
        mock_desktop = MagicMock()
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = "error message"
        mock_desktop.commands.run.return_value = mock_result

        with patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"}):
            tool = ComputerTool(mock_desktop)

            result = await tool.shell("invalid_command")

            assert isinstance(result, ToolResult)
            assert result.error == "error message"


class TestComputerToolTypeAction:
    """Tests for type action with chunking."""

    @pytest.mark.asyncio
    async def test_type_action_basic(self):
        """Test basic type action."""
        mock_desktop = MagicMock()
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_desktop.commands.run.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("computer_use_demo.tools.computer.OUTPUT_DIR", tmpdir):
                with patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"}):
                    tool = ComputerTool(mock_desktop)
                    tool._scaling_enabled = False

                    def mock_screenshot(path):
                        Path(path).write_bytes(b"fake_png_data")

                    mock_desktop.screenshot = mock_screenshot

                    result = await tool(action="type", text="hello")

                    assert isinstance(result, ToolResult)
                    assert result.base64_image is not None

    @pytest.mark.asyncio
    async def test_type_action_long_text(self):
        """Test type action with text requiring chunking."""
        mock_desktop = MagicMock()
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_desktop.commands.run.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("computer_use_demo.tools.computer.OUTPUT_DIR", tmpdir):
                with patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"}):
                    tool = ComputerTool(mock_desktop)
                    tool._scaling_enabled = False

                    def mock_screenshot(path):
                        Path(path).write_bytes(b"fake_png_data")

                    mock_desktop.screenshot = mock_screenshot

                    # Text longer than TYPING_GROUP_SIZE
                    long_text = "a" * (TYPING_GROUP_SIZE + 10)
                    result = await tool(action="type", text=long_text)

                    assert isinstance(result, ToolResult)
                    # Should have called xdotool type multiple times
                    assert mock_desktop.commands.run.call_count > 1


class TestComputerToolLeftClickDrag:
    """Tests for left_click_drag action."""

    @pytest.mark.asyncio
    async def test_left_click_drag_action(self):
        """Test left_click_drag action."""
        mock_desktop = MagicMock()
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_desktop.commands.run.return_value = mock_result

        with patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"}):
            tool = ComputerTool(mock_desktop)
            tool._scaling_enabled = False

            result = await tool(action="left_click_drag", coordinate=[500, 600])

            assert isinstance(result, ToolResult)
            call_args = mock_desktop.commands.run.call_args[0][0]
            assert "mousedown" in call_args
            assert "mousemove" in call_args
            assert "mouseup" in call_args

    @pytest.mark.asyncio
    async def test_left_click_drag_requires_coordinate(self):
        """Test that left_click_drag requires coordinate."""
        mock_desktop = MagicMock()

        with patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"}):
            tool = ComputerTool(mock_desktop)

            with pytest.raises(ToolError, match="coordinate is required"):
                await tool(action="left_click_drag")


class TestCoordinateValidation:
    """Tests for coordinate parameter validation."""

    @pytest.mark.asyncio
    async def test_coordinate_must_be_list_of_two(self):
        """Test that coordinate must be a list of length 2."""
        mock_desktop = MagicMock()

        with patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"}):
            tool = ComputerTool(mock_desktop)

            with pytest.raises(ToolError, match="must be a tuple of length 2"):
                await tool(action="mouse_move", coordinate=[100])

            with pytest.raises(ToolError, match="must be a tuple of length 2"):
                await tool(action="mouse_move", coordinate=[100, 200, 300])

    @pytest.mark.asyncio
    async def test_coordinate_must_be_non_negative_ints(self):
        """Test that coordinate values must be non-negative integers."""
        mock_desktop = MagicMock()

        with patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"}):
            tool = ComputerTool(mock_desktop)

            with pytest.raises(ToolError, match="must be a tuple of non-negative ints"):
                await tool(action="mouse_move", coordinate=[-100, 200])

            with pytest.raises(ToolError, match="must be a tuple of non-negative ints"):
                await tool(action="mouse_move", coordinate=[100, -200])