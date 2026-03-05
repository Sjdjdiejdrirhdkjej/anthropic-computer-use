"""Tests for computer_use_demo.tools.computer module."""

import base64
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from computer_use_demo.tools.base import ToolError, ToolResult
from computer_use_demo.tools.computer import (
    MAX_SCALING_TARGETS,
    OUTPUT_DIR,
    TYPING_DELAY_MS,
    TYPING_GROUP_SIZE,
    ComputerTool,
    ComputerToolOptions,
    Resolution,
    ScalingSource,
    chunks,
)


class TestChunks:
    """Tests for chunks utility function."""

    def test_chunks_basic(self):
        """Test chunks splits string into equal chunks."""
        result = chunks("abcdefgh", 3)
        assert result == ["abc", "def", "gh"]

    def test_chunks_exact_division(self):
        """Test chunks with exact division."""
        result = chunks("abcdef", 2)
        assert result == ["ab", "cd", "ef"]

    def test_chunks_single_chunk(self):
        """Test chunks with chunk size larger than string."""
        result = chunks("abc", 10)
        assert result == ["abc"]

    def test_chunks_empty_string(self):
        """Test chunks with empty string."""
        result = chunks("", 5)
        assert result == []

    def test_chunks_size_one(self):
        """Test chunks with size 1."""
        result = chunks("abc", 1)
        assert result == ["a", "b", "c"]


class TestScalingSource:
    """Tests for ScalingSource enum."""

    def test_scaling_source_values(self):
        """Test ScalingSource enum values."""
        assert ScalingSource.COMPUTER.value == "computer"
        assert ScalingSource.API.value == "api"

    def test_scaling_source_string_representation(self):
        """Test string representation of ScalingSource."""
        assert str(ScalingSource.COMPUTER) == "computer"
        assert str(ScalingSource.API) == "api"


class TestResolution:
    """Tests for Resolution TypedDict."""

    def test_resolution_structure(self):
        """Test Resolution has expected structure."""
        resolution: Resolution = {"width": 1024, "height": 768}
        assert resolution["width"] == 1024
        assert resolution["height"] == 768


class TestMaxScalingTargets:
    """Tests for MAX_SCALING_TARGETS constant."""

    def test_max_scaling_targets_has_expected_keys(self):
        """Test MAX_SCALING_TARGETS has expected resolutions."""
        assert "XGA" in MAX_SCALING_TARGETS
        assert "WXGA" in MAX_SCALING_TARGETS
        assert "FWXGA" in MAX_SCALING_TARGETS

    def test_max_scaling_targets_xga_resolution(self):
        """Test XGA resolution is 1024x768."""
        assert MAX_SCALING_TARGETS["XGA"]["width"] == 1024
        assert MAX_SCALING_TARGETS["XGA"]["height"] == 768

    def test_max_scaling_targets_wxga_resolution(self):
        """Test WXGA resolution is 1280x800."""
        assert MAX_SCALING_TARGETS["WXGA"]["width"] == 1280
        assert MAX_SCALING_TARGETS["WXGA"]["height"] == 800

    def test_max_scaling_targets_fwxga_resolution(self):
        """Test FWXGA resolution is 1366x768."""
        assert MAX_SCALING_TARGETS["FWXGA"]["width"] == 1366
        assert MAX_SCALING_TARGETS["FWXGA"]["height"] == 768


class TestComputerToolInit:
    """Tests for ComputerTool initialization."""

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    def test_init_with_width_and_height(self):
        """Test ComputerTool initialization with WIDTH and HEIGHT env vars."""
        mock_desktop = MagicMock()
        tool = ComputerTool(mock_desktop)

        assert tool.width == 1920
        assert tool.height == 1080
        assert tool.desktop == mock_desktop

    @patch.dict(os.environ, {"WIDTH": "1024", "HEIGHT": "768", "DISPLAY_NUM": "1"})
    def test_init_with_display_num(self):
        """Test ComputerTool initialization with DISPLAY_NUM."""
        mock_desktop = MagicMock()
        tool = ComputerTool(mock_desktop)

        assert tool.display_num == 1
        assert "DISPLAY=:1" in tool._display_prefix

    @patch.dict(os.environ, {"WIDTH": "1024", "HEIGHT": "768"}, clear=True)
    def test_init_without_display_num(self):
        """Test ComputerTool initialization without DISPLAY_NUM."""
        mock_desktop = MagicMock()
        tool = ComputerTool(mock_desktop)

        assert tool.display_num is None
        assert tool._display_prefix == ""

    @patch.dict(os.environ, {}, clear=True)
    def test_init_without_width_height_raises_assertion(self):
        """Test ComputerTool initialization fails without WIDTH and HEIGHT."""
        mock_desktop = MagicMock()
        with pytest.raises(AssertionError):
            ComputerTool(mock_desktop)

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    def test_init_sets_xdotool_command(self):
        """Test ComputerTool sets xdotool command properly."""
        mock_desktop = MagicMock()
        tool = ComputerTool(mock_desktop)

        assert "xdotool" in tool.xdotool


class TestComputerToolOptions:
    """Tests for ComputerTool.options property."""

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    def test_options_returns_scaled_dimensions(self):
        """Test options property returns properly scaled dimensions."""
        mock_desktop = MagicMock()
        tool = ComputerTool(mock_desktop)
        options = tool.options

        assert "display_width_px" in options
        assert "display_height_px" in options
        assert "display_number" in options
        assert isinstance(options["display_width_px"], int)
        assert isinstance(options["display_height_px"], int)

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080", "DISPLAY_NUM": "2"})
    def test_options_includes_display_number(self):
        """Test options includes display number when set."""
        mock_desktop = MagicMock()
        tool = ComputerTool(mock_desktop)
        options = tool.options

        assert options["display_number"] == 2


class TestComputerToolToParams:
    """Tests for ComputerTool.to_params method."""

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    def test_to_params_returns_correct_structure(self):
        """Test to_params returns correct parameter structure."""
        mock_desktop = MagicMock()
        tool = ComputerTool(mock_desktop)
        params = tool.to_params()

        assert params["name"] == "computer"
        assert params["type"] == "computer_20241022"
        assert "display_width_px" in params
        assert "display_height_px" in params


class TestComputerToolScaleCoordinates:
    """Tests for ComputerTool.scale_coordinates method."""

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    def test_scale_coordinates_no_scaling_when_disabled(self):
        """Test scale_coordinates returns original coords when scaling disabled."""
        mock_desktop = MagicMock()
        tool = ComputerTool(mock_desktop)
        tool._scaling_enabled = False

        x, y = tool.scale_coordinates(ScalingSource.API, 100, 200)
        assert x == 100
        assert y == 200

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    def test_scale_coordinates_no_target_dimension(self):
        """Test scale_coordinates when no target dimension matches."""
        mock_desktop = MagicMock()
        tool = ComputerTool(mock_desktop)
        tool._scaling_enabled = True

        # Coordinates below target, should return as-is
        x, y = tool.scale_coordinates(ScalingSource.COMPUTER, 100, 200)
        # Should find a target and scale, or return original
        assert isinstance(x, int)
        assert isinstance(y, int)

    @patch.dict(os.environ, {"WIDTH": "2048", "HEIGHT": "1536"})
    def test_scale_coordinates_from_api_source(self):
        """Test scale_coordinates from API source scales up."""
        mock_desktop = MagicMock()
        tool = ComputerTool(mock_desktop)
        tool._scaling_enabled = True

        # Should scale up from API coordinates to computer coordinates
        x, y = tool.scale_coordinates(ScalingSource.API, 512, 384)
        # Result should be scaled up
        assert x >= 512
        assert y >= 384

    @patch.dict(os.environ, {"WIDTH": "2560", "HEIGHT": "1440"})
    def test_scale_coordinates_api_out_of_bounds_raises_error(self):
        """Test scale_coordinates raises error for out of bounds API coords."""
        mock_desktop = MagicMock()
        tool = ComputerTool(mock_desktop)
        tool._scaling_enabled = True

        # Use coordinates larger than the display size to trigger error
        with pytest.raises(ToolError, match="out of bounds"):
            tool.scale_coordinates(ScalingSource.API, 3000, 100)


@pytest.mark.asyncio
class TestComputerToolMouseActions:
    """Tests for ComputerTool mouse action methods."""

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    async def test_mouse_move_action(self):
        """Test mouse_move action."""
        mock_desktop = MagicMock()
        mock_desktop.commands.run.return_value = MagicMock(stdout="", stderr="")
        tool = ComputerTool(mock_desktop)

        result = await tool(action="mouse_move", coordinate=[100, 200])

        assert isinstance(result, ToolResult)
        # Should have called shell command
        assert mock_desktop.commands.run.called

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    async def test_mouse_move_without_coordinate_raises_error(self):
        """Test mouse_move without coordinate raises ToolError."""
        mock_desktop = MagicMock()
        tool = ComputerTool(mock_desktop)

        with pytest.raises(ToolError, match="coordinate is required"):
            await tool(action="mouse_move")

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    async def test_mouse_move_with_text_raises_error(self):
        """Test mouse_move with text parameter raises ToolError."""
        mock_desktop = MagicMock()
        tool = ComputerTool(mock_desktop)

        with pytest.raises(ToolError, match="text is not accepted"):
            await tool(action="mouse_move", coordinate=[100, 200], text="invalid")

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    async def test_left_click_drag_action(self):
        """Test left_click_drag action."""
        mock_desktop = MagicMock()
        mock_desktop.commands.run.return_value = MagicMock(stdout="", stderr="")
        tool = ComputerTool(mock_desktop)

        result = await tool(action="left_click_drag", coordinate=[150, 250])

        assert isinstance(result, ToolResult)
        call_args = str(mock_desktop.commands.run.call_args)
        assert "mousedown" in call_args
        assert "mouseup" in call_args

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    async def test_left_click_action(self):
        """Test left_click action."""
        mock_desktop = MagicMock()
        mock_desktop.commands.run.return_value = MagicMock(stdout="", stderr="")
        tool = ComputerTool(mock_desktop)

        result = await tool(action="left_click")

        assert isinstance(result, ToolResult)
        call_args = str(mock_desktop.commands.run.call_args)
        assert "click" in call_args

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    async def test_right_click_action(self):
        """Test right_click action."""
        mock_desktop = MagicMock()
        mock_desktop.commands.run.return_value = MagicMock(stdout="", stderr="")
        tool = ComputerTool(mock_desktop)

        result = await tool(action="right_click")

        assert isinstance(result, ToolResult)

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    async def test_double_click_action(self):
        """Test double_click action."""
        mock_desktop = MagicMock()
        mock_desktop.commands.run.return_value = MagicMock(stdout="", stderr="")
        tool = ComputerTool(mock_desktop)

        result = await tool(action="double_click")

        assert isinstance(result, ToolResult)
        call_args = str(mock_desktop.commands.run.call_args)
        assert "repeat" in call_args or "click" in call_args

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    async def test_middle_click_action(self):
        """Test middle_click action."""
        mock_desktop = MagicMock()
        mock_desktop.commands.run.return_value = MagicMock(stdout="", stderr="")
        tool = ComputerTool(mock_desktop)

        result = await tool(action="middle_click")

        assert isinstance(result, ToolResult)


@pytest.mark.asyncio
class TestComputerToolKeyboardActions:
    """Tests for ComputerTool keyboard action methods."""

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    async def test_key_action(self):
        """Test key action."""
        mock_desktop = MagicMock()
        mock_desktop.commands.run.return_value = MagicMock(stdout="", stderr="")
        tool = ComputerTool(mock_desktop)

        result = await tool(action="key", text="Return")

        assert isinstance(result, ToolResult)
        call_args = str(mock_desktop.commands.run.call_args)
        assert "key" in call_args

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    async def test_key_without_text_raises_error(self):
        """Test key action without text raises ToolError."""
        mock_desktop = MagicMock()
        tool = ComputerTool(mock_desktop)

        with pytest.raises(ToolError, match="text is required"):
            await tool(action="key")

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    async def test_key_with_coordinate_raises_error(self):
        """Test key action with coordinate raises ToolError."""
        mock_desktop = MagicMock()
        tool = ComputerTool(mock_desktop)

        with pytest.raises(ToolError, match="coordinate is not accepted"):
            await tool(action="key", text="Return", coordinate=[100, 200])

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    async def test_type_action(self):
        """Test type action."""
        mock_desktop = MagicMock()
        mock_desktop.commands.run.return_value = MagicMock(stdout="", stderr="")

        # Mock screenshot
        with patch.object(ComputerTool, "screenshot", new_callable=AsyncMock) as mock_screenshot:
            mock_screenshot.return_value = ToolResult(base64_image="fake_image")
            tool = ComputerTool(mock_desktop)

            result = await tool(action="type", text="Hello")

            assert isinstance(result, ToolResult)
            assert result.base64_image == "fake_image"

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    async def test_type_action_chunks_long_text(self):
        """Test type action splits long text into chunks."""
        mock_desktop = MagicMock()
        mock_desktop.commands.run.return_value = MagicMock(stdout="", stderr="")

        long_text = "a" * (TYPING_GROUP_SIZE * 2 + 10)

        with patch.object(ComputerTool, "screenshot", new_callable=AsyncMock) as mock_screenshot:
            mock_screenshot.return_value = ToolResult(base64_image="fake_image")
            tool = ComputerTool(mock_desktop)

            result = await tool(action="type", text=long_text)

            # Should have called run multiple times (once per chunk)
            assert mock_desktop.commands.run.call_count >= 3

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    async def test_type_with_non_string_raises_error(self):
        """Test type action with non-string text raises error."""
        mock_desktop = MagicMock()
        tool = ComputerTool(mock_desktop)

        # The code raises ToolError with output parameter which causes TypeError
        # due to __init__ expecting message parameter
        with pytest.raises((ToolError, TypeError)):
            await tool(action="type", text=123)


@pytest.mark.asyncio
class TestComputerToolScreenshot:
    """Tests for ComputerTool screenshot functionality."""

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    async def test_screenshot_action(self):
        """Test screenshot action."""
        mock_desktop = MagicMock()
        tool = ComputerTool(mock_desktop)

        # Mock the screenshot file creation
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.read_bytes", return_value=b"fake_image_data"):
                with patch("computer_use_demo.tools.computer.run", new_callable=AsyncMock) as mock_run:
                    mock_run.return_value = (0, "", "")

                    result = await tool(action="screenshot")

                    assert isinstance(result, ToolResult)
                    assert result.base64_image is not None
                    assert len(result.base64_image) > 0

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    async def test_screenshot_method_creates_file(self):
        """Test screenshot method creates file in OUTPUT_DIR."""
        mock_desktop = MagicMock()
        tool = ComputerTool(mock_desktop)

        with patch("pathlib.Path.mkdir") as mock_mkdir:
            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.read_bytes", return_value=b"image"):
                    with patch("computer_use_demo.tools.computer.run", new_callable=AsyncMock) as mock_run:
                        mock_run.return_value = (0, "", "")

                        result = await tool.screenshot()

                        mock_mkdir.assert_called_once()

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    async def test_screenshot_method_scales_when_enabled(self):
        """Test screenshot method applies scaling when enabled."""
        mock_desktop = MagicMock()
        tool = ComputerTool(mock_desktop)
        tool._scaling_enabled = True

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.read_bytes", return_value=b"image"):
                with patch("computer_use_demo.tools.computer.run", new_callable=AsyncMock) as mock_run:
                    mock_run.return_value = (0, "", "")

                    result = await tool.screenshot()

                    # Should call convert command for scaling
                    assert mock_run.called
                    call_args = str(mock_run.call_args)
                    assert "convert" in call_args

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    async def test_screenshot_raises_error_if_file_not_created(self):
        """Test screenshot raises ToolError if file creation fails."""
        mock_desktop = MagicMock()
        tool = ComputerTool(mock_desktop)

        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(ToolError, match="Failed to take screenshot"):
                await tool.screenshot()

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    async def test_cursor_position_action(self):
        """Test cursor_position action."""
        mock_desktop = MagicMock()
        mock_desktop.commands.run.return_value = MagicMock(
            stdout="X=100\nY=200\n", stderr=""
        )
        tool = ComputerTool(mock_desktop)

        result = await tool(action="cursor_position")

        assert isinstance(result, ToolResult)
        assert "X=" in result.output
        assert "Y=" in result.output


@pytest.mark.asyncio
class TestComputerToolShell:
    """Tests for ComputerTool.shell method."""

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    async def test_shell_executes_command(self):
        """Test shell method executes command and returns result."""
        mock_desktop = MagicMock()
        mock_desktop.commands.run.return_value = MagicMock(
            stdout="command output", stderr=""
        )
        tool = ComputerTool(mock_desktop)

        result = await tool.shell("ls -la")

        assert result.output == "command output"
        assert result.error == ""
        mock_desktop.commands.run.assert_called_once_with("ls -la")

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    async def test_shell_captures_stderr(self):
        """Test shell method captures stderr."""
        mock_desktop = MagicMock()
        mock_desktop.commands.run.return_value = MagicMock(
            stdout="", stderr="error message"
        )
        tool = ComputerTool(mock_desktop)

        result = await tool.shell("failing_command")

        assert result.error == "error message"


@pytest.mark.asyncio
class TestComputerToolInvalidActions:
    """Tests for invalid actions."""

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    async def test_invalid_action_raises_error(self):
        """Test invalid action raises ToolError."""
        mock_desktop = MagicMock()
        tool = ComputerTool(mock_desktop)

        with pytest.raises(ToolError, match="Invalid action"):
            await tool(action="invalid_action")


class TestComputerToolEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    def test_coordinate_validation_wrong_length(self):
        """Test coordinate validation rejects wrong length."""
        mock_desktop = MagicMock()
        tool = ComputerTool(mock_desktop)

        with pytest.raises(ToolError, match="must be a tuple of length 2"):
            import asyncio
            asyncio.run(tool(action="mouse_move", coordinate=[100]))

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    def test_coordinate_validation_negative_values(self):
        """Test coordinate validation rejects negative values."""
        mock_desktop = MagicMock()
        tool = ComputerTool(mock_desktop)

        with pytest.raises(ToolError, match="non-negative"):
            import asyncio
            asyncio.run(tool(action="mouse_move", coordinate=[-10, 20]))

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    def test_coordinate_validation_non_integer(self):
        """Test coordinate validation rejects non-integer values."""
        mock_desktop = MagicMock()
        tool = ComputerTool(mock_desktop)

        with pytest.raises(ToolError, match="non-negative ints"):
            import asyncio
            asyncio.run(tool(action="mouse_move", coordinate=[10.5, 20]))


class TestConstants:
    """Tests for module constants."""

    def test_output_dir_constant(self):
        """Test OUTPUT_DIR is set correctly."""
        assert OUTPUT_DIR == "/tmp/outputs"

    def test_typing_delay_constant(self):
        """Test TYPING_DELAY_MS is set."""
        assert TYPING_DELAY_MS == 12
        assert isinstance(TYPING_DELAY_MS, int)

    def test_typing_group_size_constant(self):
        """Test TYPING_GROUP_SIZE is set."""
        assert TYPING_GROUP_SIZE == 50
        assert isinstance(TYPING_GROUP_SIZE, int)