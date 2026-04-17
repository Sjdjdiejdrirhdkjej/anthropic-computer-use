"""Tests for computer_use_demo.tools.edit module."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from computer_use_demo.tools.base import CLIResult, ToolError
from computer_use_demo.tools.edit import EditTool, SNIPPET_LINES


@pytest.fixture
def mock_desktop():
    """Create a mock DesktopSandbox."""
    desktop = MagicMock()
    desktop.is_absolute.return_value = True
    desktop.exists.return_value = True
    desktop.is_dir.return_value = False
    return desktop


@pytest.fixture
def edit_tool(mock_desktop):
    """Create an EditTool with a mock desktop."""
    return EditTool(desktop=mock_desktop)


class TestEditToolReadFile:
    """Tests for EditTool.read_file."""

    def test_read_file_success(self, edit_tool, mock_desktop):
        """read_file should return file content from the desktop."""
        expected = "Hello, world!"
        mock_desktop.files.read.return_value = expected
        result = edit_tool.read_file(Path("/path/to/test.txt"))
        assert result == expected
        mock_desktop.files.read.assert_called_once_with(Path("/path/to/test.txt"))

    def test_read_file_error(self, edit_tool, mock_desktop):
        """read_file should raise ToolError when desktop.files.read raises."""
        mock_desktop.files.read.side_effect = PermissionError("denied")
        with pytest.raises(ToolError, match="Ran into denied"):
            edit_tool.read_file(Path("/secret"))

    def test_read_file_empty(self, edit_tool, mock_desktop):
        """read_file should return empty string for empty files."""
        mock_desktop.files.read.return_value = ""
        result = edit_tool.read_file(Path("/empty.txt"))
        assert result == ""


class TestEditToolWriteFile:
    """Tests for EditTool.write_file."""

    def test_write_file_success(self, edit_tool, mock_desktop):
        """write_file should delegate to desktop.files.write."""
        edit_tool.write_file(Path("/out.txt"), "content")
        mock_desktop.files.write.assert_called_once_with(Path("/out.txt"), "content")

    def test_write_file_error(self, edit_tool, mock_desktop):
        """write_file should raise ToolError when desktop.files.write raises."""
        mock_desktop.files.write.side_effect = OSError("disk full")
        with pytest.raises(ToolError, match="Ran into disk full"):
            edit_tool.write_file(Path("/out.txt"), "content")


class TestEditToolValidatePath:
    """Tests for EditTool.validate_path."""

    def test_reject_relative_path(self, edit_tool, mock_desktop):
        """validate_path should reject non-absolute paths."""
        mock_desktop.is_absolute.return_value = False
        with pytest.raises(ToolError, match="not an absolute path"):
            edit_tool.validate_path("view", Path("relative.txt"))

    def test_reject_nonexistent_path_for_non_create(self, edit_tool, mock_desktop):
        """validate_path should reject non-existent paths for non-create commands."""
        mock_desktop.exists.return_value = False
        with pytest.raises(ToolError, match="does not exist"):
            edit_tool.validate_path("view", Path("/missing.txt"))

    def test_allow_nonexistent_path_for_create(self, edit_tool, mock_desktop):
        """validate_path should allow non-existent paths for create command."""
        mock_desktop.exists.return_value = False
        # Should not raise
        edit_tool.validate_path("create", Path("/new.txt"))

    def test_reject_existing_path_for_create(self, edit_tool, mock_desktop):
        """validate_path should reject existing paths for create command."""
        mock_desktop.exists.return_value = True
        with pytest.raises(ToolError, match="File already exists"):
            edit_tool.validate_path("create", Path("/exists.txt"))

    def test_reject_directory_for_non_view(self, edit_tool, mock_desktop):
        """validate_path should reject directories for non-view commands."""
        mock_desktop.is_dir.return_value = True
        with pytest.raises(ToolError, match="directory"):
            edit_tool.validate_path("str_replace", Path("/mydir"))

    def test_allow_directory_for_view(self, edit_tool, mock_desktop):
        """validate_path should allow directories for view command."""
        mock_desktop.is_dir.return_value = True
        # Should not raise
        edit_tool.validate_path("view", Path("/mydir"))


class TestEditToolStrReplace:
    """Tests for EditTool.str_replace."""

    def test_str_replace_success(self, edit_tool, mock_desktop):
        """str_replace should replace old_str with new_str."""
        mock_desktop.files.read.return_value = "hello world"
        result = edit_tool.str_replace(Path("/test.txt"), "hello", "goodbye")
        assert isinstance(result, CLIResult)
        mock_desktop.files.write.assert_called_once_with(Path("/test.txt"), "goodbye world")

    def test_str_replace_no_match(self, edit_tool, mock_desktop):
        """str_replace should raise when old_str is not found."""
        mock_desktop.files.read.return_value = "hello world"
        with pytest.raises(ToolError, match="did not appear verbatim"):
            edit_tool.str_replace(Path("/test.txt"), "xyz", "abc")

    def test_str_replace_multiple_matches(self, edit_tool, mock_desktop):
        """str_replace should raise when old_str appears multiple times."""
        mock_desktop.files.read.return_value = "hello hello"
        with pytest.raises(ToolError, match="Multiple occurrences"):
            edit_tool.str_replace(Path("/test.txt"), "hello", "goodbye")

    def test_str_replace_with_none_new_str(self, edit_tool, mock_desktop):
        """str_replace should treat None new_str as empty string."""
        mock_desktop.files.read.return_value = "hello world"
        result = edit_tool.str_replace(Path("/test.txt"), "hello", None)
        mock_desktop.files.write.assert_called_once_with(Path("/test.txt"), " world")

    def test_str_replace_records_history(self, edit_tool, mock_desktop):
        """str_replace should save the original content to file history."""
        original = "hello world"
        mock_desktop.files.read.return_value = original
        edit_tool.str_replace(Path("/test.txt"), "hello", "goodbye")
        assert Path("/test.txt") in edit_tool._file_history
        assert edit_tool._file_history[Path("/test.txt")] == [original]


class TestEditToolInsert:
    """Tests for EditTool.insert."""

    def test_insert_at_beginning(self, edit_tool, mock_desktop):
        """insert at line 0 should prepend text."""
        mock_desktop.files.read.return_value = "line1\nline2"
        result = edit_tool.insert(Path("/test.txt"), 0, "new_line")
        assert isinstance(result, CLIResult)
        written = mock_desktop.files.write.call_args[0][1]
        assert written == "new_line\nline1\nline2"

    def test_insert_at_end(self, edit_tool, mock_desktop):
        """insert at the last line should append text."""
        mock_desktop.files.read.return_value = "line1\nline2"
        result = edit_tool.insert(Path("/test.txt"), 2, "new_line")
        written = mock_desktop.files.write.call_args[0][1]
        assert written == "line1\nline2\nnew_line"

    def test_insert_invalid_line(self, edit_tool, mock_desktop):
        """insert with out-of-range line should raise ToolError."""
        mock_desktop.files.read.return_value = "line1\nline2"
        with pytest.raises(ToolError, match="Invalid `insert_line`"):
            edit_tool.insert(Path("/test.txt"), 5, "new_line")

    def test_insert_negative_line(self, edit_tool, mock_desktop):
        """insert with negative line should raise ToolError."""
        mock_desktop.files.read.return_value = "line1"
        with pytest.raises(ToolError, match="Invalid `insert_line`"):
            edit_tool.insert(Path("/test.txt"), -1, "new_line")


class TestEditToolUndoEdit:
    """Tests for EditTool.undo_edit."""

    def test_undo_edit_success(self, edit_tool, mock_desktop):
        """undo_edit should restore the previous file content."""
        path = Path("/test.txt")
        original = "original content"
        edit_tool._file_history[path] = [original]
        result = edit_tool.undo_edit(path)
        assert isinstance(result, CLIResult)
        mock_desktop.files.write.assert_called_once_with(path, original)

    def test_undo_edit_no_history(self, edit_tool, mock_desktop):
        """undo_edit should raise when there is no edit history."""
        with pytest.raises(ToolError, match="No edit history"):
            edit_tool.undo_edit(Path("/test.txt"))

    def test_undo_edit_pops_history(self, edit_tool, mock_desktop):
        """undo_edit should remove the last entry from history."""
        path = Path("/test.txt")
        edit_tool._file_history[path] = ["v1", "v2"]
        edit_tool.undo_edit(path)
        assert edit_tool._file_history[path] == ["v1"]


class TestEditToolMakeOutput:
    """Tests for EditTool._make_output."""

    def test_make_output_with_line_numbers(self, edit_tool):
        """_make_output should prepend line numbers to each line."""
        result = edit_tool._make_output("hello\nworld", "test.txt")
        assert "cat -n" in result
        assert "hello" in result
        assert "world" in result

    def test_make_output_with_custom_init_line(self, edit_tool):
        """_make_output should start line numbering from init_line."""
        result = edit_tool._make_output("hello", "test.txt", init_line=5)
        assert "5" in result

    def test_make_output_expands_tabs(self, edit_tool):
        """_make_output should expand tabs by default."""
        result = edit_tool._make_output("a\tb", "test.txt")
        # Tabs should be expanded (default tab stop is 8)
        assert "a       b" in result
