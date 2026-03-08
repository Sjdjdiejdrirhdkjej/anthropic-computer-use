from pathlib import Path
from unittest.mock import MagicMock

import pytest

from computer_use_demo.tools.base import ToolError
from computer_use_demo.tools.edit import EditTool


@pytest.fixture
def mock_desktop():
    desktop = MagicMock()
    return desktop


@pytest.fixture
def edit_tool(mock_desktop):
    return EditTool(desktop=mock_desktop)


def test_read_file_success(edit_tool, mock_desktop):
    """Test read_file returns the content successfully."""
    expected_content = "Hello, world!"
    mock_desktop.files.read.return_value = expected_content
    test_path = Path("/path/to/test.txt")

    result = edit_tool.read_file(test_path)

    assert result == expected_content
    mock_desktop.files.read.assert_called_once_with(test_path)


def test_read_file_error(edit_tool, mock_desktop):
    """Test read_file raises ToolError when an exception occurs."""
    error_message = "Permission denied"
    mock_desktop.files.read.side_effect = Exception(error_message)
    test_path = Path("/path/to/test.txt")

    with pytest.raises(ToolError) as exc_info:
        edit_tool.read_file(test_path)

    assert str(exc_info.value) == f"Ran into {error_message} while trying to read {test_path}"
    mock_desktop.files.read.assert_called_once_with(test_path)
