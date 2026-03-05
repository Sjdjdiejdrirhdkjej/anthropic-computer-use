"""Pytest configuration and fixtures for computer_use_demo tests."""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

import pytest

# Add the project root to sys.path so imports work correctly
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "computer_use_demo"))


# Mock streamlit before any imports
if 'streamlit' not in sys.modules:
    mock_st = MagicMock()
    mock_st.delta_generator = MagicMock()
    mock_st.delta_generator.DeltaGenerator = MagicMock()
    sys.modules['streamlit'] = mock_st
    sys.modules['streamlit.delta_generator'] = mock_st.delta_generator


@pytest.fixture(autouse=True)
def mock_desktop_sandbox(monkeypatch):
    """Mock DesktopSandbox for all tests to avoid E2B API calls."""
    # Create a mock DesktopSandbox class
    mock_sandbox_class = MagicMock()
    mock_sandbox_instance = MagicMock()
    mock_sandbox_class.return_value = mock_sandbox_instance

    # Mock the module
    mock_module = MagicMock()
    mock_module.DesktopSandbox = mock_sandbox_class

    # Register the mock in sys.modules
    sys.modules['DesktopSandbox'] = mock_module
    sys.modules['e2b_desktop'] = MagicMock()

    yield mock_sandbox_instance