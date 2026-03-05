"""Pytest configuration and fixtures."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock DesktopSandbox before any modules try to import it
sys.modules["DesktopSandbox"] = MagicMock()

# Add the parent directory to sys.path so we can import computer_use_demo
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))