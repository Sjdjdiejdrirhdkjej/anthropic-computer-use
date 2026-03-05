"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add computer_use_demo directory to the Python path so DesktopSandbox can be imported
computer_use_demo_dir = project_root / "computer_use_demo"
sys.path.insert(0, str(computer_use_demo_dir))