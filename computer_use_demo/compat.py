"""Compatibility helpers across Python versions."""

import sys
from enum import Enum

if sys.version_info >= (3, 11):
    from enum import StrEnum  # pragma: no cover
else:

    class StrEnum(str, Enum):
        """Backport of ``enum.StrEnum`` for Python < 3.11."""

        def __str__(self) -> str:
            return str(self.value)
