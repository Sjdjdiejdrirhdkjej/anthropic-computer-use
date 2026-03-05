"""Compatibility helpers across Python versions."""

import sys
from enum import Enum

if sys.version_info >= (3, 11):
    from enum import StrEnum  # pragma: no cover
else:

    class StrEnum(str, Enum):
        """Backport of ``enum.StrEnum`` for Python < 3.11."""

        def __str__(self) -> str:
            """
            Return the enum member's value as a string.
            
            Returns:
                str: The string representation of the enum member's value.
            """
            return str(self.value)
