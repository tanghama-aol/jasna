"""Select the free public build without changing the official release profile."""
from __future__ import annotations

import os

COMPILER_TCL_VARIABLES = ("TCL_LIBRARY", "TK_LIBRARY")
_public_build = False


def enable_public_build() -> None:
    """Use the public profile for this process, including its GUI and CLI workers."""
    global _public_build
    _public_build = True


def is_public_build() -> bool:
    return _public_build


def compiler_environment() -> dict[str, str]:
    """Keep the parent's settings, but not its already initialized Tk paths."""
    environment = os.environ.copy()
    for name in COMPILER_TCL_VARIABLES:
        environment.pop(name, None)
    return environment
