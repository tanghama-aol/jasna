"""GUI/CLI entry point for an executable built from the public repository."""
from __future__ import annotations

import multiprocessing
import os
import sys
from importlib import import_module

from jasna import startup_timing  # noqa: F401; record startup before loading the app
from jasna.packaging.public_runtime import COMPILER_TCL_VARIABLES, enable_public_build


def main() -> None:
    enable_public_build()
    os.environ.setdefault("OMP_WAIT_POLICY", "passive")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    multiprocessing.freeze_support()
    if sys.argv[1:2] == ["--compile-engines"]:
        # PyInstaller's Tk runtime hook also sets these in each new process.
        for name in COMPILER_TCL_VARIABLES:
            os.environ.pop(name, None)
    # The existing entry dispatches on import, including --compile-engines, so
    # GUI, CLI and compiler children keep the same argument and PID handling.
    import_module("jasna.__main__")


if __name__ == "__main__":
    main()
