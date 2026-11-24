# Public free build: retain Python source/bytecode for Torch's inspection and
# Dynamo, and put runtime resources next to the executable as Jasna expects.
import json
import os
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

configuration = json.loads(Path(os.environ["JASNA_PUBLIC_BUILD_CONFIG"]).read_text(encoding="utf-8"))
root = Path(configuration["root"])
datas = [tuple(item) for item in configuration["datas"]]
binaries = [tuple(item) for item in configuration["binaries"]]
hiddenimports = collect_submodules("jasna", filter=lambda name: not name.startswith("jasna.protection"))
hiddenimports += ["torch_tensorrt", "tensorrt", "vlc", "pystray._win32", "PIL._tkinter_finder"]

for package in ("customtkinter", "tkinterdnd2", "torch_tensorrt", "tensorrt_libs", "nvvfx", "python_vali"):
    if find_spec(package) is None:
        continue
    package_datas, package_binaries, package_imports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_imports
for distribution in metadata.distributions():
    datas += copy_metadata(distribution.metadata["Name"])

analysis = Analysis(
    [str(root / "jasna" / "packaging" / "public_entry.py")],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["jasna.protection", "pytest", "tests", "nuitka", "IPython"],
    module_collection_mode={"jasna": "pyz+py", "torch": "pyz+py", "torch_tensorrt": "pyz+py",
                            "triton": "pyz+py", "ultralytics": "pyz+py", "diffusers": "pyz+py"},
)
archive = PYZ(analysis.pure)
executable = EXE(
    archive, analysis.scripts, [],
    exclude_binaries=True,
    name="jasna-free",
    console=True,
    icon=str(root / "assets" / "jasna-logo.ico"),
    contents_directory=".",
)
COLLECT(executable, analysis.binaries, analysis.datas, name="jasna-free")
