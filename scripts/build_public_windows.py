"""Build a Windows/NVIDIA executable using only the public application source."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]


class BuildError(RuntimeError):
    pass


def validate_dependencies(project: dict) -> dict[str, str]:
    try:
        from packaging.requirements import Requirement
    except ImportError as exc:
        raise BuildError("Install the project's public-nvidia and public-build extras first.") from exc
    extras = project["optional-dependencies"]
    requirements = project["dependencies"] + extras["public-nvidia"] + extras["public-build"]
    versions = {}
    errors = []
    for text in requirements:
        requirement = Requirement(text)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        try:
            version = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            errors.append(f"{requirement}: missing")
            continue
        if not requirement.specifier.contains(version, prereleases=True):
            errors.append(f"{requirement}: installed {version}")
        versions[requirement.name] = version
    if errors:
        raise BuildError("Dependency check failed:\n  " + "\n  ".join(errors)
                         + '\nInstall with: uv pip install -e ".[public-nvidia,public-build]" '
                         + "--extra-index-url https://download.pytorch.org/whl/cu130")
    return versions


def checked_tool(name: str, directory: Path | None) -> tuple[Path, str]:
    executable = directory / f"{name}.exe" if directory else shutil.which(name)
    if not executable or not Path(executable).is_file():
        raise BuildError(f"{name}.exe is missing; pass --ffmpeg-dir with FFmpeg 8's bin directory.")
    executable = Path(executable).resolve()
    result = subprocess.run([str(executable), "-version"], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=30)
    output = result.stdout + result.stderr
    match = re.search(rf"(?m)^{name} version (?:n)?(\d+)(?:[.\s-]|$)", output)
    major = int(match[1]) if match is not None else None
    if major is None or major > 100:  # nightly builds name a date or git revision
        library = re.search(r"(?m)^\s*libavutil\s+(\d+)\s*[.]", output)
        major = int(library[1]) - 52 if library is not None else None
    if result.returncode or major != 8:
        raise BuildError(f"{executable} must be FFmpeg major version 8.")
    return executable, output.splitlines()[0]


def find_vlc(directory: Path | None, without_vlc: bool) -> Path | None:
    if without_vlc:
        return None
    if directory is None:
        directory = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "VideoLAN" / "VLC"
    directory = directory.resolve()
    required = (directory / "libvlc.dll", directory / "libvlccore.dll", directory / "plugins")
    if not all(path.exists() for path in required):
        raise BuildError("64-bit libVLC 3 is missing; pass --vlc-dir, or --without-vlc for a build "
                         "without player audio.")
    return directory


def build_inputs(root: Path, ffmpeg: Path, ffprobe: Path, vlc: Path | None) -> tuple[list, list]:
    datas = [(str(root / name), ".") for name in ("LICENSE", "LICENSING.md", "NOTICE", "RELEASE_SOURCES.md")]
    binaries = [(str(ffmpeg), "tools"), (str(ffprobe), "tools")]
    for path in sorted((root / "assets").rglob("*")):
        if path.is_file() and "benchmark" not in path.parts and path.suffix in {".png", ".ico", ".svg", ".md", ".txt"}:
            datas.append((str(path), path.parent.relative_to(root).as_posix()))
    for pattern in ("*.fatbin", "*.hipfb"):
        datas.extend((str(path), ".") for path in sorted((root / "jasna" / "media").glob(pattern)))
    for directory in (root / "jasna" / "media" / "miopen", root / "jasna" / "ltx" / "native"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and (path.suffix in {".txt", ".cpp", ".cu", ".cuh", ".h", ".hip"} or path.name == "NOTICE"):
                datas.append((str(path), path.parent.relative_to(root).as_posix()))
    if vlc is not None:
        for path in sorted(vlc.rglob("*")):
            if not path.is_file() or path.suffix.lower() == ".exe":
                continue
            target = (Path("vlc") / path.parent.relative_to(vlc)).as_posix()
            (binaries if path.suffix.lower() == ".dll" else datas).append((str(path), target))
    return datas, binaries


def copy_free_models(source: Path | None, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if source is None:
        return
    # Ship public detector checkpoints and BasicVSR++ weights, never activation
    # state, encrypted supporter models or GPU-specific cached engines.
    for pattern in ("*.pt", "rfdetr-*.onnx", "lada_mosaic_restoration_model_generic_v*.pth"):
        for path in sorted(source.glob(pattern)):
            if path.is_file():
                shutil.copy2(path, destination / path.name)


def smoke_test(executable: Path, work: Path) -> None:
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "JASNA_MAIN_PID"):
        environment.pop(name, None)
    checks = ((["--version"], 0, True), (["--help"], 0, True), (["--license-code", "unavailable"], 1, True),
              (["--compile-engines", json.dumps({"device": "cuda:0", "fp16": True})], 0, False))
    for arguments, expected_code, needs_output in checks:
        result = subprocess.run([str(executable), *arguments], cwd=work, env=environment,
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        output = result.stdout + result.stderr
        if result.returncode != expected_code or "Traceback" in output or (needs_output and not output.strip()):
            raise BuildError(f"Executable smoke test failed for {arguments}:\n{output}")
        if expected_code == 1 and "official Jasna release" not in output:
            raise BuildError(f"Executable activation guard failed:\n{output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg-dir", type=Path, help="FFmpeg 8 bin directory; otherwise use PATH")
    parser.add_argument("--vlc-dir", type=Path, help="64-bit VLC 3 directory")
    parser.add_argument("--without-vlc", action="store_true", help="omit video player audio runtime")
    parser.add_argument("--models-dir", type=Path, help="optionally include existing public detector/BasicVSR++ weights")
    parser.add_argument("--output-dir", type=Path, help="parent directory for jasna-free/")
    parser.add_argument("--work-dir", type=Path, help="build scratch directory; useful on a larger disk")
    parser.add_argument("--check-only", action="store_true", help="check prerequisites without producing an executable")
    arguments = parser.parse_args()
    try:
        if sys.platform != "win32" or sys.version_info < (3, 12):
            raise BuildError("Use 64-bit Windows Python 3.12 or newer for this build.")
        if sys.maxsize <= 2**32:
            raise BuildError("The build requires 64-bit Python.")
        with (ROOT / "pyproject.toml").open("rb") as file:
            project = tomllib.load(file)["project"]
        versions = validate_dependencies(project)
        ffmpeg, ffmpeg_version = checked_tool("ffmpeg", arguments.ffmpeg_dir)
        ffprobe, _ = checked_tool("ffprobe", arguments.ffmpeg_dir)
        vlc = find_vlc(arguments.vlc_dir, arguments.without_vlc)
        if arguments.models_dir is not None and not arguments.models_dir.is_dir():
            raise BuildError("--models-dir must be an existing directory.")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        output = (arguments.output_dir or ROOT / "dist_windows_public" / stamp).resolve()
        work = (arguments.work_dir or ROOT / "build" / "public-windows" / stamp).resolve()
        target = output / "jasna-free"
        if target.exists():
            raise BuildError(f"Output already exists: {target}. Choose a new --output-dir.")
        datas, binaries = build_inputs(ROOT, ffmpeg, ffprobe, vlc)
        print(f"Prerequisites ready: Python {sys.version.split()[0]}, PyInstaller {versions['pyinstaller']}, {ffmpeg_version}")
        if arguments.check_only:
            return 0
        work.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)
        config = work / "public-build.json"
        config.write_text(json.dumps({"root": str(ROOT), "datas": datas, "binaries": binaries}, indent=2), encoding="utf-8")
        environment = os.environ.copy()
        environment["JASNA_PUBLIC_BUILD_CONFIG"] = str(config)
        environment["PYINSTALLER_CONFIG_DIR"] = str(work / "pyinstaller-cache")
        subprocess.run([sys.executable, "-m", "PyInstaller", "--distpath", str(output), "--workpath", str(work / "work"),
                        str(ROOT / "scripts" / "pyinstaller" / "public_windows.spec")], cwd=ROOT, env=environment, check=True)
        copy_free_models(arguments.models_dir, target / "model_weights")
        revision = ""
        modified = None
        if shutil.which("git") is not None:
            revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                      capture_output=True, text=True, check=False).stdout.strip()
            status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                    capture_output=True, text=True, check=False)
            modified = bool(status.stdout.strip()) if status.returncode == 0 else None
        (target / "PUBLIC_BUILD.json").write_text(json.dumps({"profile": "public-free", "version": project["version"],
            "source_commit": revision, "source_modified": modified, "python": sys.version, "dependencies": versions,
            "ffmpeg": ffmpeg_version, "player_audio": vlc is not None}, indent=2), encoding="utf-8")
        smoke_test(target / "jasna-free.exe", work)
        print(f"Built and smoke-tested: {target / 'jasna-free.exe'}\nDistribute the entire {target} directory.")
        return 0
    except (BuildError, OSError, subprocess.SubprocessError) as error:
        print(f"Build failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
