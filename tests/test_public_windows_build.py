from pathlib import Path
import subprocess

import pytest

from scripts import build_public_windows as build


@pytest.mark.parametrize("version", ["2.12.1", "2.12.1+cu129"])
def test_nvidia_build_rejects_torch_with_the_wrong_cuda_build(monkeypatch, version):
    project = {"dependencies": [], "optional-dependencies": {
        "public-nvidia": ["torch==2.12.1+cu130"], "public-build": ["pyinstaller>=6.11,<7"],
    }}
    monkeypatch.setattr(build.metadata, "version", lambda name: version if name == "torch" else "6.22.3")
    with pytest.raises(build.BuildError, match="torch==2.12.1\\+cu130"):
        build.validate_dependencies(project)


@pytest.mark.parametrize("name", ["ffmpeg", "ffprobe"])
def test_ffmpeg_8_nightly_builds_are_identified_by_their_libraries(monkeypatch, tmp_path, name):
    executable = tmp_path / f"{name}.exe"
    executable.touch()
    output = f"{name} version 2026-01-07-git-af6a1dd0b2-full_build\nlibavutil      60. 13.100 / 60. 13.100\n"
    monkeypatch.setattr(build.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, output, ""))
    path, version = build.checked_tool(name, tmp_path)
    assert path == executable.resolve()
    assert "2026-01-07" in version


def test_old_ffmpeg_fails_before_packaging(monkeypatch, tmp_path):
    (tmp_path / "ffmpeg.exe").touch()
    monkeypatch.setattr(build.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args, 0, "ffmpeg version 7.1\nlibavutil 59.39.100\n", ""))
    with pytest.raises(build.BuildError, match="major version 8"):
        build.checked_tool("ffmpeg", tmp_path)


def test_packaged_kernels_are_at_the_executable_root_and_private_files_are_omitted():
    datas, _ = build.build_inputs(build.ROOT, Path("ffmpeg.exe"), Path("ffprobe.exe"), None)
    images = [path for path in (build.ROOT / "jasna" / "media").iterdir() if path.suffix in {".fatbin", ".hipfb"}]
    assert images
    destinations = dict(datas)
    assert all(destinations[str(path)] == "." for path in images)
    assert not any("protection" in Path(path).parts or Path(path).suffix == ".mp4" for path, _ in datas)


def test_copying_models_does_not_ship_activation_state_or_supporter_models(tmp_path):
    source = tmp_path / "models"
    source.mkdir()
    public = {"yolo11m.pt", "rfdetr-v6.onnx", "lada_mosaic_restoration_model_generic_v1.2.pth"}
    private = {"license.json", "ltx.safetensors.enc", "unet-4x.onnx", "unet-4x.engine.enc", "cached.engine"}
    for name in public | private:
        (source / name).write_bytes(b"model or local state")
    destination = tmp_path / "dist" / "model_weights"
    build.copy_free_models(source, destination)
    assert {path.name for path in destination.iterdir()} == public
    assert {path.name for path in source.iterdir()} == public | private


def test_builder_preserves_existing_output(monkeypatch, tmp_path):
    output = tmp_path / "jasna-free"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep existing build", encoding="utf-8")
    monkeypatch.setattr(build.sys, "platform", "win32")
    monkeypatch.setattr(build.sys, "argv", ["build", "--without-vlc", "--output-dir", str(tmp_path)])
    monkeypatch.setattr(build, "validate_dependencies", lambda project: {"pyinstaller": "6.22.3"})
    monkeypatch.setattr(build, "checked_tool", lambda name, directory: (tmp_path / f"{name}.exe", "version 8"))
    monkeypatch.setattr(build, "find_vlc", lambda *args: None)
    monkeypatch.setattr(build.subprocess, "run", lambda *args, **kwargs: pytest.fail("Must not replace an existing build"))

    assert build.main() == 1
    assert sentinel.read_text(encoding="utf-8") == "keep existing build"
