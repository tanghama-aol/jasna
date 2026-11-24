"""Public executables must run free workflows without importing private code."""
import runpy
import os
import sys
from types import SimpleNamespace

import pytest

from jasna import engine_paths, license_api
from jasna.packaging import public_runtime


@pytest.fixture(autouse=True)
def reset_public_profile(monkeypatch):
    monkeypatch.setattr(public_runtime, "_public_build", False)


def test_public_frozen_profile_keeps_activation_unavailable(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(license_api, "find_spec", lambda _name: pytest.fail("Public builds must not probe private code"))
    public_runtime.enable_public_build()

    assert not license_api.protection_available()
    with pytest.raises(license_api.ProtectionUnavailableError, match="official Jasna release"):
        license_api.require_protection()


def test_default_frozen_entry_still_requires_the_private_runtime(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "argv", ["jasna", "--version"])
    with pytest.raises(ImportError, match="initialize_runtime|jasna.protection"):
        runpy.run_module("jasna", run_name="__main__")


@pytest.mark.parametrize("argument", ["--ltx-worker", "--tpm-activate"])
def test_public_entry_rejects_private_worker_commands_without_importing_them(monkeypatch, argument):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "argv", ["jasna-free", argument])
    public_runtime.enable_public_build()
    with pytest.raises(SystemExit, match="official Jasna release"):
        runpy.run_module("jasna", run_name="__main__")


def test_public_frozen_linux_engine_cache_uses_the_plain_file(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    public_runtime.enable_public_build()
    engine = tmp_path / "basicvsrpp.engine"
    engine.write_bytes(b"public engine")

    assert engine_paths.basicvsrpp_sub_engine_exists(str(engine))
    assert not engine_paths.basicvsrpp_sub_engine_exists(str(tmp_path / "missing.engine"))
    assert license_api.ltx_driver_mismatch() is None


@pytest.mark.parametrize("frozen", [False, True])
def test_public_engine_compiler_children_keep_the_public_profile(monkeypatch, frozen):
    from jasna import engine_compiler

    public_runtime.enable_public_build()
    monkeypatch.setenv("TCL_LIBRARY", "parent-tcl")
    monkeypatch.setenv("TK_LIBRARY", "parent-tk")
    monkeypatch.setenv("JASNA_ENGINE_ENV_SENTINEL", "preserved")
    monkeypatch.setattr(engine_compiler, "is_frozen", lambda: frozen)
    monkeypatch.setattr(engine_compiler, "get_application_executable", lambda: "jasna-free.exe")
    calls = []

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=iter([]), wait=lambda **_kwargs: 0)

    monkeypatch.setattr(engine_compiler.subprocess, "Popen", popen)
    request = engine_compiler.EngineCompilationRequest(device="cuda:0", fp16=True, basicvsrpp=True)
    engine_compiler._run_compiler_subprocess(request, None)

    command, kwargs = calls[0]
    assert "TCL_LIBRARY" not in kwargs["env"]
    assert "TK_LIBRARY" not in kwargs["env"]
    assert kwargs["env"]["JASNA_ENGINE_ENV_SENTINEL"] == "preserved"
    assert os.environ["TCL_LIBRARY"] == "parent-tcl"
    assert os.environ["TK_LIBRARY"] == "parent-tk"
    assert command[-2:] == ["--compile-engines", request.to_json()]
    if frozen:
        assert command[0] == "jasna-free.exe"
    else:
        assert command[:3] == [sys.executable, "-m", "jasna.packaging.public_entry"]
