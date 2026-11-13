"""Exercise real native-integration entry paths before any runtime activation."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "integrations/hermes/src"
CATALOG = ROOT / "integrations/hermes-catalog"

# File loading matters: Hermes executes CLI siblings before package __init__.
ENTRIES = {
    "catalog": CATALOG / "__init__.py",
    "catalog_cli": CATALOG / "cli.py",
    "package": SRC / "mnemosyne_hermes/__init__.py",
    "plugin_entry": None,
    "provider_entry": None,
    "package_cli": SRC / "mnemosyne_hermes/cli.py",
    "installer": SRC / "mnemosyne_hermes/install.py",
    "console": SRC / "_mnemosyne_hermes_cli.py",
    "console_cached": SRC / "_mnemosyne_hermes_cli.py",
    "legacy": ROOT / "hermes_memory_provider/__init__.py",
    "legacy_cli": ROOT / "hermes_memory_provider/cli.py",
}


def _run(code, tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(HOME=str(tmp_path), HERMES_HOME=str(tmp_path),
               MNEMOSYNE_DATA_DIR=str(tmp_path / "data"), MNEMOSYNE_NO_EMBEDDINGS="1")
    return subprocess.run([sys.executable, "-I", "-B", "-c", textwrap.dedent(code)],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)


@pytest.fixture
def wrappers(tmp_path):
    # Generate on the supported build host, then load the produced artifacts in
    # a fresh excluded-platform process, including against an older dependency.
    from mnemosyne_hermes import install
    site = tmp_path / "selected-site"
    package = site / "mnemosyne_hermes"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("raise AssertionError('selected package executed')\n")
    (package / "cli.py").write_text("raise AssertionError('selected CLI executed')\n")
    wrapper = tmp_path / "wrapper"
    install._write_wrapper_plugin(wrapper, python=Path(sys.executable), site_packages=site)
    return wrapper


@pytest.mark.parametrize("platform, legacy_android, platform_label", [
    ("android", False, "android"), ("emscripten", False, "emscripten"),
    ("linux", True, "android"),
], ids=["android", "emscripten", "legacy_android"])
@pytest.mark.parametrize("dependencies", ["absent", "present", "cached"])
@pytest.mark.parametrize("entry", list(ENTRIES) + ["wrapper", "wrapper_cli"])
def test_excluded_entry_refuses_before_imports_delegation_or_registration(
    tmp_path, wrappers, platform, legacy_android, platform_label, dependencies, entry,
):
    path = ENTRIES.get(entry)
    if entry.startswith("wrapper"):
        path = wrappers / ("cli.py" if entry.endswith("cli") else "__init__.py")
    config = tmp_path / "config.yaml"
    config.write_text("memory:\n  provider: mnemosyne\nunrelated: preserve\n")
    before = config.read_bytes()
    code = f'''
        import argparse, builtins, importlib, importlib.abc, importlib.metadata
        import importlib.util, json, os, pathlib, sys, types
        platform, dependencies, entry = {platform!r}, {dependencies!r}, {entry!r}
        source, root, path = {str(SRC)!r}, {str(ROOT)!r}, {str(path)!r}
        # Dependency presence is established before the import-attempt spies.
        if dependencies in {"present", "cached"}:
            sys.path[:0] = [source, root]
            assert importlib.util.find_spec("mnemosyne") is not None
            assert importlib.util.find_spec("mnemosyne_hermes") is not None
        else:
            # Imports below simulate missing dependencies even in an editable test env.
            sys.path.insert(0, source)
        attempted, registrations, effects = [], [], []
        if dependencies == "cached":
            # Cached dependencies must not bypass the boundary either. Do not
            # cache the package under test itself: its initializer must execute.
            core = types.ModuleType("mnemosyne")
            core.__path__ = []
            sys.modules["mnemosyne"] = core
            if entry.startswith(("catalog", "wrapper")):
                provider = types.ModuleType("mnemosyne_hermes")
                provider.__path__ = []
                def register(ctx):
                    registrations.append("cached provider activation")
                provider.register = provider.register_memory_provider = register
                sys.modules["mnemosyne_hermes"] = provider
        class Context:
            def __getattr__(self, name):
                def register(*args, **kwargs):
                    registrations.append(name)
                return register
        original_import = builtins.__import__
        def spy_import(name, *args, **kwargs):
            forbidden = name == "mnemosyne" or name.startswith("mnemosyne.")
            if entry.startswith(("catalog", "wrapper")):
                forbidden = forbidden or name == "mnemosyne_hermes" or name.startswith("mnemosyne_hermes.")
            if forbidden:
                attempted.append(name)
                if dependencies == "absent":
                    raise ModuleNotFoundError("dependency absent: " + name)
                raise AssertionError("runtime dependency import attempted: " + name)
            return original_import(name, *args, **kwargs)
        original_find = importlib.util.find_spec
        def spy_find(name, *args, **kwargs):
            if name.startswith(("mnemosyne", "_mnemosyne")):
                attempted.append("find_spec:" + name)
                raise AssertionError("CLI sibling lookup before refusal")
            return original_find(name, *args, **kwargs)
        def audit(event, args):
            if event in {{"subprocess.Popen", "os.system", "sqlite3.connect", "socket.connect", "os.mkdir", "os.rename", "os.remove"}}:
                effects.append(event)
                raise AssertionError("excluded side effect: " + event)
            if event == "open" and (isinstance(args[1], str) and any(c in args[1] for c in "wax+")
                                     or isinstance(args[2], int) and args[2] & (os.O_WRONLY | os.O_RDWR)):
                effects.append("write")
                raise AssertionError("excluded file write")
        sys.addaudithook(audit)
        builtins.__import__ = spy_import
        importlib.util.find_spec = spy_find
        if {legacy_android!r}:
            def android_api_level():
                raise AssertionError("platform detection must not call Android API")
            sys.getandroidapilevel = android_api_level
        sys.platform = platform
        diagnostic = ""
        try:
            if entry in {{"plugin_entry", "provider_entry"}}:
                group = "hermes_agent.plugins" if entry == "plugin_entry" else "hermes_agent.memory_providers"
                ep = next(e for e in importlib.metadata.distribution("mnemosyne-hermes").entry_points
                          if e.group == group and e.name == "mnemosyne")
                loaded = ep.load()
                if entry == "plugin_entry":
                    loaded(Context())
                else:
                    loaded.register_memory_provider(Context())
            else:
                if entry.startswith("console"):
                    ep = next(e for e in importlib.metadata.distribution("mnemosyne-hermes").entry_points
                              if e.group == "console_scripts" and e.name == "mnemosyne-hermes")
                    main = ep.load()
                    module = sys.modules[main.__module__]
                else:
                    name = "mnemosyne_hermes" if entry == "package" else "boundary_probe"
                    spec = importlib.util.spec_from_file_location(name, path)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[name] = module
                    spec.loader.exec_module(module)
                if entry.startswith("console"):
                    if entry == "console_cached":
                        cached = types.ModuleType("_mnemosyne_hermes_installer")
                        def main(argv):
                            attempted.append("cached installer delegation")
                            return 0
                        cached.main = main
                        sys.modules[cached.__name__] = cached
                    # Console refuses with a nonzero return and safe stderr.
                    assert module.main(["install"]) != 0
                    diagnostic = "console"
                else:
                    if hasattr(module, "register"):
                        module.register(Context())
                    elif hasattr(module, "register_cli"):
                        module.register_cli(Context())
        except RuntimeError as exc:
            diagnostic = str(exc)
        assert not attempted, attempted
        assert not registrations, registrations
        assert not effects, effects
        expected = "Mnemosyne native Hermes integration is unsupported on {platform_label}; no provider or tools were activated."
        assert diagnostic == "console" or diagnostic == expected, diagnostic
    '''
    result = _run(code, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    if entry.startswith("console"):
        assert result.stderr == (
            f"Mnemosyne native Hermes integration is unsupported on {platform_label}; "
            "no provider or tools were activated.\n"
        )
    assert config.read_bytes() == before
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("entry", ["__init__.py", "cli.py"])
@pytest.mark.parametrize("platform, legacy_android, platform_label", [
    ("android", False, "android"), ("emscripten", False, "emscripten"),
    ("linux", True, "android"), ("linux", False, None),
], ids=["android", "emscripten", "legacy_android", "desktop_linux"])
def test_wrapper_refuses_excluded_platform_before_bootstrap_execution(
    tmp_path, wrappers, entry, platform, legacy_android, platform_label,
):
    """Catch bootstrap loading before refusal, retaining the desktop loading path."""
    marker = tmp_path / "bootstrap-executed"
    (wrappers / "_mnemosyne_bootstrap.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
        "raise AssertionError('bootstrap execution sentinel')\n"
    )
    code = f'''
        import importlib.util, sys
        if hasattr(sys, "getandroidapilevel"):
            del sys.getandroidapilevel
        if {legacy_android!r}:
            def android_api_level():
                raise AssertionError("platform detection must not call Android API")
            sys.getandroidapilevel = android_api_level
        sys.platform = {platform!r}
        spec = importlib.util.spec_from_file_location(
            "wrapper_probe", {str(wrappers / entry)!r}
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        platform_label = {platform_label!r}
        try:
            spec.loader.exec_module(module)
        except RuntimeError as exc:
            assert platform_label is not None
            assert str(exc) == (
                "Mnemosyne native Hermes integration is unsupported on {platform_label}; "
                "no provider or tools were activated."
            ), str(exc)
        except AssertionError as exc:
            assert platform_label is None, str(exc)
            assert str(exc) == "bootstrap execution sentinel", str(exc)
        else:
            raise AssertionError("wrapper did not reach refusal or bootstrap")
    '''
    result = _run(code, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert marker.exists() == (platform_label is None)


@pytest.mark.parametrize("entry", ["package", "catalog"])
def test_desktop_registers_provider_cli_and_tools_without_rewriting_config(tmp_path, entry):
    config = tmp_path / "config.yaml"
    config.write_text("memory:\n  provider: mnemosyne\nunrelated: preserve\n")
    before = config.read_bytes()
    code = f'''
        import importlib.util, sys
        sys.path[:0] = [{str(SRC)!r}, {str(ROOT)!r}]
        if {entry!r} == "package":
            import mnemosyne_hermes as module
        else:
            spec = importlib.util.spec_from_file_location("catalog", {str(CATALOG / '__init__.py')!r})
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        calls = []
        class Context:
            def register_memory_provider(self, provider):
                assert provider.is_available()
                calls.append("provider")
            def register_cli_command(self, **kwargs):
                assert kwargs["name"] == "mnemosyne"
                calls.append("cli")
            def register_tool(self, **kwargs):
                calls.append("tool")
        module.register(Context())
        assert calls[:2] == ["provider", "cli"], calls
        assert calls.count("tool") > 0, calls
    '''
    result = _run(code, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert config.read_bytes() == before
    # Healthy desktop registration may initialize its normal data directory.


def test_desktop_missing_core_retains_diagnostics_and_installer_help(tmp_path):
    code = f'''
        import importlib.abc, sys
        sys.path.insert(0, {str(SRC)!r})
        class MissingCore(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "mnemosyne" or fullname.startswith("mnemosyne."):
                    raise ModuleNotFoundError("No module named 'mnemosyne'")
        sys.meta_path.insert(0, MissingCore())
        import mnemosyne_hermes
        provider = mnemosyne_hermes.MnemosyneMemoryProvider()
        assert provider.is_available() is False
        from mnemosyne_hermes import install
        try:
            install.main(["--help"])
        except SystemExit as exc:
            assert exc.code == 0
    '''
    result = _run(code, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "unavailable" in result.stderr.lower()
    assert "install" in result.stdout and "status" in result.stdout
