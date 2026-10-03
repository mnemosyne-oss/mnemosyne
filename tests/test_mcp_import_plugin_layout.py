"""Regression coverage for #868's parent-directory import shadowing."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_mcp_server_resolves_to_inner_package_from_plugin_layout(tmp_path):
    """Pytest collection must not make the plugin stub shadow the core package."""
    plugin_parent = tmp_path / "plugins"
    plugin_dir = plugin_parent / "mnemosyne"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text((REPO_ROOT / "__init__.py").read_text())
    (plugin_dir / "mnemosyne").symlink_to(REPO_ROOT / "mnemosyne", target_is_directory=True)
    tests_dir = plugin_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "__init__.py").touch()

    test_file = tests_dir / "test_mcp_server_import.py"
    test_file.write_text(
        textwrap.dedent(
            f"""\
            from pathlib import Path

            import mnemosyne.mcp_server


            def test_mcp_server_comes_from_inner_package(pytestconfig):
                assert pytestconfig.getoption("importmode") == "importlib"
                assert Path(mnemosyne.mcp_server.__file__).resolve() == Path({str(REPO_ROOT / "mnemosyne" / "mcp_server.py")!r}).resolve()
            """
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-c",
            str(REPO_ROOT / "pyproject.toml"),
            str(test_file),
        ],
        cwd=plugin_dir,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"plugin-layout MCP import failed (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
