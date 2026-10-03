"""Regression coverage for the installable wheel payloads."""

import configparser
import email.parser
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _build_wheel(project_root: Path, build_name: str, tmp_path: Path) -> Path:
    build_root = tmp_path / build_name
    shutil.copytree(
        project_root,
        build_root,
        ignore=shutil.ignore_patterns(
            ".git",
            ".venv",
            ".codex",
            ".codegraph",
            "openspec",
            "build",
            "dist",
            "*.egg-info",
            "__pycache__",
        ),
    )
    wheel_dir = tmp_path / "wheels"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(wheel_dir),
            ".",
        ],
        cwd=build_root,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected one {build_name} wheel, found {wheels}"
    return wheels[0]


def _build_core_wheel(tmp_path: Path) -> Path:
    return _build_wheel(ROOT, "core", tmp_path)


def _build_standalone_hermes_wheel(tmp_path: Path) -> Path:
    return _build_wheel(ROOT / "integrations" / "hermes", "hermes", tmp_path)


def _manifest_version_from_wheel(archive: zipfile.ZipFile, path: str) -> str:
    manifest = yaml.safe_load(archive.read(path))
    assert isinstance(manifest, dict), f"invalid packaged plugin manifest in {path}"
    version = manifest.get("version")
    assert isinstance(version, str), f"missing packaged manifest version in {path}"
    return version


def _runtime_version_from_wheel(archive: zipfile.ZipFile, path: str) -> str:
    source = archive.read(path).decode("utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', source, re.MULTILINE)
    assert match, f"missing __version__ assignment in {path}"
    return match.group(1)


def _metadata_version_from_wheel(archive: zipfile.ZipFile, members: list[str]) -> str:
    metadata_files = [path for path in members if path.endswith(".dist-info/METADATA")]
    assert len(metadata_files) == 1, (
        f"expected one METADATA file, found {metadata_files}"
    )
    metadata = email.parser.BytesParser().parsebytes(archive.read(metadata_files[0]))
    version = metadata["Version"]
    assert isinstance(version, str), f"missing Version in {metadata_files[0]}"
    return version


def _entry_points_from_wheel(
    archive: zipfile.ZipFile, members: list[str]
) -> configparser.ConfigParser:
    entry_point_files = [
        path for path in members if path.endswith(".dist-info/entry_points.txt")
    ]
    assert len(entry_point_files) == 1, (
        f"expected one entry_points.txt, found {entry_point_files}"
    )
    entry_points = configparser.ConfigParser()
    entry_points.read_string(archive.read(entry_point_files[0]).decode("utf-8"))
    return entry_points


def test_core_wheel_keeps_package_integrations_and_excludes_repository_tree(tmp_path):
    wheel = _build_core_wheel(tmp_path)

    with zipfile.ZipFile(wheel) as archive:
        members = archive.namelist()
        entry_points = _entry_points_from_wheel(archive, members)

    assert "hermes_memory_provider/_verbatim_compat.py" in members
    assert "mnemosyne/__init__.py" in members
    expected_integrations = {
        "mnemosyne/integrations/memory_browser.py",
        "mnemosyne/integrations/auto_save_openwebui.py",
    }
    missing = expected_integrations.difference(members)
    assert not missing, f"Core wheel is missing package integrations: {sorted(missing)}"
    assert entry_points["console_scripts"]["mnemosyne-browser"] == (
        "mnemosyne.integrations.memory_browser:main"
    )
    assert entry_points["console_scripts"]["mnemosyne-auto-save"] == (
        "mnemosyne.integrations.auto_save_openwebui:main"
    )
    leaked = [
        path
        for path in members
        if path == "integrations" or path.startswith("integrations/")
    ]
    leaked_hermes_namespace = [
        path
        for path in members
        if path == "mnemosyne_hermes" or path.startswith("mnemosyne_hermes/")
    ]
    assert not leaked, (
        f"Core wheel contains repository-only integrations payload: {leaked}"
    )
    assert not leaked_hermes_namespace, (
        "Core wheel contains standalone Hermes payload: "
        f"{leaked_hermes_namespace}"
    )


def test_standalone_hermes_wheel_keeps_plugin_manifest(tmp_path):
    wheel = _build_standalone_hermes_wheel(tmp_path)

    with zipfile.ZipFile(wheel) as archive:
        members = archive.namelist()
        manifest_version = _manifest_version_from_wheel(
            archive, "mnemosyne_hermes/plugin.yaml"
        )
        runtime_version = _runtime_version_from_wheel(
            archive, "mnemosyne_hermes/__init__.py"
        )
        metadata_version = _metadata_version_from_wheel(archive, members)
        entry_points = _entry_points_from_wheel(archive, members)

    assert "mnemosyne_hermes/_verbatim_compat.py" in members
    assert "mnemosyne_hermes/plugin.yaml" in members
    assert manifest_version == runtime_version
    assert manifest_version == metadata_version
    assert entry_points["hermes_agent.plugins"]["mnemosyne"] == (
        "mnemosyne_hermes:register"
    )
    assert (
        entry_points["hermes_agent.memory_providers"]["mnemosyne"] == "mnemosyne_hermes"
    )
    assert entry_points["console_scripts"]["mnemosyne-hermes"] == (
        "mnemosyne_hermes.install:main"
    )


def test_core_wheel_top_level_names_are_allowlisted(tmp_path):
    """The root package finder must not sweep repository-only directories.

    #729 fixed this for ``integrations``; the same greedy discovery also
    shipped ``examples`` as an installed top-level package. Asserting the
    whole top-level surface catches the next directory instead of the last
    one, so a new repo-root package cannot reach site-packages unnoticed.
    """
    wheel = _build_core_wheel(tmp_path)

    with zipfile.ZipFile(wheel) as archive:
        members = archive.namelist()

    top_level = {path.split("/")[0] for path in members}
    dist_info = {name for name in top_level if name.endswith(".dist-info")}
    assert len(dist_info) == 1, f"expected one .dist-info, found {sorted(dist_info)}"

    packages = top_level - dist_info
    assert packages == {"mnemosyne", "hermes_memory_provider"}, (
        "Core wheel top-level surface changed; unexpected entries reach "
        f"site-packages on install: {sorted(packages)}"
    )
