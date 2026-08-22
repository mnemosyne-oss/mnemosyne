"""Prevent static Hermes plugin manifests from drifting from their packages."""

import re
import shutil
import subprocess
import sys
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _assignment_version(path: Path) -> str:
    match = re.search(
        r'^__version__\s*=\s*["\']([^"\']+)["\']',
        path.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match, f"missing __version__ assignment in {path}"
    return match.group(1)


def _manifest_version(path: Path) -> str:
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(manifest, dict), f"invalid plugin manifest in {path}"
    version = manifest.get("version")
    assert isinstance(version, str), f"missing manifest version in {path}"
    return version


def _project_version(path: Path) -> str:
    match = re.search(
        r'^version\s*=\s*["\']([^"\']+)["\']$',
        path.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match, f"missing static project version in {path}"
    return match.group(1)


def _source_manifest_paths() -> set[Path]:
    generated_roots = {".git", ".venv", "venv", "env", "build", "dist"}
    return {
        path
        for path in ROOT.rglob("plugin.yaml")
        if not any(
            part in generated_roots or part.endswith(".egg-info")
            for part in path.relative_to(ROOT).parts
        )
    }


def _build_standalone_wheel(hermes_root: Path, tmp_path: Path) -> Path:
    build_root = tmp_path / "hermes"
    shutil.copytree(
        hermes_root,
        build_root,
        ignore=shutil.ignore_patterns("build", "dist", "*.egg-info", "__pycache__"),
    )
    wheel_dir = tmp_path / "wheels"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
            ".",
        ],
        cwd=build_root,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected one standalone wheel, found {wheels}"
    return wheels[0]


def _build_core_wheel(tmp_path: Path) -> Path:
    build_root = tmp_path / "core"
    shutil.copytree(
        ROOT,
        build_root,
        ignore=shutil.ignore_patterns(
            ".git", "build", "dist", "*.egg-info", "__pycache__"
        ),
    )
    wheel_dir = tmp_path / "core-wheels"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
            ".",
        ],
        cwd=build_root,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected one core wheel, found {wheels}"
    return wheels[0]


def _wheel_metadata(archive: zipfile.ZipFile) -> bytes:
    metadata_paths = [
        path for path in archive.namelist() if path.endswith(".dist-info/METADATA")
    ]
    assert len(metadata_paths) == 1, (
        f"expected one wheel METADATA file, found {metadata_paths}"
    )
    return archive.read(metadata_paths[0])


def _manifest_version_from_wheel(archive: zipfile.ZipFile, path: str) -> str:
    manifest = yaml.safe_load(archive.read(path))
    assert isinstance(manifest, dict), f"invalid packaged plugin manifest in {path}"
    version = manifest.get("version")
    assert isinstance(version, str), f"missing packaged manifest version in {path}"
    return version


def test_all_plugin_manifests_have_an_explicit_version_contract():
    core_version = _assignment_version(ROOT / "mnemosyne" / "__init__.py")
    hermes_root = ROOT / "integrations" / "hermes"
    hermes_version = _assignment_version(
        hermes_root / "src" / "mnemosyne_hermes" / "__init__.py"
    )
    expected_versions = {
        ROOT / "hermes_memory_provider" / "plugin.yaml": core_version,
        hermes_root / "plugin.yaml": hermes_version,
        hermes_root / "src" / "mnemosyne_hermes" / "plugin.yaml": hermes_version,
    }

    assert _source_manifest_paths() == set(expected_versions)
    for manifest_path, expected_version in expected_versions.items():
        assert _manifest_version(manifest_path) == expected_version


def test_package_metadata_uses_the_same_version_contract():
    core_project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(
        r'^version\s*=\s*\{attr\s*=\s*"mnemosyne\.__version__"\}$',
        core_project,
        re.MULTILINE,
    )

    hermes_project = (ROOT / "integrations" / "hermes" / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    hermes_version = _assignment_version(
        ROOT / "integrations" / "hermes" / "src" / "mnemosyne_hermes" / "__init__.py"
    )
    assert re.search(
        rf'^version\s*=\s*"{re.escape(hermes_version)}"$',
        hermes_project,
        re.MULTILINE,
    )


def test_standalone_hermes_release_source_surfaces_agree():
    """All four standalone version surfaces must carry the same value.

    The version itself is read from pyproject.toml, which RELEASING.md names as
    the source of truth, rather than hardcoded. Pinning a literal here meant
    every plugin release had to rename this test, and a rename is easy to do
    without re-reading what it asserts.
    """
    hermes_root = ROOT / "integrations" / "hermes"
    distribution_version = _project_version(hermes_root / "pyproject.toml")
    runtime_version = _assignment_version(
        hermes_root / "src" / "mnemosyne_hermes" / "__init__.py"
    )
    source_manifest_version = _manifest_version(hermes_root / "plugin.yaml")
    packaged_manifest_version = _manifest_version(
        hermes_root / "src" / "mnemosyne_hermes" / "plugin.yaml"
    )

    assert distribution_version, "pyproject.toml declares no [project].version"
    assert runtime_version == distribution_version
    assert source_manifest_version == distribution_version
    assert packaged_manifest_version == distribution_version


def test_standalone_hermes_release_wheel_matches_the_declared_version(tmp_path):
    """The built wheel must carry the version pyproject.toml declares."""
    hermes_root = ROOT / "integrations" / "hermes"
    expected = _project_version(hermes_root / "pyproject.toml")
    wheel = _build_standalone_wheel(hermes_root, tmp_path)

    with zipfile.ZipFile(wheel) as archive:
        metadata = BytesParser(policy=policy.default).parsebytes(
            _wheel_metadata(archive)
        )
        manifest_path = "mnemosyne_hermes/plugin.yaml"

        assert manifest_path in archive.namelist()
        assert metadata["Name"] == "mnemosyne-hermes"
        assert metadata["Version"] == expected
        assert _manifest_version_from_wheel(archive, manifest_path) == expected


def test_core_wheel_ships_the_hermes_memory_provider_plugin_manifest(tmp_path):
    wheel = _build_core_wheel(tmp_path)

    with zipfile.ZipFile(wheel) as archive:
        assert "hermes_memory_provider/plugin.yaml" in archive.namelist()
