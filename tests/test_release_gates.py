"""Regression coverage for standalone and core release tag gates."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / ".githooks" / "pre-push"
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
PYTHON310 = shutil.which("python3.10")


def _tag_push(tag: str) -> str:
    return f"refs/tags/{tag} deadbeef refs/tags/{tag} {'0' * 40}\n"


def _run_hook(
    repo: Path, *tags: str, python310: str | None = None, tmp_path: Path | None = None
) -> subprocess.CompletedProcess[str]:
    env = None
    if python310 is not None:
        assert tmp_path is not None
        python_bin = tmp_path / "python-bin"
        python_bin.mkdir(parents=True)
        (python_bin / "python3").symlink_to(python310)
        env = {"HOME": str(tmp_path / "home"), "LC_ALL": "C", "PATH": f"{python_bin}:/usr/bin:/bin"}

    return subprocess.run(
        ["/usr/bin/bash", ".githooks/pre-push"],
        cwd=repo,
        input="".join(_tag_push(tag) for tag in tags),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _release_repo(
    tmp_path: Path,
    standalone_version: str = "0.6.0",
    *,
    high_standalone_tag: bool = False,
    project_toml: str | None = None,
) -> Path:
    repo = tmp_path / "release-repo"
    (repo / ".githooks").mkdir(parents=True)
    (repo / "mnemosyne").mkdir()
    (repo / "integrations" / "hermes").mkdir(parents=True)
    shutil.copy2(HOOK, repo / ".githooks" / "pre-push")
    (repo / "mnemosyne" / "__init__.py").write_text('__version__ = "3.16.0"\n')
    (repo / "integrations" / "hermes" / "pyproject.toml").write_text(
        project_toml
        or "[project]\nname = \"mnemosyne-hermes\"\n" f"version = \"{standalone_version}\"\n"
    )
    commands = [
        ["git", "init", "-q"],
        ["git", "config", "user.email", "release-gate@example.test"],
        ["git", "config", "user.name", "Release Gate"],
        ["git", "add", "."],
        ["git", "commit", "-qm", "fixture"],
        ["git", "tag", "v3.15.1"],
        ["git", "tag", "v0.5.0"],
    ]
    if high_standalone_tag:
        commands.append(["git", "tag", "v0.999.0"])
    for command in commands:
        subprocess.run(command, cwd=repo, check=True)
    return repo


def test_standalone_0_6_0_tag_succeeds_against_actual_repo_state():
    result = _run_hook(ROOT, "v0.6.0")

    assert result.returncode == 0, result.stderr + result.stdout
    assert "integrations/hermes/pyproject.toml" in result.stdout


@pytest.mark.skipif(PYTHON310 is None, reason="python3.10 is unavailable for hook compatibility coverage")
def test_standalone_0_6_0_tag_succeeds_with_python_3_10(tmp_path):
    result = _run_hook(ROOT, "v0.6.0", python310=PYTHON310, tmp_path=tmp_path)

    assert result.returncode == 0, result.stderr + result.stdout
    assert "ModuleNotFoundError" not in result.stderr


def test_hook_keeps_standalone_and_core_namespaces_independent(tmp_path):
    repo = _release_repo(tmp_path)

    mixed_result = _run_hook(repo, "v0.6.0", "v3.16.0")
    assert mixed_result.returncode == 0, mixed_result.stderr + mixed_result.stdout
    assert "v0.6.0" in mixed_result.stdout
    assert "v3.16.0" in mixed_result.stdout

    high_standalone_repo = _release_repo(tmp_path / "high", high_standalone_tag=True)
    core_result = _run_hook(high_standalone_repo, "v3.16.0")
    assert core_result.returncode == 0, core_result.stderr + core_result.stdout


def test_hook_rejects_mismatched_standalone_tag_before_release(tmp_path):
    repo = _release_repo(tmp_path, standalone_version="0.6.0")

    result = _run_hook(repo, "v0.7.0")

    assert result.returncode != 0
    assert "does not match the release version" in result.stdout
    assert "integrations/hermes/pyproject.toml" in result.stdout


@pytest.mark.skipif(PYTHON310 is None, reason="python3.10 is unavailable for hook compatibility coverage")
def test_hook_rejects_mismatched_standalone_tag_with_python_3_10(tmp_path):
    repo = _release_repo(tmp_path, standalone_version="0.6.0")

    result = _run_hook(repo, "v0.7.0", python310=PYTHON310, tmp_path=tmp_path / "run")

    assert result.returncode != 0
    assert "does not match the release version" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr


@pytest.mark.parametrize(
    ("project_toml", "error"),
    [
        ('[build-system]\nrequires = []\n', "[project] section is missing"),
        ('[project]\nname = "mnemosyne-hermes"\n', "version field is missing"),
        ('[project]\nversion = 0.6\n', "must be a non-empty static string"),
        ('[project]\nversion = "0.6.0"\nversion = "0.6.0"\n', "duplicate version fields"),
        ('[project]\nversion = "0.6.0"\nversion = "0.7.0"\n', "conflicting version fields"),
    ],
)
def test_hook_rejects_invalid_static_project_version(tmp_path, project_toml, error):
    repo = _release_repo(tmp_path, project_toml=project_toml)

    result = _run_hook(repo, "v0.6.0")

    assert result.returncode != 0
    assert error in result.stderr


def test_standalone_workflow_has_tomllib_tag_version_gate_before_build():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    job = workflow.split("  build-and-release-hermes:", maxsplit=1)[1]

    assert "- name: Validate standalone tag version" in job
    assert "RELEASE_TAG: ${{ github.ref_name }}" in job
    assert "import tomllib" in job
    assert 'tomllib.load(project_file)["project"]["version"]' in job
    assert 'expected = tag.removeprefix("v")' in job
    assert "if version != expected:" in job
    assert job.index("- name: Validate standalone tag version") < job.index("- name: Build package")
