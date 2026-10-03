"""Regression tests for the staged-diff PII check in the pre-commit hook."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / ".githooks" / "pre-commit"
PII = "1641797+AxDSan" + "@users.noreply.github.com"


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def _hook_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "hook-repo"
    (repo / ".githooks").mkdir(parents=True)
    shutil.copy2(HOOK, repo / ".githooks" / "pre-commit")
    (repo / "pyproject.toml").write_text(
        f'[project]\nauthors = [{{name = "Maintainer", email = "{PII}"}}]\n',
        encoding="utf-8",
    )
    for args in (
        ("init", "-q"),
        ("config", "user.email", "hook-test@example.test"),
        ("config", "user.name", "Hook Test"),
        ("add", "."),
        ("commit", "-qm", "fixture"),
    ):
        result = _run(repo, *args)
        assert result.returncode == 0, result.stderr
    return repo


def _run_hook(
    repo: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/sh", ".githooks/pre-commit"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_precommit_allows_harmless_pyproject_change_with_historical_pii(tmp_path):
    repo = _hook_repo(tmp_path)
    pyproject = repo / "pyproject.toml"
    pyproject.write_text(pyproject.read_text(encoding="utf-8") + 'version = "1.0.0"\n', encoding="utf-8")
    assert _run(repo, "add", "pyproject.toml").returncode == 0

    result = _run_hook(repo)

    assert result.returncode == 0, result.stdout + result.stderr


def test_precommit_rejects_newly_added_pii(tmp_path):
    repo = _hook_repo(tmp_path)
    (repo / "new-contact.txt").write_text(f"contact: {PII}\n", encoding="utf-8")
    assert _run(repo, "add", "new-contact.txt").returncode == 0

    result = _run_hook(repo)

    assert result.returncode != 0
    assert "ERROR: PII detected in staged files." in result.stdout


def test_precommit_rejects_added_pii_with_diff_header_prefix(tmp_path):
    repo = _hook_repo(tmp_path)
    (repo / "header-prefix.txt").write_text(f"+++{PII}\n", encoding="utf-8")
    assert _run(repo, "add", "header-prefix.txt").returncode == 0

    result = _run_hook(repo)

    assert result.returncode != 0
    assert "ERROR: PII detected in staged files." in result.stdout


def test_precommit_rejects_pii_in_staged_binary_content(tmp_path):
    repo = _hook_repo(tmp_path)
    (repo / "contact.bin").write_bytes(b"\x00binary\xff" + PII.encode() + b"\x00")
    assert _run(repo, "add", "contact.bin").returncode == 0

    result = _run_hook(repo)

    assert result.returncode != 0
    assert "ERROR: PII detected in staged files." in result.stdout


def test_precommit_ignores_removed_historical_pii(tmp_path):
    repo = _hook_repo(tmp_path)
    pyproject = repo / "pyproject.toml"
    pyproject.write_text('[project]\nauthors = []\n', encoding="utf-8")
    assert _run(repo, "add", "pyproject.toml").returncode == 0

    result = _run_hook(repo)

    assert result.returncode == 0, result.stdout + result.stderr


def test_precommit_fails_closed_when_temp_file_creation_fails(tmp_path):
    repo = _hook_repo(tmp_path)
    missing_tmpdir = tmp_path / "missing-tmpdir"
    assert not missing_tmpdir.exists()

    result = _run_hook(repo, os.environ | {"TMPDIR": str(missing_tmpdir)})

    assert result.returncode != 0
    assert "ERROR: Unable to create temporary file for staged-diff PII check. Aborting commit." in result.stderr
    assert not missing_tmpdir.exists()


def test_precommit_excludes_hook_file_from_pii_check(tmp_path):
    repo = _hook_repo(tmp_path)
    hook = repo / ".githooks" / "pre-commit"
    hook.write_text(hook.read_text(encoding="utf-8") + f"# contact: {PII}\n", encoding="utf-8")
    assert _run(repo, "add", ".githooks/pre-commit").returncode == 0

    result = _run_hook(repo)

    assert result.returncode == 0, result.stdout + result.stderr


def test_precommit_fails_closed_when_git_diff_fails(tmp_path):
    repo = _hook_repo(tmp_path)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "diff" ]; then\n'
        '  echo "simulated git diff failure" >&2\n'
        "  exit 128\n"
        "fi\n"
        'exec "$REAL_GIT" "$@"\n',
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    real_git = shutil.which("git")
    assert real_git is not None
    hook_tmpdir = tmp_path / "hook-tmpdir"
    hook_tmpdir.mkdir()
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "REAL_GIT": real_git,
        "TMPDIR": str(hook_tmpdir),
    }

    result = _run_hook(repo, env)

    assert result.returncode != 0
    assert "ERROR: Unable to inspect staged changes; git diff failed. Aborting commit." in result.stderr
    assert not list(hook_tmpdir.glob("pre-commit.*"))


@pytest.mark.parametrize(
    "global_config",
    [
        "[core]\n\thooksPath = .githooks\n",
        "[commit]\n\tgpgsign = true\n",
    ],
    ids=["hooks-path", "gpg-signing"],
)
def test_precommit_hook_tests_ignore_hostile_global_git_config(tmp_path, global_config):
    config_path = tmp_path / "global.gitconfig"
    config_path.write_text(global_config, encoding="utf-8")
    hook_tests = (
        "test_precommit_allows_harmless_pyproject_change_with_historical_pii",
        "test_precommit_rejects_newly_added_pii",
        "test_precommit_rejects_added_pii_with_diff_header_prefix",
        "test_precommit_rejects_pii_in_staged_binary_content",
        "test_precommit_ignores_removed_historical_pii",
        "test_precommit_fails_closed_when_temp_file_creation_fails",
        "test_precommit_excludes_hook_file_from_pii_check",
        "test_precommit_fails_closed_when_git_diff_fails",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            *(f"{__file__}::{test}" for test in hook_tests),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ | {"GIT_CONFIG_GLOBAL": str(config_path)},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "8 passed" in result.stdout
