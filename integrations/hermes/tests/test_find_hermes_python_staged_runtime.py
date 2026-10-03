"""Discovery must prefer Hermes 0.21's staged runtime over the TCC anchor (#1068).

Hermes 0.21 splits what used to be one venv. ``<home>/hermes-agent/venv`` stays
only as the macOS TCC anchor, and the provider executes from
``<home>/installs/<key>/environments/<generation>/venv``, which Hermes' PM
records in ``installs/<key>/facts.json``. ``<key>`` is
``sha256(resolved checkout path)[:16]`` (``pm.environments.install_key``), so the
active install is derived from the checkout and ``installs/`` is never scanned.

The tree below is synthetic: fake interpreters plus ``pyvenv.cfg``, as the other
discovery tests build them. The anchor reports Python 3.11 and the staged
runtime 3.14, the split-minor shape from the report.
"""

from __future__ import annotations

import hashlib
import json
import logging
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from mnemosyne_hermes import install


def _fake_python(path: Path, version: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/bin/sh\n"
        f'case "$1" in --version) echo "Python {version}";; '
        f"*) echo '{{\"runtime\": \"python\", \"version\": \"{version}\"}}';; esac\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _venv(root: Path, version: str) -> Path:
    """A venv with an executable interpreter and its ``pyvenv.cfg``."""
    (root / "pyvenv.cfg").parent.mkdir(parents=True, exist_ok=True)
    (root / "pyvenv.cfg").write_text(f"version = {version}\n", encoding="utf-8")
    return _fake_python(root / "bin" / "python", version)


def _key(checkout: Path) -> str:
    """Hermes' ``pm.environments.install_key``, restated to pin the scheme."""
    return hashlib.sha256(str(checkout.resolve()).encode("utf-8")).hexdigest()[:16]


def _record(home: Path, checkout: Path, environment) -> Path:
    facts = home / "installs" / _key(checkout) / "facts.json"
    facts.parent.mkdir(parents=True, exist_ok=True)
    facts.write_text(
        json.dumps({"packages": {"venv": {"environment": environment}}}), encoding="utf-8"
    )
    return facts


@dataclass
class _World:
    home: Path
    checkout: Path
    anchor_python: Path
    staged_venv: Path
    staged_python: Path
    facts: Path


@pytest.fixture
def split_world(tmp_path, monkeypatch):
    """Anchor venv (3.11) beside a committed staged runtime (3.14).

    Every other discovery signal is neutralized, as in ``hermes_world``, so each
    test opts in to the launcher or environment it exercises.
    """
    home = tmp_path / "hermes-home"
    checkout = home / "hermes-agent"
    anchor_python = _venv(checkout / "venv", "3.11.15")
    staged_venv = home / "installs" / _key(checkout) / "environments" / "gen1" / "venv"
    staged_python = _venv(staged_venv, "3.14.7")
    facts = _record(home, checkout, str(staged_venv))

    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("PATH", str(empty_path))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user-home"))
    return _World(home, checkout, anchor_python, staged_venv, staged_python, facts)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX path known answer")
def test_find_hermes_python_install_key_matches_hermes_scheme():
    assert install._hermes_install_key(Path("/opt/hermes/hermes-agent")) == "bceb66d11c12ceaa"


@pytest.mark.parametrize("scoped", [False, True], ids=["default-home", "explicit-home"])
def test_find_hermes_python_prefers_staged_runtime_over_anchor(split_world, scoped):
    kwargs = {"hermes_home_path": split_world.home} if scoped else {}

    found = install._find_hermes_python(**kwargs)

    assert found == split_world.staged_python
    assert found != split_world.anchor_python


def test_find_hermes_python_returns_staged_path_lexically_under_home(split_world, tmp_path):
    """A symlinked home must not turn the path into a resolved one.

    The #1064 PM-generation warning matches the lexical ``<home>/installs`` shape.
    """
    if sys.platform == "win32":
        pytest.skip("POSIX symlink test")
    alias = tmp_path / "home-alias"
    alias.symlink_to(split_world.home)

    found = install._find_hermes_python(hermes_home_path=alias)

    assert found == alias / split_world.staged_python.relative_to(split_world.home)
    assert install._hermes_pm_generation_target(found, alias)


def test_find_hermes_python_staged_runtime_beats_launcher_in_the_anchor(split_world, monkeypatch):
    """A launcher that resolves into the anchor venv must not return the anchor."""
    launcher = split_world.anchor_python.parent / "hermes"
    launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    launcher.chmod(0o755)
    monkeypatch.setenv("PATH", str(launcher.parent))

    assert install._find_hermes_python() == split_world.staged_python


def test_find_hermes_python_staged_runtime_for_launcher_in_a_custom_checkout(
    split_world, tmp_path, monkeypatch
):
    """The launcher's own checkout is keyed, not just ``<home>/hermes-agent``."""
    checkout = tmp_path / "opt" / "hermes-agent"
    anchor = _venv(checkout / "venv", "3.11.15")
    launcher = anchor.parent / "hermes"
    launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    launcher.chmod(0o755)
    staged_venv = split_world.home / "installs" / _key(checkout) / "environments" / "g" / "venv"
    staged = _venv(staged_venv, "3.14.7")
    _record(split_world.home, checkout, str(staged_venv))
    monkeypatch.setenv("PATH", str(launcher.parent))

    found = install._find_hermes_python()

    assert found == staged
    assert found != anchor


def test_find_hermes_python_picks_the_checkouts_install_not_a_directory(split_world):
    """Several ``installs/<id>`` directories: only this checkout's key is read."""
    for other in ("0" * 16, "f" * 16):
        venv = split_world.home / "installs" / other / "environments" / "old" / "venv"
        _venv(venv, "3.12.0")
        facts = split_world.home / "installs" / other / "facts.json"
        facts.write_text(
            json.dumps({"packages": {"venv": {"environment": str(venv)}}}), encoding="utf-8"
        )

    assert install._find_hermes_python() == split_world.staged_python

    # Nothing committed for this checkout: never adopt a neighbour's generation.
    split_world.facts.unlink()
    assert install._find_hermes_python() == split_world.anchor_python


@pytest.mark.parametrize(
    "facts_text",
    [
        None,
        "{}",
        json.dumps({"packages": {}}),
        json.dumps({"packages": {"venv": {}}}),
        json.dumps({"packages": {"venv": {"environment": None}}}),
    ],
    ids=["no-record", "empty", "no-venv-fact", "no-environment", "null-environment"],
)
def test_find_hermes_python_falls_back_to_anchor_when_nothing_is_committed(
    split_world, facts_text
):
    """Hermes answers with the checkout's own venv here, so discovery does too."""
    if facts_text is None:
        split_world.facts.unlink()
    else:
        split_world.facts.write_text(facts_text, encoding="utf-8")

    assert install._find_hermes_python() == split_world.anchor_python
    assert install._find_hermes_python(hermes_home_path=split_world.home) == split_world.anchor_python


def test_find_hermes_python_pm_shaped_directory_without_a_record_is_not_selected(tmp_path, monkeypatch):
    """A venv under ``installs/`` that no record names stays invisible."""
    home = tmp_path / "hermes-home"
    anchor = _venv(home / "hermes-agent" / "venv", "3.11.15")
    _venv(home / "installs" / "id" / "environments" / "gen" / "venv", "3.14.7")
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)

    assert install._find_hermes_python(hermes_home_path=home) == anchor


def _break_missing_pyvenv(world):
    (world.staged_venv / "pyvenv.cfg").unlink()


def _break_missing_tree(world):
    for child in sorted(world.staged_venv.rglob("*"), reverse=True):
        child.unlink() if child.is_file() else child.rmdir()
    world.staged_venv.rmdir()


def _break_not_executable(world):
    world.staged_python.chmod(0o644)


def _break_malformed_json(world):
    world.facts.write_text("{not json", encoding="utf-8")


def _break_wrong_shape(world):
    world.facts.write_text("[]", encoding="utf-8")


def _break_packages_null(world):
    world.facts.write_text(json.dumps({"packages": None}), encoding="utf-8")


def _break_environment_not_a_path(world):
    _record(world.home, world.checkout, 3141)


def _break_environment_is_the_anchor(world):
    _record(world.home, world.checkout, str(world.anchor_python.parent.parent))


def _break_environment_escapes_by_traversal(world):
    escaping = world.staged_venv.parent / ".." / ".." / ".." / ".." / "hermes-agent" / "venv"
    _record(world.home, world.checkout, str(escaping))


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission and layout test")
@pytest.mark.parametrize(
    "break_record",
    [
        _break_missing_pyvenv,
        _break_missing_tree,
        _break_not_executable,
        _break_malformed_json,
        _break_wrong_shape,
        _break_packages_null,
        _break_environment_not_a_path,
        _break_environment_is_the_anchor,
        _break_environment_escapes_by_traversal,
    ],
)
@pytest.mark.parametrize("scoped", [False, True], ids=["default-home", "explicit-home"])
def test_find_hermes_python_fails_closed_when_the_staged_record_is_unusable(
    split_world, caplog, break_record, scoped
):
    """A record Hermes cannot run from is an error, never permission to use the anchor."""
    break_record(split_world)
    kwargs = {"hermes_home_path": split_world.home} if scoped else {}

    with caplog.at_level(logging.WARNING, logger=install.LOGGER.name):
        found = install._find_hermes_python(**kwargs)

    assert found is None
    assert any(
        str(split_world.facts) in record.getMessage() and "--python" in record.getMessage()
        for record in caplog.records
    )


def test_find_hermes_python_unusable_record_also_blocks_the_launcher_route(split_world, monkeypatch):
    launcher = split_world.anchor_python.parent / "hermes"
    launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    launcher.chmod(0o755)
    monkeypatch.setenv("PATH", str(launcher.parent))
    split_world.facts.write_text("{not json", encoding="utf-8")

    assert install._find_hermes_python() is None


def test_find_hermes_python_explicit_python_beats_staged_runtime(split_world, tmp_path):
    chosen = _fake_python(tmp_path / "chosen" / "python", "3.13.0")

    assert install._find_hermes_python(explicit_python=chosen) == chosen

    split_world.facts.write_text("{not json", encoding="utf-8")
    assert install._find_hermes_python(explicit_python=str(chosen)) == chosen


def test_runtime_python_json_reports_the_staged_runtime(split_world, capsys):
    rc = install.main(["--hermes-home", str(split_world.home), "runtime-python", "--json"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == {
        "ok": True,
        "python": str(split_world.staged_python),
        "version": "3.14.7",
    }


def test_runtime_python_json_fails_closed_for_an_unusable_record(split_world, capsys):
    split_world.facts.write_text("{not json", encoding="utf-8")

    rc = install.main(["--hermes-home", str(split_world.home), "runtime-python", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["ok"] is False
    assert "Could not identify Hermes' Python" in payload["error"]
    assert "3.11.15" not in json.dumps(payload)


def _stub_wrapper_probe(monkeypatch, tmp_path):
    site = tmp_path / "site-packages"
    site.mkdir()
    monkeypatch.setattr(install, "_site_packages_for_python", lambda *a, **kw: site)
    monkeypatch.setattr(install, "_check_wrapper_import", lambda *a, **kw: (True, None, False))
    return site


def test_wrapper_dry_run_plans_the_staged_runtime(split_world, tmp_path, monkeypatch, capsys):
    _stub_wrapper_probe(monkeypatch, tmp_path)

    rc = install.main(
        ["--hermes-home", str(split_world.home), "install", "--mode", "wrapper", "--dry-run"]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert f"Hermes Python: {split_world.staged_python}" in out
    assert f"Wrapper Python: {split_world.staged_python}" in out
    assert str(split_world.anchor_python) not in out
    # Discovery now hands the #1064 diagnostic exactly the layout it describes.
    assert "replaceable Hermes PM generation" in out
    assert not install.plugin_target_dir(split_world.home).exists()


def test_wrapper_install_records_the_staged_runtime(split_world, tmp_path, monkeypatch):
    _stub_wrapper_probe(monkeypatch, tmp_path)

    rc = install.main(["--hermes-home", str(split_world.home), "install", "--mode", "wrapper"])

    assert rc == 0
    state = install.plugin_state(hermes_home_path=split_world.home)
    assert state.mode == "wrapper"
    assert state.wrapper_python == split_world.staged_python


def test_wrapper_install_fails_closed_for_an_unusable_record(
    split_world, tmp_path, monkeypatch, capsys
):
    _stub_wrapper_probe(monkeypatch, tmp_path)
    split_world.facts.write_text("{not json", encoding="utf-8")

    rc = install.main(["--hermes-home", str(split_world.home), "install", "--mode", "wrapper"])

    assert rc == 1
    assert "Pass --python" in capsys.readouterr().err
    assert not install.plugin_target_dir(split_world.home).exists()


def test_status_reports_the_staged_runtime_as_hermes_python(split_world, capsys):
    install.main(["--hermes-home", str(split_world.home), "status"])

    out = capsys.readouterr().out
    assert f"Hermes' Python: {split_world.staged_python} (Python 3.14.7)" in out
    assert str(split_world.anchor_python) not in out
