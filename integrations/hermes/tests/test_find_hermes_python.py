"""Regression tests for Hermes interpreter discovery (issue #618).

``_find_hermes_python`` resolved the ``hermes`` launcher on PATH and took a
sibling ``python``. ``Path.resolve()`` follows symlinks but not an ``exec`` line,
so a shell-wrapper launcher left discovery pointing at the shim directory, and an
unrelated ``~/.local/bin/python`` was returned as "Hermes' Python". Bootstrap then
pip-installed ``mnemosyne-hermes[all]`` into that interpreter while reporting that
it had installed into Hermes' venv.

These tests model that layout: a wrapper launcher plus an unrelated PATH sibling,
while a valid known Hermes install root exists.
"""

import stat
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from mnemosyne_hermes import install


def _write_executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _make_venv(root: Path, *, python_target: Path | None = None) -> Path:
    """Create a directory that looks like a real venv. Returns its bin/python.

    ``pyvenv.cfg`` is the marker that separates a venv from a shim directory
    such as ``~/.local/bin``, which can hold both a ``hermes`` launcher and an
    unrelated ``python`` without being a venv at all.
    """
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    (root / "pyvenv.cfg").write_text("home = /usr/bin\ninclude-system-site-packages = false\n", encoding="utf-8")
    python = bin_dir / "python"
    if python_target is not None:
        python.symlink_to(python_target)
    else:
        _write_executable(python, "#!/bin/sh\nexit 0\n")
    return python


@dataclass
class _World:
    home: Path
    shims: Path
    venv: Path
    venv_python: Path
    shim_python: Path


@pytest.fixture
def hermes_world(tmp_path, monkeypatch):
    """A wrapper launcher on PATH, an unrelated sibling python, a real Hermes venv.

    Every other discovery signal is neutralized so each test opts in to exactly
    the layout it is exercising: ``VIRTUAL_ENV`` is cleared, ``sys.prefix`` is
    forced to look like a non-venv interpreter, and ``Path.home`` points at a
    temporary directory so a real ``~/hermes-agent`` cannot leak in.
    """
    home = tmp_path / "hermes-home"
    fake_user_home = tmp_path / "user-home"
    fake_user_home.mkdir(parents=True)

    venv = home / "hermes-agent" / "venv"
    venv_python = _make_venv(venv)
    _write_executable(venv / "bin" / "hermes", "#!/bin/sh\nexit 0\n")

    shims = tmp_path / "shims"
    _write_executable(
        shims / "hermes",
        "#!/usr/bin/env bash\n"
        "unset PYTHONPATH\n"
        "unset PYTHONHOME\n"
        f'exec "{venv / "bin" / "hermes"}" "$@"\n',
    )
    shim_python = _write_executable(shims / "python", "#!/bin/sh\nexit 0\n")

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("PATH", str(shims))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_user_home))

    return _World(
        home=home,
        shims=shims,
        venv=venv,
        venv_python=venv_python,
        shim_python=shim_python,
    )


def test_wrapper_launcher_does_not_hijack_hermes_venv(hermes_world):
    """The reported bug: a shell wrapper must not make its shim dir look like a venv."""
    found = install._find_hermes_python()

    assert found == hermes_world.venv_python
    assert found != hermes_world.shim_python


def test_unparseable_launcher_falls_back_to_known_hermes_root(hermes_world):
    """An opaque launcher yields no exec target, so the known install root wins."""
    _write_executable(
        hermes_world.shims / "hermes",
        "#!/usr/bin/env bash\nexec python -c 'import hermes.cli; hermes.cli.main()' \"$@\"\n",
    )

    assert install._find_hermes_python() == hermes_world.venv_python


@pytest.mark.parametrize("quote", ['"', "'"])
def test_quoted_launcher_target_may_contain_spaces(tmp_path, monkeypatch, quote):
    """A venv under a path with a space must still be followed, not skipped."""
    venv = tmp_path / "Ada Lovelace" / ".hermes" / "hermes-agent" / "venv"
    venv_python = _make_venv(venv)
    _write_executable(venv / "bin" / "hermes", "#!/bin/sh\nexit 0\n")

    shims = tmp_path / "shims"
    _write_executable(
        shims / "hermes",
        f"#!/usr/bin/env bash\nexec {quote}{venv / 'bin' / 'hermes'}{quote} \"$@\"\n",
    )
    shim_python = _write_executable(shims / "python", "#!/bin/sh\nexit 0\n")

    # No known root: if the exec line fails to parse, the shim sibling is all
    # that is left, which is the failure this guards against.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "empty-home"))
    monkeypatch.setenv("PATH", str(shims))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user-home"))

    found = install._find_hermes_python()

    assert found == venv_python
    assert found != shim_python


def test_oversized_launcher_is_not_read(tmp_path):
    """A stray binary named `hermes` must not be slurped in and parsed."""
    launcher = _write_executable(
        tmp_path / "hermes",
        "#!/bin/sh\n" + "# padding\n" * 20_000 + 'exec "/opt/hermes/venv/bin/hermes" "$@"\n',
    )
    assert launcher.stat().st_size > install._MAX_LAUNCHER_BYTES

    assert install._launcher_exec_target(launcher) is None


def test_launcher_without_shebang_is_ignored(tmp_path):
    binary = tmp_path / "hermes"
    binary.write_bytes(b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 64)

    assert install._launcher_exec_target(binary) is None


def test_explicit_python_is_authoritative(hermes_world, tmp_path):
    """--python wins over every probe, including a valid PATH venv."""
    chosen = _make_venv(tmp_path / "chosen")

    assert install._find_hermes_python(explicit_python=chosen) == chosen
    assert install._find_hermes_python(explicit_python=str(chosen)) == chosen


def test_venv_python_symlink_is_not_resolved(tmp_path, monkeypatch):
    """A venv's bin/python symlinks to its base interpreter; resolving loses the venv."""
    base = _write_executable(tmp_path / "base" / "bin" / "python3.11", "#!/bin/sh\nexit 0\n")
    home = tmp_path / "hermes-home"
    venv_python = _make_venv(home / "hermes-agent" / "venv", python_target=base)

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user-home"))

    found = install._find_hermes_python()

    assert found == venv_python
    assert found != base
    assert found.resolve() == base  # the symlink is real; we just must not follow it


def test_path_sibling_is_used_when_it_is_a_real_venv(tmp_path, monkeypatch):
    """Do not regress #388: a launcher inside a genuine venv still wins early."""
    venv = tmp_path / "usr" / "local" / "lib" / "custom-hermes" / "venv"
    venv_python = _make_venv(venv)
    _write_executable(venv / "bin" / "hermes", "#!/bin/sh\nexit 0\n")

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "empty-home"))
    monkeypatch.setenv("PATH", str(venv / "bin"))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user-home"))

    assert install._find_hermes_python() == venv_python


def test_unvalidated_launcher_sibling_is_never_returned(tmp_path, monkeypatch):
    """No validated venv means no candidate at all, not a plausible-looking guess.

    The sibling here is the shape of a Homebrew or system interpreter sitting
    next to a launcher. Returning it is what let `mnemosyne-hermes[all]` be
    installed into an unrelated Python, so discovery gives up instead.
    """
    system_bin = tmp_path / "usr" / "bin"
    _write_executable(system_bin / "hermes", "#!/bin/sh\nexit 0\n")
    system_python = _write_executable(system_bin / "python", "#!/bin/sh\nexit 0\n")

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "empty-home"))
    monkeypatch.setenv("PATH", str(system_bin))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user-home"))

    found = install._find_hermes_python()

    assert found is None
    assert found != system_python


def test_run_install_fails_clearly_when_nothing_validates(tmp_path, monkeypatch, capsys):
    """The no-validated-venv path must stop and name --python, not install anyway."""
    system_bin = tmp_path / "usr" / "bin"
    _write_executable(system_bin / "hermes", "#!/bin/sh\nexit 0\n")
    _write_executable(system_bin / "python", "#!/bin/sh\nexit 0\n")

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "empty-home"))
    monkeypatch.setenv("PATH", str(system_bin))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user-home"))
    monkeypatch.setattr(install, "check_mnemosyne_core", lambda: True)

    def _fail(*args, **kwargs):
        raise AssertionError("must not act without a validated Hermes runtime")

    monkeypatch.setattr(install, "_bootstrap_hermes_venv", _fail)
    monkeypatch.setattr(install, "install_plugin", _fail)
    monkeypatch.setattr(install, "install_bundled_skill", _fail)

    rc = install.run_install(hermes_home_path=tmp_path / "empty-home")

    assert rc == 1
    assert "--python" in capsys.readouterr().err


def test_no_bootstrap_continues_without_a_validated_interpreter(tmp_path, monkeypatch, capsys):
    """--no-bootstrap already forbids touching Hermes' venv, so nothing to prevent.

    Failing here would also preempt the guard that refuses to replace an
    existing wrapper install, replacing a data-safety message with a discovery
    one.
    """
    system_bin = tmp_path / "usr" / "bin"
    _write_executable(system_bin / "hermes", "#!/bin/sh\nexit 0\n")
    _write_executable(system_bin / "python", "#!/bin/sh\nexit 0\n")

    class _SkillResult:
        message = "skipped"

    link = tmp_path / "plugin-link"
    link.symlink_to(tmp_path)

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "empty-home"))
    monkeypatch.setenv("PATH", str(system_bin))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user-home"))
    monkeypatch.setattr(install, "check_mnemosyne_core", lambda: True)
    monkeypatch.setattr(install, "install_plugin", lambda **kwargs: link)
    monkeypatch.setattr(install, "install_bundled_skill", lambda **kwargs: _SkillResult())

    def _fail(*args, **kwargs):
        raise AssertionError("--no-bootstrap must never bootstrap")

    monkeypatch.setattr(install, "_bootstrap_hermes_venv", _fail)

    rc = install.run_install(hermes_home_path=tmp_path / "empty-home", no_bootstrap=True)

    assert rc == 0
    assert "Continuing without dependency validation" in capsys.readouterr().err


def test_wrapper_mode_is_unaffected_by_failed_discovery(tmp_path, monkeypatch):
    """Wrapper installs validate their own interpreter and must not be blocked."""
    system_bin = tmp_path / "usr" / "bin"
    _write_executable(system_bin / "hermes", "#!/bin/sh\nexit 0\n")

    class _SkillResult:
        message = "skipped"

    target = tmp_path / "wrapper-target"
    target.mkdir()

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "empty-home"))
    monkeypatch.setenv("PATH", str(system_bin))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user-home"))
    monkeypatch.setattr(install, "check_mnemosyne_core", lambda: True)
    monkeypatch.setattr(install, "install_plugin", lambda **kwargs: target)
    monkeypatch.setattr(install, "install_bundled_skill", lambda **kwargs: _SkillResult())
    monkeypatch.setattr(
        install,
        "plugin_state",
        lambda **kwargs: install.PluginState(
            status="installed",
            installed=True,
            target=target,
            mode="wrapper",
            message="ok",
        ),
    )

    rc = install.run_install(hermes_home_path=tmp_path / "empty-home", mode="wrapper")

    assert rc == 0


def test_run_install_bootstraps_hermes_venv_not_path_sibling(
    hermes_world, tmp_path, monkeypatch
):
    """The maintainer's acceptance criterion, at the bootstrap call site."""
    bootstrapped: list[Path] = []

    class _SkillResult:
        message = "skipped"

    link = tmp_path / "plugin-link"
    link.symlink_to(hermes_world.home)

    monkeypatch.setattr(install, "check_mnemosyne_core", lambda: True)
    monkeypatch.setattr(install, "check_mnemosyne_core_for_hermes_python", lambda python: None)
    monkeypatch.setattr(
        install,
        "_bootstrap_hermes_venv",
        lambda python: (bootstrapped.append(Path(python)), True)[1],
    )
    monkeypatch.setattr(install, "install_plugin", lambda **kwargs: link)
    monkeypatch.setattr(install, "install_bundled_skill", lambda **kwargs: _SkillResult())

    rc = install.run_install(hermes_home_path=hermes_world.home)

    assert rc == 0
    assert bootstrapped == [hermes_world.venv_python]
    assert hermes_world.shim_python not in bootstrapped


def test_run_install_honours_explicit_python(hermes_world, tmp_path, monkeypatch):
    """--python reaches discovery in symlink mode, not just wrapper mode."""
    bootstrapped: list[Path] = []
    chosen = _make_venv(tmp_path / "chosen")

    class _SkillResult:
        message = "skipped"

    link = tmp_path / "plugin-link"
    link.symlink_to(hermes_world.home)

    monkeypatch.setattr(install, "check_mnemosyne_core", lambda: True)
    monkeypatch.setattr(install, "check_mnemosyne_core_for_hermes_python", lambda python: None)
    monkeypatch.setattr(
        install,
        "_bootstrap_hermes_venv",
        lambda python: (bootstrapped.append(Path(python)), True)[1],
    )
    monkeypatch.setattr(install, "install_plugin", lambda **kwargs: link)
    monkeypatch.setattr(install, "install_bundled_skill", lambda **kwargs: _SkillResult())

    rc = install.run_install(hermes_home_path=hermes_world.home, python=chosen)

    assert rc == 0
    assert bootstrapped == [chosen]
