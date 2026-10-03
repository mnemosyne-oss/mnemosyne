"""Regression tests for Hermes interpreter validation (issue #618, after #620).

#620 taught discovery to follow a shell-wrapper launcher through its ``exec``
line, which fixed the reported symptom. These tests cover what remains: a
candidate is accepted only after it is shown to be a real virtual environment,
the venv symlink is not resolved away, ``--python`` reaches discovery, and a run
with nothing validated stops rather than bootstrapping into a guess.

Two layouts still returned the wrong interpreter with #620 alone:

* a launcher on PATH that is neither a symlink nor an ``exec`` wrapper -- a
  script that calls the real binary as a subprocess, or a compiled shim with no
  ``exec`` line -- resolves to itself, leaving ``bin_dir`` as the shim directory
  and an unrelated ``~/.local/bin/python`` as "Hermes' Python";
* the known-root branches returned ``candidate.resolve()``, and a venv's
  ``bin/python`` is a symlink to its base interpreter, so the venv was discarded.
"""

import os
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
    (root / "pyvenv.cfg").write_text(
        "home = /usr/bin\ninclude-system-site-packages = false\n", encoding="utf-8"
    )
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


def test_validation_does_not_regress_wrapper_resolution(hermes_world):
    """The #620 path must still win once the candidate has to prove it is a venv."""
    found = install._find_hermes_python()

    assert found == hermes_world.venv_python
    assert found != hermes_world.shim_python


def test_non_exec_launcher_does_not_hijack_a_known_hermes_root(tmp_path, monkeypatch):
    """A launcher that neither symlinks nor `exec`s resolves to its own shim dir.

    `_resolve_hermes_bin` correctly reports this launcher as the binary, so the
    sibling `python` is a shim-directory neighbour rather than a venv member.
    The valid Hermes root must win instead of `~/.local/bin/python`.
    """
    shims = tmp_path / "local" / "bin"
    # No `exec`: the wrapper runs the real binary as a subprocess and waits.
    _write_executable(
        shims / "hermes",
        '#!/usr/bin/env bash\n"/opt/hermes/real/bin/hermes" "$@"\n',
    )
    shim_python = _write_executable(shims / "python", "#!/bin/sh\nexit 0\n")

    home = tmp_path / "hermes-home"
    venv_python = _make_venv(home / "hermes-agent" / "venv")

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("PATH", str(shims))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user-home"))

    found = install._find_hermes_python()

    assert found == venv_python
    assert found != shim_python


def _launcher_world(tmp_path, monkeypatch, launcher_body, *, binary=False):
    """A launcher on PATH beside a validated decoy venv, plus a valid Hermes root.

    The shim directory here is itself a real venv, so `pyvenv.cfg` validation
    cannot save us: whether the sibling is trusted rests entirely on how the
    launcher is classified. Returns (decoy_python, hermes_python).
    """
    shim_venv = tmp_path / "decoy-venv"
    decoy_python = _make_venv(shim_venv)
    launcher = shim_venv / "bin" / "hermes"
    launcher.parent.mkdir(parents=True, exist_ok=True)
    if binary:
        launcher.write_bytes(launcher_body)
    else:
        launcher.write_text(launcher_body, encoding="utf-8")
    launcher.chmod(0o755)

    home = tmp_path / "hermes-home"
    hermes_python = _make_venv(home / "hermes-agent" / "venv")

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("PATH", str(shim_venv / "bin"))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user-home"))
    return decoy_python, hermes_python


def test_oversized_shebang_wrapper_is_not_treated_as_a_binary(tmp_path, monkeypatch):
    """A handoff past the read bound must not be read as "no handoff".

    The parser only reads a bounded prefix. Returning "not a wrapper" for a
    shebang script it could not read to the end licenses the caller to trust the
    interpreter beside the shim, which is the #618 outcome via a longer file.
    """
    padding = "# filler\n" * 700
    body = f'#!/usr/bin/env bash\n{padding}exec "/opt/hermes/real/bin/hermes" "$@"\n'
    encoded = body.encode()
    assert len(encoded) > install._MAX_WRAPPER_READ_BYTES
    # The handoff has to fall past the bound or this asserts nothing.
    assert b"exec" not in encoded[: install._MAX_WRAPPER_READ_BYTES]

    decoy_python, hermes_python = _launcher_world(tmp_path, monkeypatch, body)
    launcher = tmp_path / "decoy-venv" / "bin" / "hermes"

    assert install._wrapper_exec_target(launcher) == (True, None)

    found = install._find_hermes_python()
    assert found == hermes_python
    assert found != decoy_python


def test_non_shebang_executable_stays_a_direct_launcher(tmp_path, monkeypatch):
    """A compiled console script has no exec line, and that is a real answer.

    Failing closed here would break the pipx/#388 layout the launcher branch
    exists to serve, so the read must distinguish "binary" from "unreadable".
    """
    decoy_python, _ = _launcher_world(
        tmp_path, monkeypatch, b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 64, binary=True
    )
    launcher = tmp_path / "decoy-venv" / "bin" / "hermes"

    assert install._wrapper_exec_target(launcher) == (False, None)
    assert install._find_hermes_python() == decoy_python


def test_unreadable_launcher_is_not_treated_as_a_binary(tmp_path, monkeypatch):
    """If we cannot read the launcher we cannot classify it, so we must not guess."""
    decoy_python, hermes_python = _launcher_world(
        tmp_path, monkeypatch, "#!/bin/sh\nexec /opt/hermes/bin/hermes \"$@\"\n"
    )
    launcher = tmp_path / "decoy-venv" / "bin" / "hermes"
    launcher.chmod(0o111)  # executable, not readable

    if os.access(launcher, os.R_OK):  # pragma: no cover - root ignores the bit
        pytest.skip("running as root; the unreadable case cannot be modelled")

    assert install._wrapper_exec_target(launcher) == (True, None)

    found = install._find_hermes_python()
    assert found == hermes_python
    assert found != decoy_python


def test_unparseable_exec_line_is_not_treated_as_a_binary(tmp_path, monkeypatch):
    """An exec line that will not tokenize is an unresolved handoff, not the absence of one."""
    decoy_python, hermes_python = _launcher_world(
        tmp_path, monkeypatch, '#!/bin/sh\nexec "/opt/hermes/bin/hermes "$@"\n'
    )
    launcher = tmp_path / "decoy-venv" / "bin" / "hermes"

    assert install._wrapper_exec_target(launcher) == (True, None)

    found = install._find_hermes_python()
    assert found == hermes_python
    assert found != decoy_python


@pytest.mark.parametrize(
    "body",
    [
        '#!/bin/sh\nif true; then exec "/opt/hermes/bin/hermes" "$@"; fi\n',
        '#!/bin/sh\n[ -x /opt/hermes ] && exec "/opt/hermes/bin/hermes" "$@"\n',
        '#!/bin/sh\ncd /tmp; exec "/opt/hermes/bin/hermes" "$@"\n',
        '#!/bin/sh\nrun() { exec "/opt/hermes/bin/hermes" "$@"; }\nrun "$@"\n',
        '#!/bin/sh\nsh -c "exec /opt/hermes/bin/hermes"\n',
    ],
    ids=["if-then", "and-chain", "sequence", "function", "nested-sh-c"],
)
def test_unsupported_exec_form_is_not_treated_as_a_binary(tmp_path, monkeypatch, body):
    """A handoff we cannot parse is still a handoff, not the absence of one.

    Only the leading `exec ...` form is resolved. Every other shape must fail
    closed, because "no handoff" is what licenses trusting the launcher's
    sibling interpreter.
    """
    decoy_python, hermes_python = _launcher_world(tmp_path, monkeypatch, body)
    launcher = tmp_path / "decoy-venv" / "bin" / "hermes"

    assert install._wrapper_exec_target(launcher) == (True, None)

    found = install._find_hermes_python()
    assert found == hermes_python
    assert found != decoy_python


@pytest.mark.parametrize(
    "body",
    [
        "#!/usr/bin/env python3\nfrom hermes.cli import main\nmain()\n",
        "#!/usr/bin/env python3\nimport os\nos.execv(target, argv)\n",
        '#!/usr/bin/env python3\nexec(compile(src, path, "exec"))\n',
    ],
    ids=["plain", "os-execv", "exec-builtin"],
)
def test_python_console_script_stays_a_direct_launcher(tmp_path, monkeypatch, body):
    """Failing closed on the word `exec` anywhere would break the pipx layout.

    A console script that calls `os.execv` or the `exec` builtin is still a
    direct launcher, so the check matches a leading `exec` word rather than the
    substring.
    """
    decoy_python, _ = _launcher_world(tmp_path, monkeypatch, body)
    launcher = tmp_path / "decoy-venv" / "bin" / "hermes"

    assert install._wrapper_exec_target(launcher) == (False, None)
    assert install._find_hermes_python() == decoy_python


def test_explicit_python_is_authoritative(hermes_world, tmp_path):
    """--python wins over every probe, including a valid PATH venv."""
    chosen = _make_venv(tmp_path / "chosen")

    assert install._find_hermes_python(explicit_python=chosen) == chosen
    assert install._find_hermes_python(explicit_python=str(chosen)) == chosen


@pytest.mark.parametrize("empty", ["", "   ", "\t\n"])
def test_empty_explicit_python_is_rejected_not_ignored(hermes_world, empty):
    """An empty --python names nothing; falling through would substitute silently.

    `hermes_world` leaves a valid implicit candidate in place, so discovery
    would happily answer with a different interpreter than the one the user
    asked for. None stays the only "not supplied" signal.
    """
    with pytest.raises(ValueError, match="empty value"):
        install._find_hermes_python(explicit_python=empty)

    # The implicit candidate that would have been substituted is still there.
    assert install._find_hermes_python() == hermes_world.venv_python


def test_explicit_python_path_keeps_its_trailing_whitespace(hermes_world, tmp_path):
    """A POSIX filename may end in a space, and --python must preserve it.

    The blank check strips only to decide whether anything was named. Stripping
    the value that gets returned would name a path that does not exist here,
    silently sending the install elsewhere.
    """
    # A venv whose only interpreter has a filename ending in a space, so the
    # path string ends in one and the stripped form names nothing.
    venv = tmp_path / "padded"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    padded_python = _write_executable(venv / "bin" / "python ", "#!/bin/sh\nexit 0\n")
    assert str(padded_python).endswith(" ")
    assert not Path(str(padded_python).strip()).exists()

    found = install._find_hermes_python(explicit_python=str(padded_python))

    assert found == padded_python
    assert found.is_file()
    assert found != hermes_world.venv_python


def test_cli_reports_an_empty_python_instead_of_installing(hermes_world, capsys):
    """The CLI surfaces it as an error and exits non-zero rather than guessing."""
    rc = install.main(["install", "--dry-run", "--python", ""])

    assert rc == 1
    err = capsys.readouterr().err
    assert "--python was given an empty value" in err


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


def test_known_root_without_pyvenv_cfg_is_rejected(tmp_path, monkeypatch):
    """A directory named `venv` is not evidence that it is one.

    Held to the same bar as the launcher sibling: a half-removed environment,
    or one whose base interpreter is gone, still has bin/python sitting there.
    """
    home = tmp_path / "hermes-home"
    bin_dir = home / "hermes-agent" / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    damaged = _write_executable(bin_dir / "python", "#!/bin/sh\nexit 0\n")
    assert not (bin_dir.parent / "pyvenv.cfg").exists()

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user-home"))

    found = install._find_hermes_python()

    assert found is None
    assert found != damaged

    # Same layout, now a real venv: the root is found again, so the rejection
    # above is the missing pyvenv.cfg and not the path shape.
    (bin_dir.parent / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    assert install._find_hermes_python() == damaged


def test_known_root_python_must_be_executable(tmp_path, monkeypatch):
    """A validated venv whose bin/python cannot be executed is not a runtime.

    Everything downstream runs the candidate as `<python> -m pip ...`, so the
    known-root branch is held to the same executable check the launcher branch
    already applies.
    """
    home = tmp_path / "hermes-home"
    venv_python = _make_venv(home / "hermes-agent" / "venv")
    venv_python.chmod(0o644)

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user-home"))

    found = install._find_hermes_python()

    assert found is None
    assert found != venv_python

    # Same layout, now executable: the root is found again, so the rejection
    # above is the missing permission bit and nothing else about the fixture.
    venv_python.chmod(0o755)
    assert install._find_hermes_python() == venv_python


def test_stale_virtual_env_pointing_outside_a_venv_is_rejected(tmp_path, monkeypatch):
    """`VIRTUAL_ENV` is an environment variable, not proof that a venv is live.

    A stale or hand-set value such as `/usr` names `/usr/bin/python` and would
    hand bootstrap the system interpreter, which is the outcome this function
    exists to prevent.
    """
    fake_usr = tmp_path / "usr"
    system_python = _write_executable(fake_usr / "bin" / "python", "#!/bin/sh\nexit 0\n")
    assert not (fake_usr / "pyvenv.cfg").exists()

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "empty-home"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setenv("VIRTUAL_ENV", str(fake_usr))
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user-home"))

    found = install._find_hermes_python()

    assert found is None
    assert found != system_python

    # A real venv at the same variable is still honoured, so the rejection above
    # is the missing marker rather than the branch being dead.
    real = _make_venv(tmp_path / "real-venv")
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "real-venv"))
    assert install._find_hermes_python() == real


def test_virtual_env_interpreter_must_be_executable(tmp_path, monkeypatch):
    """A venv whose bin/python cannot be executed is not a runtime here either."""
    venv_python = _make_venv(tmp_path / "active-venv")
    venv_python.chmod(0o644)

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "empty-home"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "active-venv"))
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user-home"))

    assert install._find_hermes_python() is None

    venv_python.chmod(0o755)
    assert install._find_hermes_python() == venv_python


def test_active_prefix_interpreter_must_be_validated(tmp_path, monkeypatch):
    """`sys.prefix != sys.base_prefix` describes the running interpreter only.

    It says nothing about the `bin/python` being asked for here, which can be
    absent or non-executable in a partially built environment.
    """
    prefix = tmp_path / "active-prefix"
    prefix_python = _make_venv(prefix)
    prefix_python.chmod(0o644)

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "empty-home"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user-home"))

    assert install._find_hermes_python() is None

    prefix_python.chmod(0o755)
    assert install._find_hermes_python() == prefix_python

    # And a prefix that is not a venv at all is rejected outright.
    bare = tmp_path / "bare-prefix"
    _write_executable(bare / "bin" / "python", "#!/bin/sh\nexit 0\n")
    monkeypatch.setattr(sys, "prefix", str(bare))
    assert install._find_hermes_python() is None


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


@pytest.mark.parametrize("mode", ["symlink", "wrapper"])
@pytest.mark.parametrize("blank_python", ["", " \t\n "])
def test_run_install_rejects_blank_python_before_any_preflight(
    tmp_path, monkeypatch, blank_python, mode
):
    """A supplied blank --python never reaches preflight, discovery, or install."""
    calls: list[str] = []

    def fail(name):
        def _fail(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"blank --python reached {name}")

        return _fail

    monkeypatch.setattr(install, "check_mnemosyne_core", fail("core preflight"))
    monkeypatch.setattr(install, "_find_hermes_python", fail("discovery"))
    monkeypatch.setattr(install, "install_plugin", fail("plugin install"))

    with pytest.raises(ValueError, match="--python was given an empty value"):
        install.run_install(
            hermes_home_path=tmp_path / "home", mode=mode, python=blank_python
        )

    assert calls == []


@pytest.mark.parametrize("blank_python", ["", " \t\n "])
def test_wrapper_explicit_blank_python_fails_without_target_or_fallback(
    tmp_path, monkeypatch, capsys, blank_python
):
    """An explicit blank wrapper --python is invalid, not a request for this Python."""
    fallback_python = tmp_path / "fallback-python"
    _write_executable(fallback_python, "#!/bin/sh\nexit 0\n")
    hermes_home = tmp_path / "hermes-home"
    target = install.plugin_target_dir(hermes_home)

    monkeypatch.setattr(sys, "executable", str(fallback_python))

    def fail_if_fallback_is_validated(*args, **kwargs):
        raise AssertionError("blank --python must not fall back to sys.executable")

    monkeypatch.setattr(install, "_site_packages_for_python", fail_if_fallback_is_validated)

    rc = install.main(
        [
            "--hermes-home",
            str(hermes_home),
            "install",
            "--mode",
            "wrapper",
            "--python",
            blank_python,
        ]
    )

    assert rc == 1
    assert "--python was given an empty value" in capsys.readouterr().err
    assert not target.exists()


def test_wrapper_without_python_uses_discovered_hermes_interpreter(
    tmp_path, monkeypatch
):
    """An omitted wrapper --python records and validates Hermes' runtime, not ours."""
    discovered = _make_venv(tmp_path / "hermes-venv")
    installer_python = tmp_path / "installer-python"
    _write_executable(installer_python, "#!/bin/sh\nexit 0\n")
    selected: list[Path | None] = []
    target = tmp_path / "wrapper-target"
    target.mkdir()

    class _SkillResult:
        message = "skipped"

    monkeypatch.setattr(sys, "executable", str(installer_python))
    monkeypatch.setattr(install, "check_mnemosyne_core", lambda: True)
    monkeypatch.setattr(install, "_find_hermes_python", lambda **kwargs: discovered)
    monkeypatch.setattr(
        install,
        "install_plugin",
        lambda **kwargs: (selected.append(kwargs["python"]), target)[1],
    )
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

    assert install.run_install(hermes_home_path=tmp_path / "home", mode="wrapper") == 0
    assert selected == [discovered]
    assert selected != [Path(sys.executable)]


def test_wrapper_without_discovered_python_fails_closed(tmp_path, monkeypatch, capsys):
    """An omitted wrapper --python must not fall back to this Python or mutate."""
    fallback_python = tmp_path / "installer-python"
    _write_executable(fallback_python, "#!/bin/sh\nexit 0\n")
    target = install.plugin_target_dir(tmp_path / "home")
    calls: list[str] = []

    def fail(name):
        def _fail(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"missing wrapper runtime reached {name}")

        return _fail

    monkeypatch.setattr(sys, "executable", str(fallback_python))
    monkeypatch.setattr(install, "_find_hermes_python", lambda **kwargs: None)
    monkeypatch.setattr(install, "check_mnemosyne_core", fail("core preflight"))
    monkeypatch.setattr(install, "_site_packages_for_python", fail("fallback probe"))
    monkeypatch.setattr(install, "install_plugin", fail("plugin install"))
    monkeypatch.setattr(install, "install_bundled_skill", fail("skill install"))

    assert install.run_install(hermes_home_path=tmp_path / "home", mode="wrapper") == 1
    assert "Pass --python" in capsys.readouterr().err
    assert calls == []
    assert not target.exists()


def test_relative_wrapper_python_is_lexically_absolute_before_isolated_probe(
    tmp_path, monkeypatch
):
    """A relative --python survives the probe's different temporary cwd intact."""
    selected = tmp_path / "venv" / "bin" / "python"
    selected.parent.mkdir(parents=True)
    selected.symlink_to(Path(sys.executable))
    site_packages = tmp_path / "site-packages"
    _write_executable(site_packages / "mnemosyne_hermes" / "__init__.py", "__version__ = 'test'\n")
    (site_packages / "mnemosyne" / "core").mkdir(parents=True)
    (site_packages / "mnemosyne" / "__init__.py").write_text("", encoding="utf-8")
    (site_packages / "mnemosyne" / "core" / "__init__.py").write_text("", encoding="utf-8")
    (site_packages / "mnemosyne" / "core" / "beam.py").write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    relative_python = os.path.relpath(selected, start=tmp_path)
    expected = Path(relative_python).absolute()
    seen: list[Path] = []

    def site_for_python(python, **kwargs):
        seen.append(Path(python))
        return site_packages

    monkeypatch.setattr(install, "_site_packages_for_python", site_for_python)

    wrapper_python, returned_site = install._validated_wrapper_environment(relative_python)

    assert wrapper_python == expected == selected
    assert wrapper_python.resolve() == Path(sys.executable).resolve()
    assert seen == [expected]
    assert returned_site == site_packages


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


def test_dry_run_reports_explicitly_selected_interpreter(tmp_path, monkeypatch, capsys):
    """`install --dry-run --python X` must report X, not what discovery would pick.

    Covers the `main()` dry-run route rather than `run_install()`. The launcher
    here sits in a genuine venv, so discovery would return it if `--python` were
    not threaded through, which is what this pins down.
    """
    chosen = _make_venv(tmp_path / "chosen")

    launcher_venv = tmp_path / "path-venv"
    discovered = _make_venv(launcher_venv)
    _write_executable(launcher_venv / "bin" / "hermes", "#!/bin/sh\nexit 0\n")

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PATH", str(launcher_venv / "bin"))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user-home"))

    rc = install.main(["install", "--dry-run", "--python", str(chosen)])

    out = capsys.readouterr().out
    assert rc == 0
    assert f"Hermes Python: {chosen}" in out
    assert str(discovered) not in out


def test_wrapper_dry_run_does_not_report_a_bootstrap(tmp_path, monkeypatch, capsys):
    """Wrapper installs never bootstrap, so the dry run must not claim one.

    `--python` is honoured in both modes, so it gave the wrapper dry run a
    truthy interpreter and printed `Will bootstrap: True` for work that
    `run_install()` does not do in wrapper mode.
    """
    chosen = _make_venv(tmp_path / "chosen")

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user-home"))

    rc = install.main(
        ["install", "--dry-run", "--mode", "wrapper", "--python", str(chosen)]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "Will bootstrap:" not in out
    # The interpreter is still reported; only the bootstrap claim is dropped.
    assert f"Wrapper Python: {chosen}" in out

    # Same interpreter under a symlink install still reports the bootstrap, so
    # the line is gated on mode and not simply deleted.
    rc = install.main(
        ["install", "--dry-run", "--mode", "symlink", "--python", str(chosen)]
    )
    assert rc == 0
    assert "Will bootstrap: True" in capsys.readouterr().out


def test_wrapper_dry_run_uses_discovered_interpreter_for_display_and_probe(
    tmp_path, monkeypatch, capsys
):
    """The public dry-run path matches wrapper install's discovered runtime."""
    discovered = _make_venv(tmp_path / "hermes-venv")
    installer_python = tmp_path / "installer-python"
    _write_executable(installer_python, "#!/bin/sh\nexit 0\n")
    site_packages = tmp_path / "site-packages"
    probed: list[Path] = []

    monkeypatch.setattr(sys, "executable", str(installer_python))
    monkeypatch.setattr(install, "_find_hermes_python", lambda **kwargs: discovered)
    monkeypatch.setattr(
        install,
        "_site_packages_for_python",
        lambda python, **kwargs: (probed.append(Path(python)), site_packages)[1],
    )

    rc = install.main(["install", "--mode", "wrapper", "--dry-run"])

    out = capsys.readouterr().out
    assert rc == 0
    assert f"Wrapper Python: {discovered}" in out
    assert f"Wrapper site-packages: {site_packages}" in out
    assert str(installer_python) not in out
    assert probed == [discovered]


def test_wrapper_dry_run_without_discovered_python_fails_without_fallback_or_probe(
    tmp_path, monkeypatch, capsys
):
    """Dry-run has the same fail-closed wrapper-runtime boundary as installation."""
    fallback_python = tmp_path / "installer-python"
    _write_executable(fallback_python, "#!/bin/sh\nexit 0\n")
    calls: list[str] = []

    def fail(name):
        def _fail(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"missing wrapper runtime reached {name}")

        return _fail

    monkeypatch.setattr(sys, "executable", str(fallback_python))
    monkeypatch.setattr(install, "_find_hermes_python", lambda **kwargs: None)
    monkeypatch.setattr(install, "_site_packages_for_python", fail("fallback probe"))
    monkeypatch.setattr(install, "install_bundled_skill", fail("skill planning"))
    monkeypatch.setattr(install, "plugin_target_dir", fail("target planning"))

    assert install.main(["install", "--mode", "wrapper", "--dry-run"]) == 1
    assert "Pass --python" in capsys.readouterr().err
    assert calls == []


def test_wrapper_dry_run_makes_explicit_relative_python_absolute_before_probe(
    tmp_path, monkeypatch, capsys
):
    """The public dry-run path probes and reports relative --python lexically."""
    selected = _make_venv(tmp_path / "hermes-venv")
    site_packages = tmp_path / "site-packages"
    relative_python = os.path.relpath(selected, start=tmp_path)
    probed: list[Path] = []

    monkeypatch.chdir(tmp_path)
    expected = Path(relative_python).absolute()
    monkeypatch.setattr(
        install,
        "_site_packages_for_python",
        lambda python, **kwargs: (probed.append(Path(python)), site_packages)[1],
    )

    rc = install.main(
        ["install", "--mode", "wrapper", "--dry-run", "--python", relative_python]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert f"Wrapper Python: {expected}" in out
    assert f"Wrapper site-packages: {site_packages}" in out
    assert probed == [expected]


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


@pytest.mark.parametrize(
    ("mode", "cli_core_present", "selected_core_present", "expected_rc"),
    [
        pytest.param("wrapper", False, False, 0, id="wrapper-cli-missing-selected-missing"),
        pytest.param("wrapper", False, True, 0, id="wrapper-cli-missing-selected-present"),
        pytest.param("wrapper", True, False, 0, id="wrapper-cli-present-selected-missing"),
        pytest.param("wrapper", True, True, 0, id="wrapper-cli-present-selected-present"),
        pytest.param("symlink", False, False, 1, id="symlink-cli-missing-selected-missing"),
        pytest.param("symlink", False, True, 1, id="symlink-cli-missing-selected-present"),
        pytest.param("symlink", True, False, 1, id="symlink-cli-present-selected-missing"),
        pytest.param("symlink", True, True, 0, id="symlink-cli-present-selected-present"),
    ],
)
def test_explicit_python_core_preflight_matrix(
    tmp_path,
    monkeypatch,
    capsys,
    mode,
    cli_core_present,
    selected_core_present,
    expected_rc,
):
    """Wrapper --python bypasses CLI preflight; symlink mode retains it."""
    selected_python = tmp_path / "selected-python"
    selected_python.write_text("#!/bin/sh\n", encoding="utf-8")
    selected_python.chmod(0o755)
    target = tmp_path / "plugin-target"
    target.mkdir()
    symlink_target = tmp_path / "plugin-link"
    symlink_target.symlink_to(target, target_is_directory=True)
    cli_checks: list[bool] = []
    selected_checks: list[Path] = []
    install_calls: list[dict[str, object]] = []

    class _SkillResult:
        message = "skipped"

    def check_cli_core():
        cli_checks.append(True)
        return cli_core_present

    def check_selected_core(python):
        selected_checks.append(Path(python))
        return "4.0" if selected_core_present else None

    def install_selected(**kwargs):
        install_calls.append(kwargs)
        return target if mode == "wrapper" else symlink_target

    monkeypatch.setattr(install, "check_mnemosyne_core", check_cli_core)
    monkeypatch.setattr(install, "_find_hermes_python", lambda **kwargs: selected_python)
    monkeypatch.setattr(install, "check_mnemosyne_core_for_hermes_python", check_selected_core)
    monkeypatch.setattr(install, "install_plugin", install_selected)
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

    rc = install.run_install(
        hermes_home_path=tmp_path / "home",
        mode=mode,
        python=selected_python,
        no_bootstrap=True,
    )

    captured = capsys.readouterr()
    stderr = captured.err
    stdout = captured.out
    assert rc == expected_rc
    if mode == "wrapper":
        assert cli_checks == []
        assert selected_checks == []
        assert install_calls and install_calls[0]["python"] == selected_python
        assert "mnemosyne-memory NOT found in this Python" not in stderr
    else:
        assert cli_checks == [True]
        if not cli_core_present:
            assert selected_checks == []
            assert install_calls == []
            assert "mnemosyne-memory NOT found in this Python" in stderr
        else:
            assert selected_checks == [selected_python]
            if selected_core_present:
                assert install_calls and install_calls[0]["python"] == selected_python
            else:
                assert install_calls == []
                assert "Hermes' Python at" in stdout


def test_default_wrapper_without_explicit_python_defers_core_validation_to_wrapper(
    tmp_path, monkeypatch
):
    """The discovered wrapper runtime reaches its own selected-env validation."""
    cli_checks: list[bool] = []
    discovered = _make_venv(tmp_path / "hermes-venv")
    selected: list[Path | None] = []
    target = tmp_path / "wrapper-target"
    target.mkdir()

    class _SkillResult:
        message = "skipped"

    def check_cli_core():
        cli_checks.append(True)
        return False

    monkeypatch.setattr(install, "default_install_mode", lambda: "wrapper")
    monkeypatch.setattr(install, "check_mnemosyne_core", check_cli_core)
    monkeypatch.setattr(install, "_find_hermes_python", lambda **kwargs: discovered)
    monkeypatch.setattr(
        install,
        "install_plugin",
        lambda **kwargs: (selected.append(kwargs["python"]), target)[1],
    )
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

    rc = install.run_install(hermes_home_path=tmp_path / "home")

    assert rc == 0
    assert cli_checks == []
    assert selected == [discovered]


def test_default_symlink_retains_cli_core_preflight(tmp_path, monkeypatch, capsys):
    """The default symlink path still stops before discovery or installation."""
    cli_checks: list[bool] = []

    def check_cli_core():
        cli_checks.append(True)
        return False

    def fail(*args, **kwargs):
        raise AssertionError("symlink preflight must stop before later install work")

    monkeypatch.setattr(install, "default_install_mode", lambda: "symlink")
    monkeypatch.setattr(install, "check_mnemosyne_core", check_cli_core)
    monkeypatch.setattr(install, "_find_hermes_python", fail)
    monkeypatch.setattr(install, "install_plugin", fail)

    rc = install.run_install(hermes_home_path=tmp_path / "home")

    assert rc == 1
    assert cli_checks == [True]
    assert "mnemosyne-memory NOT found in this Python" in capsys.readouterr().err


def test_explicit_hermes_home_excludes_path_launcher(tmp_path, monkeypatch):
    """A scoped home wins over a different valid Hermes launcher on PATH."""
    selected_home = tmp_path / "selected-home"
    selected_python = _make_venv(selected_home / "hermes-agent" / "venv")

    other_venv = tmp_path / "other-hermes-venv"
    _make_venv(other_venv)
    _write_executable(other_venv / "bin" / "hermes", "#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("PATH", str(other_venv / "bin"))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)

    assert install._find_hermes_python(hermes_home_path=selected_home) == selected_python


def test_explicit_hermes_home_fails_closed_without_its_runtime(tmp_path, monkeypatch):
    """A scoped home cannot fall back to an unrelated valid launcher on PATH."""
    selected_home = tmp_path / "selected-home"
    other_venv = tmp_path / "other-hermes-venv"
    _make_venv(other_venv)
    _write_executable(other_venv / "bin" / "hermes", "#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("PATH", str(other_venv / "bin"))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)

    assert install._find_hermes_python(hermes_home_path=selected_home) is None


def _make_windows_venv(root: Path) -> Path:
    """Create a native Windows venv layout without changing ``os.name``."""
    scripts = root / "Scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (root / "pyvenv.cfg").write_text("home = C:/Python\n", encoding="utf-8")
    return _write_executable(scripts / "python.exe", "#!/bin/sh\nexit 0\n")


def _isolate_windows_discovery(tmp_path, monkeypatch):
    """Neutralize every implicit route; each test enables one route explicitly."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "empty-home"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user-home"))
    # Keep the simulated layout local to the installer module. In particular,
    # changing process-global ``os.name`` would make pathlib and shutil act as
    # though this Linux test process were Windows.
    monkeypatch.setattr(install, "_is_windows_platform", lambda: True, raising=False)


def test_windows_launcher_sibling_discovers_scripts_python_exe(tmp_path, monkeypatch):
    """Route 1: a native launcher sibling must find its validated Windows venv."""
    _isolate_windows_discovery(tmp_path, monkeypatch)
    venv = tmp_path / "launcher-venv"
    python = _make_windows_venv(venv)
    launcher = _write_executable(venv / "Scripts" / "hermes.exe", "#!/bin/sh\nexit 0\n")
    monkeypatch.setattr(install.shutil, "which", lambda _: str(launcher))

    assert install._find_hermes_python() == python


def test_windows_known_root_discovers_scripts_python_exe(tmp_path, monkeypatch):
    """Route 2: a known Hermes install root must support ``Scripts/python.exe``."""
    _isolate_windows_discovery(tmp_path, monkeypatch)
    home = tmp_path / "hermes-home"
    python = _make_windows_venv(home / "hermes-agent" / "venv")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert install._find_hermes_python() == python


def test_windows_active_prefix_discovers_scripts_python_exe(tmp_path, monkeypatch):
    """Route 3: the active venv prefix must find its native Windows runtime."""
    _isolate_windows_discovery(tmp_path, monkeypatch)
    prefix = tmp_path / "active-venv"
    python = _make_windows_venv(prefix)
    monkeypatch.setattr(sys, "prefix", str(prefix))

    assert install._find_hermes_python() == python


def test_windows_virtual_env_discovers_scripts_python_exe(tmp_path, monkeypatch):
    """Route 4: VIRTUAL_ENV must find its validated native Windows runtime."""
    _isolate_windows_discovery(tmp_path, monkeypatch)
    venv = tmp_path / "virtual-env"
    python = _make_windows_venv(venv)
    monkeypatch.setenv("VIRTUAL_ENV", str(venv))

    assert install._find_hermes_python() == python


@pytest.mark.parametrize("kind", ["missing", "nonexecutable", "unrelated"])
def test_windows_known_root_rejects_invalid_scripts_candidates(tmp_path, monkeypatch, kind):
    """Windows-shaped candidates retain the existing marker and executable checks."""
    _isolate_windows_discovery(tmp_path, monkeypatch)
    home = tmp_path / "hermes-home"
    root = home / "hermes-agent" / "venv"
    candidate = root / "Scripts" / "python.exe"
    if kind == "nonexecutable":
        _make_windows_venv(root).chmod(0o644)
    elif kind == "unrelated":
        _write_executable(candidate, "#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert install._find_hermes_python() is None


def test_windows_explicit_python_still_precedes_implicit_discovery(tmp_path, monkeypatch):
    """An explicit path remains authoritative over a valid Windows route."""
    _isolate_windows_discovery(tmp_path, monkeypatch)
    discovered = _make_windows_venv(tmp_path / "virtual-env")
    chosen = tmp_path / "chosen" / "python.exe"
    monkeypatch.setenv("VIRTUAL_ENV", str(discovered.parent.parent))

    assert install._find_hermes_python(explicit_python=chosen) == chosen


def test_posix_known_root_precedence_stays_python_then_python3(tmp_path, monkeypatch):
    """Windows support does not alter the established POSIX candidate order."""
    home = tmp_path / "hermes-home"
    venv = home / "hermes-agent" / "venv"
    python = _make_venv(venv)
    python3 = _write_executable(venv / "bin" / "python3", "#!/bin/sh\nexit 0\n")
    windows_python = _make_windows_venv(venv)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user-home"))
    monkeypatch.setattr(install, "_is_windows_platform", lambda: False, raising=False)

    assert install._find_hermes_python() == python
    assert python != python3 != windows_python
