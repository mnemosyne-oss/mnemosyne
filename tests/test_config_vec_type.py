"""Fresh-process regressions for #1378: VEC_TYPE ignored config.yaml.

beam.py resolves its module-level VEC_TYPE once at import. Before the fix it
read MNEMOSYNE_VEC_TYPE directly, so a value written by `mnemosyne config set
vec_type` or `mnemosyne profile apply` — both of which store it in
config.yaml — never reached the vector tables despite the documented
`config.yaml > env vars > defaults` precedence and the CLI telling users a
restart applies it. Each case runs a fresh interpreter because the constant
is bound at import, exactly like the reported reproduction.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _vec_type_in_subprocess(
    tmp_path: Path,
    env: dict[str, str],
    code: str = "from mnemosyne.core.beam import VEC_TYPE; print(VEC_TYPE)",
) -> str:
    environment = os.environ.copy()
    environment.pop("MNEMOSYNE_VEC_TYPE", None)
    environment.update(env)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), environment.get("PYTHONPATH", "")]
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        env=environment,
        check=True,
    )
    return proc.stdout.strip()


@pytest.mark.parametrize("yaml_value,expected", [("float32", "float32"), ("bit", "bit")])
def test_vec_type_reads_config_yaml(tmp_path: Path, yaml_value: str, expected: str):
    """A config.yaml value reaches the import-time resolution (#1378)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.yaml").write_text(f"vec_type: {yaml_value}\n")
    assert (
        _vec_type_in_subprocess(tmp_path, {"MNEMOSYNE_DATA_DIR": str(data_dir)})
        == expected
    )


def test_vec_type_yaml_beats_env(tmp_path: Path):
    """Documented precedence is config.yaml > env vars, so yaml wins."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.yaml").write_text("vec_type: bit\n")
    assert (
        _vec_type_in_subprocess(
            tmp_path,
            {"MNEMOSYNE_DATA_DIR": str(data_dir), "MNEMOSYNE_VEC_TYPE": "int8"},
        )
        == "bit"
    )


@pytest.mark.parametrize(
    ("yaml_text", "expected"),
    [
        ("memory:\n  mnemosyne:\n    vec_type: bit\n", "bit"),
        ("vec_type: bit\nmemory:\n  mnemosyne:\n    vec_type: int8\n", "int8"),
    ],
    ids=["nested-yaml", "duplicate-leaf"],
)
def test_nested_vec_type_uses_the_same_keys_as_get(tmp_path, yaml_text, expected):
    """Startup and runtime configuration agree when nested YAML competes with env."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config = data_dir / "config.yaml"
    config.write_text(yaml_text)
    before = config.stat().st_mtime_ns

    code = (
        "from mnemosyne.core.beam import VEC_TYPE; "
        "from mnemosyne.core.config import get, peek; "
        "print(VEC_TYPE, peek('vec_type'), get('vec_type'))"
    )
    values = _vec_type_in_subprocess(
        tmp_path,
        {"MNEMOSYNE_DATA_DIR": str(data_dir), "MNEMOSYNE_VEC_TYPE": "float32"},
        code=code,
    )
    assert values.split() == [expected] * 3
    assert list(data_dir.iterdir()) == [config]
    assert config.read_text() == yaml_text
    assert config.stat().st_mtime_ns == before


def test_vec_type_default_without_yaml_leaves_data_dir_absent(tmp_path):
    """Import keeps the int8 default and does not seed a missing config."""
    data_dir = tmp_path / "missing-data"
    assert _vec_type_in_subprocess(
        tmp_path, {"MNEMOSYNE_DATA_DIR": str(data_dir)}
    ) == "int8"
    assert not data_dir.exists()


def test_vec_type_env_fallback_with_unrelated_yaml(tmp_path):
    """An unrelated YAML key leaves the env value in effect."""
    config = tmp_path / "config.yaml"
    yaml_text = "# retain this file\nwm_max_items: 150\n"
    config.write_text(yaml_text)
    before = config.stat().st_mtime_ns
    assert _vec_type_in_subprocess(
        tmp_path, {"MNEMOSYNE_DATA_DIR": str(tmp_path), "MNEMOSYNE_VEC_TYPE": "bit"}
    ) == "bit"
    assert list(tmp_path.iterdir()) == [config]
    assert config.read_text() == yaml_text
    assert config.stat().st_mtime_ns == before


def test_vec_type_malformed_yaml_falls_back(tmp_path: Path):
    """Malformed YAML reads as empty; env and the default still apply."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.yaml").write_text("vec_type: [\n")
    env = {"MNEMOSYNE_DATA_DIR": str(data_dir)}
    assert (
        _vec_type_in_subprocess(tmp_path, {**env, "MNEMOSYNE_VEC_TYPE": "bit"})
        == "bit"
    )
    assert _vec_type_in_subprocess(tmp_path, env) == "int8"


def test_vec_type_unreadable_yaml_falls_back(tmp_path: Path):
    """A config.yaml that exists but cannot be read falls back the same way.

    Path.read_text is patched to raise PermissionError so the read-error path
    is deterministic: chmod(0) is ignored under elevated credentials such as
    CAP_DAC_OVERRIDE, which would let the file contents leak through. The file
    deliberately holds a third value so a leaked read is distinguishable.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.yaml").write_text("vec_type: float32\n")
    code = """
import os, pathlib
_target = pathlib.Path(os.environ["MNEMOSYNE_DATA_DIR"]) / "config.yaml"
_real_read = pathlib.Path.read_text
def _deny_read(self, *args, **kwargs):
    if self == _target:
        raise PermissionError(13, "Permission denied")
    return _real_read(self, *args, **kwargs)
pathlib.Path.read_text = _deny_read
from mnemosyne.core.beam import VEC_TYPE
print(VEC_TYPE)
"""
    env = {"MNEMOSYNE_DATA_DIR": str(data_dir)}
    assert (
        _vec_type_in_subprocess(tmp_path, {**env, "MNEMOSYNE_VEC_TYPE": "bit"}, code=code)
        == "bit"
    )
    assert _vec_type_in_subprocess(tmp_path, env, code=code) == "int8"


def test_vec_type_env_still_applies_without_yaml(tmp_path: Path):
    """The env var keeps working when config.yaml does not set the key."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    assert (
        _vec_type_in_subprocess(
            tmp_path,
            {"MNEMOSYNE_DATA_DIR": str(data_dir), "MNEMOSYNE_VEC_TYPE": "bit"},
        )
        == "bit"
    )


def test_vec_type_invalid_yaml_falls_back(tmp_path: Path):
    """An unrecognized value keeps the existing float32 fallback."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.yaml").write_text("vec_type: bogus\n")
    assert (
        _vec_type_in_subprocess(tmp_path, {"MNEMOSYNE_DATA_DIR": str(data_dir)})
        == "float32"
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits cannot deny the owner on Windows")
def test_vec_type_unreadable_data_dir_falls_back(tmp_path: Path):
    """An unreadable data dir (e.g. root-owned volume) must not break import.

    Path.exists() propagates EACCES for an unreachable config.yaml, which used
    to crash the module import in the non-root container acceptance check.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    data_dir.chmod(0)
    try:
        assert (
            _vec_type_in_subprocess(
                tmp_path,
                {"MNEMOSYNE_DATA_DIR": str(data_dir), "MNEMOSYNE_VEC_TYPE": "bit"},
            )
            == "bit"
        )
    finally:
        data_dir.chmod(0o700)


def test_vec_type_forced_oserror_falls_back(tmp_path: Path):
    """Deterministically cover peek's OSError guard on every Python version.

    The chmod-0 case above exercises a real permission denial, but on Python
    >= 3.14 Path.exists() returns False for EACCES instead of raising, so the
    except-branch is never hit there. Forcing the raise pins the handler's
    contract: env fallback when set, the passed default otherwise.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    code = """
import os, pathlib
_target = pathlib.Path(os.environ["MNEMOSYNE_DATA_DIR"]) / "config.yaml"
_real_exists = pathlib.Path.exists
def _deny_config(self):
    if self == _target:
        raise PermissionError(13, "Permission denied")
    return _real_exists(self)
pathlib.Path.exists = _deny_config
from mnemosyne.core.beam import VEC_TYPE
print(VEC_TYPE)
"""
    env = {"MNEMOSYNE_DATA_DIR": str(data_dir)}
    assert (
        _vec_type_in_subprocess(tmp_path, {**env, "MNEMOSYNE_VEC_TYPE": "bit"}, code=code)
        == "bit"
    )
    assert _vec_type_in_subprocess(tmp_path, env, code=code) == "int8"
