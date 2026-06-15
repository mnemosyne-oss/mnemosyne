import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _run_cli(args, tmp_path, extra_env):
    env = os.environ.copy()
    env["HOME"] = str(tmp_path / "home")
    env["MNEMOSYNE_NO_EMBEDDINGS"] = "1"
    env.pop("MNEMOSYNE_DATA_DIR", None)
    env.pop("HERMES_HOME", None)
    env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-m", "mnemosyne.cli", *args],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_cli_uses_hermes_home_when_data_dir_is_unset(tmp_path):
    hermes_home = tmp_path / "custom-hermes-home"
    expected_db = hermes_home / "mnemosyne" / "data" / "mnemosyne.db"
    default_home_db = tmp_path / "home" / ".hermes" / "mnemosyne" / "data" / "mnemosyne.db"

    result = _run_cli(["stats"], tmp_path, {"HERMES_HOME": str(hermes_home)})

    assert result.returncode == 0, result.stderr
    assert f"DB path: {expected_db}" in result.stdout
    assert expected_db.exists()
    assert not default_home_db.exists()


def test_cli_mnemosyne_data_dir_overrides_hermes_home(tmp_path):
    hermes_home = tmp_path / "custom-hermes-home"
    data_dir = tmp_path / "explicit-mnemosyne-data"
    expected_db = data_dir / "mnemosyne.db"
    hermes_home_db = hermes_home / "mnemosyne" / "data" / "mnemosyne.db"

    result = _run_cli(
        ["stats"],
        tmp_path,
        {
            "HERMES_HOME": str(hermes_home),
            "MNEMOSYNE_DATA_DIR": str(data_dir),
        },
    )

    assert result.returncode == 0, result.stderr
    assert f"DB path: {expected_db}" in result.stdout
    assert expected_db.exists()
    assert not hermes_home_db.exists()
