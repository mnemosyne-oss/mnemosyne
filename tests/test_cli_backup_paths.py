"""CLI backup-path normalization regressions for Windows/MSYS (#659)."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from mnemosyne import cli


def test_msys_drive_path_is_normalized_only_on_windows():
    assert cli._normalize_backup_output_dir_arg("/c/Users/alice/backups", windows=True) == (
        "C:/Users/alice/backups"
    )
    assert cli._normalize_backup_output_dir_arg("/D/archive", windows=True) == "D:/archive"
    assert cli._normalize_backup_output_dir_arg("/c", windows=True) == "C:/"
    assert cli._normalize_backup_output_dir_arg(r"/c\Users\alice\backups", windows=True) == (
        "C:/Users/alice/backups"
    )
    assert cli._normalize_backup_output_dir_arg("/c/Users/alice/backups", windows=False) == (
        "/c/Users/alice/backups"
    )


@pytest.mark.parametrize(
    "path",
    [
        "C:/Users/alice/backups",
        r"C:\Users\alice\backups",
        r"\\server\share\backups",
        "//server/share/backups",
        "relative/backups",
    ],
)
def test_native_unc_and_relative_paths_are_preserved(path):
    assert cli._normalize_backup_output_dir_arg(path, windows=True) == path


@pytest.mark.parametrize("path", ["/tmp/backups", "///server/share", "/1/backups", "/é/backups"])
def test_ambiguous_posix_rooted_path_is_rejected_on_windows(path):
    with pytest.raises(ValueError, match="MSYS drive path"):
        cli._normalize_backup_output_dir_arg(path, windows=True)


def test_cmd_backup_passes_normalized_msys_path_to_backend(monkeypatch, capsys):
    captured = {}

    def fake_normalize(value):
        assert value == "/c/Users/alice/backups"
        return "C:/Users/alice/backups"

    def fake_backup(*, backup_dir):
        captured["backup_dir"] = backup_dir
        return {
            "backup_path": backup_dir / "mnemosyne_backup.db.gz",
            "original_size": 10,
            "backup_size": 5,
            "db_checksum": "abc123",
        }

    from mnemosyne.dr import recovery

    monkeypatch.setattr(cli, "_normalize_backup_output_dir_arg", fake_normalize)
    monkeypatch.setattr(recovery, "create_backup", fake_backup)

    cli.cmd_backup(["/c/Users/alice/backups"])

    assert captured["backup_dir"] == Path("C:/Users/alice/backups")
    assert "Backup created:" in capsys.readouterr().out


def test_production_os_detection_drives_cmd_backup_normalization(monkeypatch, capsys):
    captured = {}

    def fake_backup(*, backup_dir):
        captured["backup_dir"] = backup_dir
        return {
            "backup_path": backup_dir / "mnemosyne_backup.db.gz",
            "original_size": 10,
            "backup_size": 5,
            "db_checksum": "abc123",
        }

    from mnemosyne.dr import recovery

    monkeypatch.setattr(cli, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(recovery, "create_backup", fake_backup)

    cli.cmd_backup(["/c/Users/alice/backups"])

    assert captured["backup_dir"] == Path("C:/Users/alice/backups")
    assert "Backup created:" in capsys.readouterr().out


def test_cmd_backup_fails_before_backend_for_rejected_path(monkeypatch, capsys):
    from mnemosyne.dr import recovery

    def fail_if_called(**_kwargs):
        raise AssertionError("backend must not run for an invalid output path")

    def reject(_value):
        raise ValueError("bad path")

    monkeypatch.setattr(cli, "_normalize_backup_output_dir_arg", reject)
    monkeypatch.setattr(recovery, "create_backup", fail_if_called)

    with pytest.raises(SystemExit) as exc:
        cli.cmd_backup(["/tmp/backups"])

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "bad path" in captured.err
    assert "Backup created:" not in captured.out
