"""Regression tests for diagnostics path resolution."""

def test_diagnose_log_dir_honors_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

    import mnemosyne.diagnose as diagnose

    assert diagnose._default_log_dir() == tmp_path / "hermes" / "mnemosyne" / "logs"
