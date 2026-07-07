"""Regression tests for issue #421 cross-session visibility bugs."""

import importlib
import os
import sqlite3


def _reload_beam_with_cross_session(enabled: bool):
    if enabled:
        os.environ["MNEMOSYNE_CROSS_SESSION"] = "1"
    else:
        os.environ.pop("MNEMOSYNE_CROSS_SESSION", None)

    import mnemosyne.core.beam as beam

    return importlib.reload(beam)


def test_cross_session_recall_does_not_misbind_sql_params(tmp_path):
    beam = _reload_beam_with_cross_session(True)
    try:
        db_path = tmp_path / "cross_session.db"
        session_a = beam.BeamMemory(db_path=db_path, session_id="session-a")
        session_a.remember("issue 421 cross session memory", scope="global")

        session_b = beam.BeamMemory(db_path=db_path, session_id="session-b")
        results = session_b.recall("issue 421 cross session memory")

        assert any("issue 421 cross session memory" in r["content"] for r in results)
    finally:
        _reload_beam_with_cross_session(False)


def test_cli_store_honors_default_scope_global(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MNEMOSYNE_DEFAULT_SCOPE", "global")

    import mnemosyne.cli as cli

    data_dir = tmp_path / "mnemosyne-data"
    data_dir.mkdir()
    monkeypatch.setattr(cli, "DATA_DIR", str(data_dir))

    cli.cmd_store(["issue 421 cli global scope memory"])
    capsys.readouterr()

    with sqlite3.connect(data_dir / "mnemosyne.db") as conn:
        scope = conn.execute(
            "SELECT scope FROM working_memory WHERE content = ?",
            ("issue 421 cli global scope memory",),
        ).fetchone()[0]

    assert scope == "global"
