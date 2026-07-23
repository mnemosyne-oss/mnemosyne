"""Regression tests for issue #421 cross-session/default-scope bugs."""

import sqlite3

from mnemosyne.core.beam import BeamMemory


def _use_empty_config(tmp_path, monkeypatch):
    """Keep env-fallback regressions independent of the developer's config."""
    data_dir = tmp_path / "config-data"
    data_dir.mkdir()
    (data_dir / "config.yaml").write_text("")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))


def _contents(results):
    return [r.get("content", "") for r in results]


def test_cross_session_recall_does_not_bind_session_params_when_filter_disabled(tmp_path, monkeypatch):
    """MNEMOSYNE_CROSS_SESSION=1 should not leave stale bind params behind."""
    _use_empty_config(tmp_path, monkeypatch)
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION", "1")
    db_path = tmp_path / "mnemosyne.db"

    writer = BeamMemory(session_id="session-a", db_path=db_path)
    reader = BeamMemory(session_id="session-b", db_path=db_path)
    writer.remember(
        "issue421 cross session visible sentinel",
        source="test",
        importance=0.9,
        scope="session",
    )

    results = reader.recall("issue421 sentinel", top_k=10)

    assert any("cross session visible sentinel" in c for c in _contents(results))


def test_non_cross_session_recall_still_shows_only_session_and_global_scope(tmp_path, monkeypatch):
    """The issue #421 fix must preserve default session/global visibility."""
    _use_empty_config(tmp_path, monkeypatch)
    monkeypatch.delenv("MNEMOSYNE_CROSS_SESSION", raising=False)
    db_path = tmp_path / "mnemosyne.db"

    session_a = BeamMemory(session_id="session-a", db_path=db_path)
    session_b = BeamMemory(session_id="session-b", db_path=db_path)
    session_a.remember(
        "issue421 private sentinel",
        source="test",
        importance=0.9,
        scope="session",
    )
    session_a.remember(
        "issue421 global sentinel",
        source="test",
        importance=0.9,
        scope="global",
    )

    results = session_b.recall("issue421 sentinel", top_k=10)
    contents = _contents(results)

    assert any("global sentinel" in c for c in contents)
    assert not any("private sentinel" in c for c in contents)


def test_cli_store_honors_mnemosyne_default_scope_global(tmp_path, monkeypatch):
    """mnemosyne store should match MCP's MNEMOSYNE_DEFAULT_SCOPE behavior."""
    from mnemosyne import cli

    data_dir = tmp_path / "data"
    monkeypatch.setattr(cli, "DATA_DIR", str(data_dir))
    monkeypatch.setenv("MNEMOSYNE_DEFAULT_SCOPE", "global")

    cli.cmd_store(["issue421 cli global sentinel", "cli", "0.8"])

    db_path = data_dir / "mnemosyne.db"
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT scope FROM working_memory WHERE content = ?",
            ("issue421 cli global sentinel",),
        ).fetchone()

    assert row == ("global",)
