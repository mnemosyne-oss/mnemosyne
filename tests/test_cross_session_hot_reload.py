"""In-process hot-reload coverage for cross-session recall."""

from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.config import MnemosyneConfig, get_config


def test_cross_session_recall_follows_config_reload(tmp_path, monkeypatch):
    """One Beam reader observes false -> true -> false without reconstruction."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config_path = data_dir / "config.yaml"
    config_path.write_text("cross_session: false\n")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION", "1")
    MnemosyneConfig.reset_instance()

    writer = reader = None
    try:
        db_path = tmp_path / "memories.db"
        writer = BeamMemory(session_id="writer", db_path=db_path)
        reader = BeamMemory(session_id="reader", db_path=db_path)
        writer.remember("reload scope sentinel", source="test", importance=0.9)

        def visible():
            return any("reload scope sentinel" in row["content"] for row in reader.recall("reload scope sentinel"))

        assert not visible()
        config_path.write_text("cross_session: true\n")
        get_config().reload()
        assert visible()
        config_path.write_text("cross_session: false\n")
        get_config().reload()
        assert not visible()
    finally:
        if writer is not None:
            writer.conn.close()
        if reader is not None:
            reader.conn.close()
        MnemosyneConfig.reset_instance()
