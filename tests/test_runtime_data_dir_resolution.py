"""Runtime data-dir resolution follows the environment on every call.

HERMES_HOME used to be read once, at import, and frozen into module
constants. A later in-process change must move banks, memory, beam, and
triples together. These tests never create files under the real home.
"""

from pathlib import Path


def _clear_data_env(monkeypatch):
    monkeypatch.delenv("MNEMOSYNE_DATA_DIR", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)


def test_t1_hermes_home_sets_data_dir(monkeypatch, tmp_path):
    """HERMES_HOME is the parent of mnemosyne/data when no override is set."""
    _clear_data_env(monkeypatch)
    home = tmp_path / "profile-a"
    monkeypatch.setenv("HERMES_HOME", str(home))

    from mnemosyne.core.paths import default_data_dir, default_db_path, default_root

    expected = home / "mnemosyne" / "data"
    assert default_root() == expected
    assert default_data_dir() == expected
    assert default_db_path() == expected / "mnemosyne.db"


def test_t2_hermes_home_change_without_reload(monkeypatch, tmp_path):
    """Changing HERMES_HOME after import returns the new data dir."""
    _clear_data_env(monkeypatch)
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    monkeypatch.setenv("HERMES_HOME", str(home_a))

    import mnemosyne.core.banks as banks
    import mnemosyne.core.beam as beam
    import mnemosyne.core.memory as memory
    import mnemosyne.core.triples as triples

    expected_a = home_a / "mnemosyne" / "data"
    assert banks._default_data_dir() == expected_a

    monkeypatch.setenv("HERMES_HOME", str(home_b))
    expected_b = home_b / "mnemosyne" / "data"
    # Existing helpers first, so an unfixed tree fails here with A vs B
    # rather than on the paths import below.
    assert banks._default_data_dir() == expected_b
    assert banks.DEFAULT_DATA_DIR == expected_b
    assert banks.BANKS_DIR == expected_b / "banks"
    assert memory._default_data_dir() == expected_b
    assert memory.DEFAULT_DATA_DIR == expected_b
    assert memory.DEFAULT_DB_PATH == expected_b / "mnemosyne.db"
    assert beam._default_data_dir() == expected_b
    assert beam.DEFAULT_DATA_DIR == expected_b
    assert beam.DEFAULT_DB_PATH == expected_b / "mnemosyne.db"
    assert triples.DEFAULT_DATA_DIR == expected_b
    assert triples.DEFAULT_DB == expected_b / "triples.db"

    from mnemosyne.core.paths import default_data_dir

    assert default_data_dir() == expected_b


def test_t3_core_modules_agree(monkeypatch, tmp_path):
    """banks, memory, beam, and triples resolve one shared data dir."""
    _clear_data_env(monkeypatch)
    home = tmp_path / "profile"
    monkeypatch.setenv("HERMES_HOME", str(home))
    expected = home / "mnemosyne" / "data"

    import mnemosyne.core.banks as banks
    import mnemosyne.core.beam as beam
    import mnemosyne.core.memory as memory
    import mnemosyne.core.triples as triples
    from mnemosyne.core.paths import default_data_dir

    resolved = {
        "paths": default_data_dir(),
        "banks": banks.default_data_dir(),
        "memory": memory._default_data_dir(),
        "beam": beam._default_data_dir(),
        "triples": triples.DEFAULT_DATA_DIR,
    }
    assert resolved == {name: expected for name in resolved}


def test_t4_mnemosyne_data_dir_wins_over_hermes_home(monkeypatch, tmp_path):
    """MNEMOSYNE_DATA_DIR is the data dir itself and beats HERMES_HOME."""
    home = tmp_path / "profile"
    override = tmp_path / "explicit-data"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(override))

    import mnemosyne.core.banks as banks
    import mnemosyne.core.beam as beam
    import mnemosyne.core.memory as memory
    import mnemosyne.core.triples as triples
    from mnemosyne.core.paths import default_data_dir

    assert default_data_dir() == override
    assert banks._default_data_dir() == override
    assert memory._default_data_dir() == override
    assert beam._default_data_dir() == override
    assert triples.DEFAULT_DATA_DIR == override


def test_t5_fallback_is_user_hermes_data_dir(monkeypatch):
    """With neither env var set, the data dir is ~/.hermes/mnemosyne/data."""
    _clear_data_env(monkeypatch)
    expected = Path.home() / ".hermes" / "mnemosyne" / "data"

    import mnemosyne.core.banks as banks
    import mnemosyne.core.beam as beam
    import mnemosyne.core.memory as memory
    import mnemosyne.core.triples as triples
    from mnemosyne.core.paths import default_data_dir

    assert default_data_dir() == expected
    assert banks._default_data_dir() == expected
    assert memory._default_data_dir() == expected
    assert beam._default_data_dir() == expected
    assert triples.DEFAULT_DATA_DIR == expected
