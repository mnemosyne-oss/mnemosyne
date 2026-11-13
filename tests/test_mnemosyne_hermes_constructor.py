"""Constructor regression with a portable Hermes home-resolution contract.

The host module is intentionally minimal: Hermes is not a test dependency.
It returns real Paths and uses task-local overrides like hermes_constants;
provider initialization and persisted writes use the real SQLite backend.
"""
from contextvars import ContextVar
import os
from pathlib import Path
import sqlite3
import sys
from types import ModuleType

import pytest

from mnemosyne.core.config import MnemosyneConfig
from mnemosyne_hermes import MnemosyneMemoryProvider


@pytest.fixture
def resolvable_hermes_home(monkeypatch, tmp_path):
    home = ContextVar("constructor_test_hermes_home", default=None)
    module = ModuleType("hermes_constants")
    module.get_hermes_home = lambda: home.get() or Path(os.environ["HERMES_HOME"])
    module.hermes_home_key = lambda value=None: os.path.normcase(
        str(Path(value or module.get_hermes_home()).expanduser().resolve())
    )
    module.set_hermes_home_override = lambda value: home.set(Path(value))
    module.reset_hermes_home_override = home.reset
    monkeypatch.setitem(sys.modules, "hermes_constants", module)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "ambient"))
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "ambient-data"))
    MnemosyneConfig.reset_instance()
    try:
        yield module
    finally:
        MnemosyneConfig.reset_instance()


def _rows(path):
    with sqlite3.connect(path) as conn:
        return conn.execute(
            "SELECT session_id, content FROM working_memory ORDER BY id"
        ).fetchall()


@pytest.mark.parametrize("profile_isolation", [False, True])
@pytest.mark.parametrize("different_home", [False, True])
def test_construct_with_resolvable_home_then_initialize_and_persist(
    tmp_path, resolvable_hermes_home, profile_isolation, different_home
):
    host = resolvable_hermes_home
    ambient = host.get_hermes_home()
    assert ambient == tmp_path / "ambient"
    # Regression: main's constructor writes _beam through a property before
    # a binding exists for this already-resolvable host home.
    provider = MnemosyneMemoryProvider()
    assert provider._beam is None
    assert provider._bindings["default"]["beam"] is None
    target = tmp_path / "profile-b" if different_home else ambient
    token = None
    beam = None
    try:
        # Keep ambient active: explicit hermes_home must choose the database,
        # not merely agree with an already-switched host scope.
        assert host.get_hermes_home() == ambient
        provider.initialize(
            "CONSTRUCTOR", hermes_home=str(target), agent_identity="profile-b",
            profile_isolation=profile_isolation, auto_sleep=False,
        )
        assert provider._init_error is None
        assert provider._init_home is None
        assert host.get_hermes_home() == ambient
        if different_home:
            assert provider._beam is None
            assert not (ambient / "mnemosyne").exists()
        # Public writes run under the initialized home's host scope, with an
        # explicit per-operation session; no unscoped session guarantee here.
        token = host.set_hermes_home_override(target)
        beam = provider._beam
        assert beam is not None
        expected = target / "mnemosyne" / "data"
        if profile_isolation:
            expected = expected / "banks" / "profile-b"
        expected = expected / "mnemosyne.db"
        assert Path(beam.db_path).resolve() == expected.resolve()
        provider.sync_turn("constructor regression write", "", session_id="CONSTRUCTOR")
        expected_rows = [("hermes_CONSTRUCTOR", "[USER] constructor regression write")]
        assert _rows(expected) == expected_rows
        # Keep the established refusal: another uninitialized active home
        # must not acquire this binding or write through it.
        foreign = tmp_path / "uninitialized"
        foreign_token = host.set_hermes_home_override(foreign)
        try:
            assert provider._beam is None
            with pytest.raises(RuntimeError, match="refusing to write through another home's binding"):
                provider._session_id = "foreign"
            provider.sync_turn("must not leak", "", session_id="FOREIGN")
            assert not (foreign / "mnemosyne").exists()
        finally:
            host.reset_hermes_home_override(foreign_token)
        assert provider._beam is beam
        assert _rows(expected) == expected_rows
        if different_home:
            assert not (ambient / "mnemosyne").exists()
    finally:
        if token is not None:
            provider.shutdown()
            if beam is not None:
                beam.conn.close()
            host.reset_hermes_home_override(token)
