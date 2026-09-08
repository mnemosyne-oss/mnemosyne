from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_memory_provider import MnemosyneMemoryProvider
from mnemosyne_hermes import MnemosyneMemoryProvider as PackagedMemoryProvider

# The provider is duplicated between hermes_memory_provider/ and
# integrations/hermes/src/mnemosyne_hermes/ (docs/rfc/0003-media-moments.md).
# Delete-cascade behaviour must stay identical in both copies.
PROVIDER_CLASSES = [MnemosyneMemoryProvider, PackagedMemoryProvider]


def _provider(tmp_path: Path, monkeypatch, agent_identity="Sisyphus",
              provider_cls=MnemosyneMemoryProvider):
    data_dir = tmp_path / "mnemosyne-data"
    hermes_home = tmp_path / "profiles" / agent_identity.lower()
    hermes_home.mkdir(parents=True)
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir / "private"))
    monkeypatch.setenv("MNEMOSYNE_HOST_LLM_ENABLED", "0")
    provider = provider_cls()
    provider.initialize(
        session_id=f"{agent_identity.lower()}-session",
        hermes_home=str(hermes_home),
        agent_identity=agent_identity,
        shared_surface_path=str(data_dir / "shared" / "mnemosyne.db"),
    )
    assert provider._beam is not None
    return provider


def _call(provider, name, args):
    return json.loads(provider.handle_tool_call(name, args))


def _seed_private(provider, content, author="Sisyphus"):
    res = _call(provider, "mnemosyne_remember", {
        "content": content,
        "importance": 0.7,
        "source": "fact",
    })
    assert res["status"] == "stored"
    mid = res["memory_id"]
    # Tag the seed row as authored by `author` so we can verify preservation
    provider._beam.conn.execute(
        "UPDATE working_memory SET author_id = ? WHERE id = ?",
        (author, mid),
    )
    provider._beam.conn.commit()
    return mid


def _seed_surface(provider, content, kind="preference"):
    res = _call(provider, "mnemosyne_shared_remember", {
        "content": content,
        "kind": kind,
        "importance": 0.7,
    })
    assert res["status"] == "stored_shared"
    return res["memory_id"]


def _row(provider, mid, bank="private"):
    beam = provider._beam if bank == "private" else provider._surface_beam
    return beam.conn.execute(
        "SELECT id, content, author_id, validator, validated_at, "
        "validation_count, valid_until "
        "FROM working_memory WHERE id = ?",
        (mid,),
    ).fetchone()


def _validation_log(provider, mid, bank="private"):
    beam = provider._beam if bank == "private" else provider._surface_beam
    return beam.conn.execute(
        "SELECT validator, action, new_content, note "
        "FROM memory_validations WHERE memory_id = ? ORDER BY validation_id",
        (mid,),
    ).fetchall()


# --- Schema migration sanity ----------------------------------------------

def test_validator_columns_exist_after_init(tmp_path, monkeypatch):
    provider = _provider(tmp_path, monkeypatch)
    cols = [r[1] for r in provider._beam.conn.execute(
        "PRAGMA table_info(working_memory)"
    ).fetchall()]
    assert "validator" in cols
    assert "validated_at" in cols
    assert "validation_count" in cols


def test_memory_validations_table_exists(tmp_path, monkeypatch):
    provider = _provider(tmp_path, monkeypatch)
    row = provider._beam.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_validations'"
    ).fetchone()
    assert row is not None


def test_trim_trigger_exists(tmp_path, monkeypatch):
    provider = _provider(tmp_path, monkeypatch)
    row = provider._beam.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger' AND name='trim_validations_to_3'"
    ).fetchone()
    assert row is not None


# --- Action: attest --------------------------------------------------------

def test_validate_attest_preserves_author_and_records_validator(tmp_path, monkeypatch):
    provider = _provider(tmp_path, monkeypatch)
    mid = _seed_private(provider, "SSH key at /home/hilman/.ssh/pc", author="Sisyphus")

    res = _call(provider, "mnemosyne_validate", {
        "memory_id": mid,
        "action": "attest",
        "validator": "Albedo",
        "note": "confirmed during deploy",
    })

    assert res["status"] == "validation_attest"
    assert res["validator"] == "Albedo"
    assert res["author_id"] == "Sisyphus"

    row = _row(provider, mid)
    # author preserved, validator updated, count incremented
    assert row[2] == "Sisyphus"          # author_id
    assert row[3] == "Albedo"            # validator
    assert row[4] is not None            # validated_at
    assert row[5] == 1                   # validation_count


def test_validate_attest_falls_back_to_agent_identity(tmp_path, monkeypatch):
    provider = _provider(tmp_path, monkeypatch, agent_identity="Hopz")
    mid = _seed_private(provider, "Project at /tmp/proj", author="Sisyphus")

    res = _call(provider, "mnemosyne_validate", {
        "memory_id": mid,
        "action": "attest",
    })

    assert res["validator"] == "Hopz"


# --- Action: update --------------------------------------------------------

def test_validate_update_replaces_content_and_keeps_author(tmp_path, monkeypatch):
    provider = _provider(tmp_path, monkeypatch)
    mid = _seed_private(provider, "SSH key at /home/hilman/.ssh/pc", author="Sisyphus")

    res = _call(provider, "mnemosyne_validate", {
        "memory_id": mid,
        "action": "update",
        "validator": "Albedo",
        "new_content": "SSH key at /home/hilman/.ssh/laptop",
    })

    assert res["status"] == "validation_update"
    row = _row(provider, mid)
    assert row[1] == "SSH key at /home/hilman/.ssh/laptop"  # content updated
    assert row[2] == "Sisyphus"                               # author preserved
    assert row[3] == "Albedo"                                 # validator updated


def test_validate_update_requires_new_content(tmp_path, monkeypatch):
    provider = _provider(tmp_path, monkeypatch)
    mid = _seed_private(provider, "test fact")

    res = _call(provider, "mnemosyne_validate", {
        "memory_id": mid,
        "action": "update",
        "validator": "Albedo",
    })
    assert "new_content is required" in res["error"]


# --- Action: invalidate ----------------------------------------------------

def test_validate_invalidate_sets_valid_until(tmp_path, monkeypatch):
    provider = _provider(tmp_path, monkeypatch)
    mid = _seed_private(provider, "outdated fact about VPN", author="Sisyphus")

    res = _call(provider, "mnemosyne_validate", {
        "memory_id": mid,
        "action": "invalidate",
        "validator": "Hopz",
        "note": "user changed VPN",
    })

    assert res["status"] == "validation_invalidate"
    row = _row(provider, mid)
    assert row[6] is not None  # valid_until set
    assert row[3] == "Hopz"    # validator recorded
    assert row[2] == "Sisyphus"  # author preserved


# --- Action: delete --------------------------------------------------------

def test_validate_delete_removes_row(tmp_path, monkeypatch):
    provider = _provider(tmp_path, monkeypatch)
    mid = _seed_private(provider, "stale fact", author="Sisyphus")

    res = _call(provider, "mnemosyne_validate", {
        "memory_id": mid,
        "action": "delete",
        "validator": "Albedo",
    })

    assert res["status"] == "validation_delete"
    assert _row(provider, mid) is None


@pytest.mark.parametrize("provider_cls", PROVIDER_CLASSES,
                         ids=lambda c: c.__module__)
def test_validate_delete_cascades_support_rows(tmp_path, monkeypatch, provider_cls):
    """Deleting through the provider must not leave the memory's children behind.

    The provider used to remove only the working_memory row, so every delete
    orphaned the memory's gist (and its annotations and fallback embedding)
    permanently -- see #904.
    """
    provider = _provider(tmp_path, monkeypatch, provider_cls=provider_cls)
    conn = provider._beam.conn
    delete_id = _seed_private(provider, "stale fact with children")
    keep_id = _seed_private(provider, "unrelated fact with children")

    def seed_children(memory_id):
        conn.execute(
            "INSERT INTO gists (id, text, memory_id) VALUES (?, ?, ?)",
            (f"gist-{memory_id}", "gist text", memory_id),
        )
        conn.execute(
            "INSERT INTO memory_embeddings (memory_id, embedding_json) VALUES (?, ?)",
            (memory_id, "[0.1, 0.2]"),
        )
        conn.execute(
            "INSERT INTO annotations "
            "(memory_id, kind, value, source, confidence, created_at) "
            "VALUES (?, 'fact', 'note', 'test', 1.0, CURRENT_TIMESTAMP)",
            (memory_id,),
        )
        conn.commit()

    def count(table, memory_id):
        return conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE memory_id = ?", (memory_id,)
        ).fetchone()[0]

    seed_children(delete_id)
    seed_children(keep_id)
    keep_counts = {
        table: count(table, keep_id)
        for table in ("gists", "memory_embeddings", "annotations")
    }

    res = _call(provider, "mnemosyne_validate", {
        "memory_id": delete_id,
        "action": "delete",
        "validator": "Albedo",
    })

    assert res["status"] == "validation_delete"
    assert _row(provider, delete_id) is None
    for table in ("gists", "memory_embeddings", "annotations"):
        assert count(table, delete_id) == 0, f"{table} orphaned by delete"
        assert count(table, keep_id) == keep_counts[table]


@pytest.mark.parametrize("provider_cls", PROVIDER_CLASSES,
                         ids=lambda c: c.__module__)
def test_validate_delete_rolls_back_when_validation_log_fails(tmp_path, monkeypatch,
                                                              provider_cls):
    """A failure after the cascade must not leave the deletes pending.

    The handler returns validation_failed without rolling back, so the
    support-row and parent-row deletes stayed in the connection's implicit
    transaction and a later unrelated commit would make them permanent.
    """
    provider = _provider(tmp_path, monkeypatch, provider_cls=provider_cls)
    conn = provider._beam.conn
    mid = _seed_private(provider, "must survive a failed delete")
    conn.execute(
        "INSERT INTO gists (id, text, memory_id) VALUES (?, ?, ?)",
        (f"gist-{mid}", "gist text", mid),
    )
    conn.execute(
        "CREATE TRIGGER fail_validation_log "
        "BEFORE INSERT ON memory_validations "
        "BEGIN SELECT RAISE(ABORT, 'forced validation-log failure'); END"
    )
    conn.commit()
    # remember() writes a gist of its own, so count rather than assume one.
    gist_count = conn.execute(
        "SELECT COUNT(*) FROM gists WHERE memory_id = ?", (mid,)
    ).fetchone()[0]
    assert gist_count > 0

    res = _call(provider, "mnemosyne_validate", {
        "memory_id": mid,
        "action": "delete",
        "validator": "Albedo",
    })

    assert res["error"] == "validation_failed"
    assert "forced validation-log failure" in res["reason"]
    assert not conn.in_transaction
    assert _row(provider, mid) is not None
    assert conn.execute(
        "SELECT COUNT(*) FROM gists WHERE memory_id = ?", (mid,)
    ).fetchone()[0] == gist_count


@pytest.mark.parametrize("provider_cls", PROVIDER_CLASSES,
                         ids=lambda c: c.__module__)
def test_validate_delete_removes_vec_working_row(tmp_path, monkeypatch, provider_cls):
    """The provider delete removes the target's vector row when present."""
    provider = _provider(tmp_path, monkeypatch, provider_cls=provider_cls)
    conn = provider._beam.conn
    delete_id = _seed_private(provider, "delete with vec")
    keep_id = _seed_private(provider, "keep with vec")
    rowids = {
        mid: conn.execute(
            "SELECT rowid FROM working_memory WHERE id = ?", (mid,)
        ).fetchone()[0]
        for mid in (delete_id, keep_id)
    }
    conn.execute("DROP TABLE IF EXISTS vec_working")
    conn.execute("CREATE TABLE vec_working (rowid INTEGER PRIMARY KEY)")
    for rowid in rowids.values():
        conn.execute("INSERT INTO vec_working (rowid) VALUES (?)", (rowid,))
    conn.commit()

    res = _call(provider, "mnemosyne_validate", {
        "memory_id": delete_id,
        "action": "delete",
        "validator": "Albedo",
    })

    assert res["status"] == "validation_delete"
    assert conn.execute(
        "SELECT COUNT(*) FROM vec_working WHERE rowid = ?", (rowids[delete_id],)
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM vec_working WHERE rowid = ?", (rowids[keep_id],)
    ).fetchone()[0] == 1


@pytest.mark.parametrize("provider_cls", PROVIDER_CLASSES,
                         ids=lambda c: c.__module__)
def test_validate_delete_handles_missing_gists_table(tmp_path, monkeypatch, provider_cls):
    """The optional gists table may be absent on older databases."""
    provider = _provider(tmp_path, monkeypatch, provider_cls=provider_cls)
    mid = _seed_private(provider, "delete without gists")
    provider._beam.conn.execute("DROP TABLE gists")
    provider._beam.conn.commit()

    res = _call(provider, "mnemosyne_validate", {
        "memory_id": mid,
        "action": "delete",
        "validator": "Albedo",
    })

    assert res["status"] == "validation_delete"
    assert _row(provider, mid) is None


# --- Cross-bank: surface validation ---------------------------------------

def test_validate_works_on_shared_surface(tmp_path, monkeypatch):
    provider = _provider(tmp_path, monkeypatch)
    mid = _seed_surface(provider, "User prefers Tailscale over OpenVPN")

    res = _call(provider, "mnemosyne_validate", {
        "memory_id": mid,
        "action": "attest",
        "validator": "Albedo",
        "bank": "surface",
    })

    assert res["status"] == "validation_attest"
    assert res["bank"] == "surface"
    row = _row(provider, mid, bank="surface")
    assert row[3] == "Albedo"


# --- Ring buffer behavior --------------------------------------------------

def test_ring_buffer_keeps_only_last_three_validations(tmp_path, monkeypatch):
    provider = _provider(tmp_path, monkeypatch)
    mid = _seed_private(provider, "SSH key location", author="Sisyphus")

    for i, who in enumerate(["v1", "v2", "v3", "v4", "v5"]):
        _call(provider, "mnemosyne_validate", {
            "memory_id": mid,
            "action": "attest",
            "validator": who,
        })

    log = _validation_log(provider, mid)
    validators = [row[0] for row in log]
    assert len(log) == 3
    assert validators == ["v3", "v4", "v5"]


def test_validation_count_grows_unbounded(tmp_path, monkeypatch):
    """validation_count on the live row keeps growing even though the log is trimmed."""
    provider = _provider(tmp_path, monkeypatch)
    mid = _seed_private(provider, "tracked fact")

    for who in ["a", "b", "c", "d", "e", "f"]:
        _call(provider, "mnemosyne_validate", {
            "memory_id": mid,
            "action": "attest",
            "validator": who,
        })

    row = _row(provider, mid)
    assert row[5] == 6  # validation_count


# --- Error handling --------------------------------------------------------

def test_validate_unknown_memory_returns_error(tmp_path, monkeypatch):
    provider = _provider(tmp_path, monkeypatch)
    res = _call(provider, "mnemosyne_validate", {
        "memory_id": "nonexistent",
        "action": "attest",
    })
    assert res["error"] == "memory_not_found"


def test_validate_unknown_action_rejected(tmp_path, monkeypatch):
    provider = _provider(tmp_path, monkeypatch)
    mid = _seed_private(provider, "fact")
    res = _call(provider, "mnemosyne_validate", {
        "memory_id": mid,
        "action": "frobnicate",
    })
    assert "unknown action" in res["error"]


def test_validate_unknown_bank_rejected(tmp_path, monkeypatch):
    provider = _provider(tmp_path, monkeypatch)
    mid = _seed_private(provider, "fact")
    res = _call(provider, "mnemosyne_validate", {
        "memory_id": mid,
        "action": "attest",
        "bank": "weird",
    })
    assert "unknown bank" in res["error"]


def test_validate_missing_memory_id_rejected(tmp_path, monkeypatch):
    provider = _provider(tmp_path, monkeypatch)
    res = _call(provider, "mnemosyne_validate", {"action": "attest"})
    assert "memory_id is required" in res["error"]


# --- Cross-agent collaborative scenario (Master's SSH example) ------------

def test_collaborative_attestation_chain(tmp_path, monkeypatch):
    """Master's example: sisyphus authors, albedo updates, sisyphus updates,
    hopz attests. Author preserved throughout, ring buffer captures last 3."""
    provider = _provider(tmp_path, monkeypatch)
    mid = _seed_private(provider, "SSH key at /home/hilman/.ssh/pc", author="Sisyphus")

    _call(provider, "mnemosyne_validate", {
        "memory_id": mid, "action": "update", "validator": "Albedo",
        "new_content": "SSH key at /home/hilman/.ssh/laptop",
    })
    _call(provider, "mnemosyne_validate", {
        "memory_id": mid, "action": "update", "validator": "Sisyphus",
        "new_content": "SSH key at /home/hilman/.ssh/main",
    })
    _call(provider, "mnemosyne_validate", {
        "memory_id": mid, "action": "attest", "validator": "Hopz",
    })

    row = _row(provider, mid)
    assert row[2] == "Sisyphus"            # author preserved
    assert row[3] == "Hopz"                # latest validator
    assert row[5] == 3                     # validation_count
    assert "main" in row[1]                # latest content

    log = _validation_log(provider, mid)
    assert [r[0] for r in log] == ["Albedo", "Sisyphus", "Hopz"]
    assert [r[1] for r in log] == ["update", "update", "attest"]
