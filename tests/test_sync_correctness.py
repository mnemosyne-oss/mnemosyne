"""Regression tests for production-safe sync state discovery and cursors."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone

import pytest

from mnemosyne.core.memory import Mnemosyne
from mnemosyne.core.sync import SyncEncryption, SyncEngine


@pytest.fixture
def memory(tmp_path):
    return Mnemosyne(db_path=tmp_path / "memory.db")


def _event_ops(engine: SyncEngine):
    return [
        (row["memory_id"], row["operation"])
        for row in engine.conn.execute(
            "SELECT memory_id, operation FROM memory_events ORDER BY timestamp, event_id"
        ).fetchall()
    ]


def test_discover_local_mutations_emits_create_update_delete(memory):
    engine = SyncEngine(
        memory, device_id="device-a", allow_unscoped_sync=True
    )
    memory_id = memory.remember("version one", source="test", importance=0.7)

    first = engine.discover_local_mutations()
    assert first["created"] == 1
    assert _event_ops(engine) == [(memory_id, "CREATE")]

    assert memory.update(memory_id, content="version two", importance=0.8)
    second = engine.discover_local_mutations()
    assert second["updated"] == 1
    assert _event_ops(engine)[-1] == (memory_id, "UPDATE")

    assert memory.forget(memory_id)
    third = engine.discover_local_mutations()
    assert third["deleted"] == 1
    assert _event_ops(engine)[-1] == (memory_id, "DELETE")

    # A second scan is a no-op rather than another tombstone.
    assert engine.discover_local_mutations() == {
        "created": 0,
        "updated": 0,
        "deleted": 0,
        "events": [],
    }


def test_discovery_rolls_back_event_and_shadow_together(memory, monkeypatch):
    engine = SyncEngine(
        memory, device_id="device-a", allow_unscoped_sync=True
    )
    memory.remember("atomic discovery", source="test")

    def fail_state(*_args, **_kwargs):
        raise RuntimeError("injected shadow failure")

    monkeypatch.setattr(engine, "_state_set", fail_state)
    with pytest.raises(RuntimeError, match="injected shadow failure"):
        engine.discover_local_mutations()

    assert engine.conn.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0] == 0
    assert engine.conn.execute("SELECT COUNT(*) FROM sync_memory_state").fetchone()[0] == 0


def test_discover_local_mutations_does_not_stop_at_5000(memory):
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        (
            f"bulk-{i:05d}",
            f"bulk memory {i}",
            "test",
            now,
            "default",
            0.5,
            "{}",
            "unknown",
        )
        for i in range(5001)
    ]
    memory.beam.conn.executemany(
        """INSERT INTO working_memory
           (id, content, source, timestamp, session_id, importance, metadata_json, veracity)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    memory.beam.conn.commit()

    engine = SyncEngine(
        memory, device_id="device-a", allow_unscoped_sync=True
    )
    result = engine.discover_local_mutations()

    assert result["created"] == 5001
    assert engine.conn.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0] == 5001


def test_pull_cursor_orders_events_with_identical_timestamps(memory):
    engine = SyncEngine(
        memory, device_id="device-a", allow_unscoped_sync=True
    )
    first = engine.log_event("m1", "CREATE", {"content": "one"})
    second = engine.log_event("m2", "CREATE", {"content": "two"})
    shared_timestamp = "2026-07-11T10:00:00+00:00"
    engine.conn.execute(
        "UPDATE memory_events SET timestamp = ?, timestamp_epoch = ? WHERE event_id IN (?, ?)",
        (
            shared_timestamp,
            datetime.fromisoformat(shared_timestamp).timestamp(),
            first.event_id,
            second.event_id,
        ),
    )
    engine.conn.commit()

    page_one = engine.pull_changes(limit=1)
    assert page_one["has_more"] is True
    assert page_one["total"] == 1

    page_two = engine.pull_changes(since_cursor=page_one["next_cursor"], limit=1)
    assert page_two["total"] == 1
    assert page_two["events"][0]["event_id"] != page_one["events"][0]["event_id"]


def test_incoming_older_event_does_not_overwrite_newer_local_state(memory):
    engine = SyncEngine(memory, device_id="local")
    memory_id = memory.remember("local newest", source="test", importance=0.8)
    engine.discover_local_mutations()
    local_event = engine.conn.execute(
        "SELECT * FROM memory_events WHERE memory_id = ? ORDER BY timestamp DESC LIMIT 1",
        (memory_id,),
    ).fetchone()

    older = {
        "event_id": "remote-older",
        "memory_id": memory_id,
        "operation": "UPDATE",
        "timestamp": "2000-01-01T00:00:00+00:00",
        "device_id": "remote",
        "payload": json.dumps({"content": "remote stale", "source": "sync"}),
        "parent_event_ids": "[]",
        "importance": 1.0,
        "event_hash": "remote-older-hash",
    }
    result = engine.push_changes([older])

    assert result["conflicts"] == 1
    assert result["errors"] == 0
    assert memory.get(memory_id)["content"] == "local newest"
    assert local_event["event_id"] != older["event_id"]
    exported_ids = {
        event["event_id"] for event in engine.pull_changes(limit=100)["events"]
    }
    assert older["event_id"] not in exported_ids


def test_pull_only_discovers_local_mutation_before_conflict_resolution(memory, monkeypatch):
    import urllib.request

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return json.dumps(
                {
                    "events": [remote_event],
                    "next_cursor": None,
                    "has_more": False,
                }
            ).encode()

    engine = SyncEngine(
        memory,
        device_id="local",
        allow_unscoped_sync=True,
    )
    memory_id = memory.remember("local v1", source="test")
    engine.discover_local_mutations()
    assert memory.update(memory_id, content="local v2")
    remote_event = {
        "event_id": "remote-stale-pull",
        "memory_id": memory_id,
        "operation": "UPDATE",
        "timestamp": "2000-01-01T00:00:00+00:00",
        "device_id": "remote",
        "payload": json.dumps({"content": "remote stale", "source": "sync"}),
        "parent_event_ids": "[]",
        "importance": 0.5,
        "event_hash": None,
    }
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *_args, **_kwargs: FakeResponse()
    )

    result = engine.sync_with("https://relay.invalid", mode="pull")

    assert not result["errors"]
    assert result["pull"]["conflicts"] == 1
    assert memory.get(memory_id)["content"] == "local v2"


def test_synced_update_preserves_explicit_null_optional_fields(memory, tmp_path):
    source_memory = Mnemosyne(db_path=tmp_path / "nullable-source.db")
    source = SyncEngine(source_memory, device_id="source")
    receiver = SyncEngine(memory, device_id="receiver")
    memory_id = "nullable-memory"

    created = source.log_event(
        memory_id,
        "CREATE",
        {
            "content": "version one",
            "source": "test",
            "memory_type": "note",
            "valid_until": "2030-01-01",
        },
    ).to_dict()
    assert receiver.push_changes([created])["accepted"] == 1

    updated = source.log_event(
        memory_id,
        "UPDATE",
        {
            "content": "version two",
            "source": "test",
            "memory_type": None,
            "valid_until": None,
        },
    ).to_dict()
    assert receiver.push_changes([updated])["accepted"] == 1

    row = memory.beam.conn.execute(
        "SELECT content, memory_type, valid_until FROM working_memory WHERE id = ?",
        (memory_id,),
    ).fetchone()
    assert tuple(row) == ("version two", None, None)


def test_pull_stops_when_remote_cursor_does_not_advance(memory, monkeypatch):
    import urllib.request

    calls = 0
    remote_event = {
        "event_id": "stagnant-cursor-event",
        "memory_id": "stagnant-cursor-memory",
        "operation": "CREATE",
        "timestamp": "2026-07-11T10:00:00+00:00",
        "device_id": "remote",
        "payload": json.dumps({"content": "remote", "source": "sync"}),
        "parent_event_ids": "[]",
        "importance": 0.5,
        "event_hash": None,
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return json.dumps(
                {
                    "events": [remote_event],
                    "next_cursor": None,
                    "has_more": True,
                }
            ).encode()

    def fake_urlopen(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    engine = SyncEngine(memory, device_id="local", allow_unscoped_sync=True)

    result = engine.sync_with("https://relay.invalid", mode="pull")

    assert calls == 1
    assert result["pull"]["accepted"] == 1
    assert result["errors"] == [
        "remote reported has_more without advancing next_cursor"
    ]


def test_duplicate_event_id_requires_exact_event_match(memory):
    source_memory = Mnemosyne(db_path=memory.beam.db_path.parent / "duplicate-source.db")
    source = SyncEngine(source_memory, device_id="source")
    receiver = SyncEngine(memory, device_id="receiver")
    event = source.log_event(
        "duplicate-memory", "CREATE", {"content": "original", "source": "test"}
    ).to_dict()
    assert receiver.push_changes([event])["accepted"] == 1
    tampered = dict(event)
    tampered["payload"] = json.dumps({"content": "tampered", "source": "test"})
    tampered["event_hash"] = None

    result = receiver.push_changes([tampered])

    assert result["errors"] == 1
    assert result["duplicates"] == 0
    assert result["acknowledged_event_ids"] == []
    assert memory.get("duplicate-memory")["content"] == "original"


def test_plaintext_delete_without_payload_is_applied(memory):
    engine = SyncEngine(memory, device_id="receiver")
    memory_id = memory.remember("delete me", source="test", scope="global")
    event = {
        "event_id": "plaintext-delete-no-payload",
        "memory_id": memory_id,
        "operation": "DELETE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device_id": "source",
        "payload": None,
        "parent_event_ids": "[]",
        "importance": 0.5,
        "event_hash": None,
    }

    result = engine.push_changes([event])

    assert result["accepted"] == 1
    assert result["acknowledged_event_ids"] == [event["event_id"]]
    assert memory.get(memory_id) is None


def test_cross_session_apply_keeps_destination_ownership_and_lifecycle(tmp_path):
    source_memory = Mnemosyne(db_path=tmp_path / "source-session.db", session_id="source")
    destination_memory = Mnemosyne(
        db_path=tmp_path / "destination-session.db", session_id="destination"
    )
    source = SyncEngine(source_memory, device_id="source")
    destination = SyncEngine(destination_memory, device_id="destination")

    memory_id = source_memory.remember(
        "cross-session v1", source="test", scope="session"
    )
    created = source.discover_local_mutations()["events"]
    payload = json.loads(created[0]["payload"])
    assert set(payload).isdisjoint(
        {"session_id", "scope", "author_id", "author_type", "channel_id", "trust_tier"}
    )
    assert destination.push_changes(created)["errors"] == 0
    destination_row = destination.conn.execute(
        "SELECT content, session_id, scope FROM working_memory WHERE id = ?",
        (memory_id,),
    ).fetchone()
    assert tuple(destination_row) == ("cross-session v1", "destination", "global")

    assert source_memory.update(memory_id, content="cross-session v2")
    updated = source.discover_local_mutations()["events"]
    assert destination.push_changes(updated)["errors"] == 0
    assert destination_memory.get(memory_id)["content"] == "cross-session v2"

    assert source_memory.forget(memory_id)
    deleted = source.discover_local_mutations()["events"]
    assert destination.push_changes(deleted)["errors"] == 0
    assert destination_memory.get(memory_id) is None


def test_future_timestamp_is_rejected_without_acknowledgement(memory):
    engine = SyncEngine(memory, device_id="receiver", max_future_skew_seconds=300)
    event = {
        "event_id": "future-event",
        "memory_id": "future-memory",
        "operation": "CREATE",
        "timestamp": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        "device_id": "sender",
        "payload": json.dumps({"content": "future poison", "source": "test"}),
        "parent_event_ids": "[]",
        "importance": 1.0,
        "event_hash": "future-hash",
    }

    result = engine.push_changes([event])

    assert result["errors"] == 1
    assert result["acknowledged_event_ids"] == []
    assert memory.get("future-memory") is None


def test_wrong_encryption_key_fails_without_advancing_or_mutating(memory):
    sender = SyncEncryption.from_config(SyncEncryption.generate_key())
    receiver = SyncEncryption.from_config(SyncEncryption.generate_key())
    assert sender is not None
    assert receiver is not None
    engine = SyncEngine(memory, device_id="receiver", encryption=receiver)
    encrypted = sender.encrypt({"content": "must not be stored", "source": "sync"})
    event = {
        "event_id": "encrypted-event",
        "memory_id": "secret-memory",
        "operation": "CREATE",
        "timestamp": "2026-07-11T10:00:00+00:00",
        "device_id": "sender",
        "payload": encrypted,
        "parent_event_ids": "[]",
        "importance": 0.9,
        "event_hash": "encrypted-event-hash",
    }

    result = engine.push_changes([event])

    assert result["errors"] == 1
    assert result["accepted"] == 0
    assert memory.get("secret-memory") is None
    assert engine.conn.execute(
        "SELECT COUNT(*) FROM memory_events WHERE event_id = 'encrypted-event'"
    ).fetchone()[0] == 0


def test_incoming_event_rolls_back_after_post_materialization_failure(
    memory, monkeypatch
):
    source_memory = Mnemosyne(db_path=memory.beam.db_path.parent / "pending-source.db")
    source = SyncEngine(source_memory, device_id="source")
    destination = SyncEngine(memory, device_id="destination")
    memory_id = source_memory.remember("pending retry", source="test", scope="global")
    event = source.discover_local_mutations()["events"][0]
    original_mark = destination._mark_event_state

    def fail_mark(*_args, **_kwargs):
        raise RuntimeError("injected post-materialization failure")

    monkeypatch.setattr(destination, "_mark_event_state", fail_mark)
    first = destination.push_changes([event])
    assert first["errors"] == 1
    assert memory.get(memory_id) is None
    assert destination.conn.execute(
        "SELECT COUNT(*) FROM memory_events WHERE event_id = ?", (event["event_id"],)
    ).fetchone()[0] == 0
    assert destination.conn.execute(
        "SELECT COUNT(*) FROM sync_memory_state WHERE memory_id = ?", (memory_id,)
    ).fetchone()[0] == 0

    monkeypatch.setattr(destination, "_mark_event_state", original_mark)
    retried = destination.push_changes([event])
    assert retried["errors"] == 0
    assert retried["accepted"] == 1
    assert destination.conn.execute(
        "SELECT apply_state FROM memory_events WHERE event_id = ?", (event["event_id"],)
    ).fetchone()[0] == "applied"
    assert destination.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE id = ?", (memory_id,)
    ).fetchone()[0] == 1


def test_blind_relay_never_materializes_encrypted_delete(tmp_path):
    relay_memory = Mnemosyne(db_path=tmp_path / "relay-delete.db")
    sender_memory = Mnemosyne(db_path=tmp_path / "sender-delete.db")
    encryption = SyncEncryption.from_config(SyncEncryption.generate_key())
    assert encryption is not None
    relay = SyncEngine(
        relay_memory, device_id="relay", require_encryption=True, relay_mode=True
    )
    sender = SyncEngine(sender_memory, device_id="sender", encryption=encryption)
    victim_id = relay_memory.remember("relay-local victim", source="test", scope="global")
    encrypted_delete = sender.log_event(
        victim_id, "DELETE", {"deleted": True}
    ).to_dict()
    encrypted_delete["_transport_authenticated"] = True

    result = relay.push_changes([encrypted_delete])

    assert result["accepted"] == 1
    assert result["errors"] == 0
    assert relay_memory.get(victim_id)["content"] == "relay-local victim"
    assert relay.conn.execute(
        "SELECT apply_state FROM memory_events WHERE event_id = ?",
        (encrypted_delete["event_id"],),
    ).fetchone()[0] == "relayed"


def test_blind_relay_accepts_encrypted_event_without_materializing(memory):
    sender = SyncEncryption.from_config(SyncEncryption.generate_key())
    assert sender is not None
    engine = SyncEngine(
        memory,
        device_id="relay",
        encryption=None,
        require_encryption=True,
        relay_mode=True,
    )
    ciphertext = "mne1:" + sender.encrypt(
        {"content": "relay cannot read this", "source": "sync"}
    )
    event = {
        "event_id": "opaque-event",
        "memory_id": "opaque-memory",
        "operation": "CREATE",
        "timestamp": "2026-07-11T10:00:00+00:00",
        "device_id": "sender",
        "payload": ciphertext,
        "parent_event_ids": "[]",
        "importance": 0.9,
        "event_hash": "opaque-event-hash",
        "_transport_authenticated": True,
    }

    result = engine.push_changes([event])

    assert result["accepted"] == 1
    assert result["errors"] == 0
    assert memory.get("opaque-memory") is None
    stored = engine.conn.execute(
        "SELECT payload FROM memory_events WHERE event_id = 'opaque-event'"
    ).fetchone()[0]
    assert stored == ciphertext
    assert "relay cannot read this" not in stored


def test_blind_relay_rejects_malformed_encrypted_wire_payload(memory):
    engine = SyncEngine(
        memory,
        device_id="relay",
        require_encryption=True,
        relay_mode=True,
    )
    event = {
        "event_id": "poison-event",
        "memory_id": "poison-memory",
        "operation": "CREATE",
        "timestamp": "2026-07-11T10:00:00+00:00",
        "device_id": "sender",
        "payload": "mne1:not-valid-base64!",
        "parent_event_ids": "[]",
        "importance": 0.5,
        "event_hash": "poison-hash",
        "_transport_authenticated": True,
    }

    result = engine.push_changes([event])

    assert result["accepted"] == 0
    assert result["errors"] == 1
    assert engine.conn.execute(
        "SELECT COUNT(*) FROM memory_events WHERE event_id = 'poison-event'"
    ).fetchone()[0] == 0


def test_blind_relay_rejects_unauthenticated_structural_ciphertext(memory):
    engine = SyncEngine(
        memory,
        device_id="relay",
        require_encryption=True,
        relay_mode=True,
    )
    event = {
        "event_id": "unauthenticated-poison",
        "memory_id": "poison-memory",
        "operation": "CREATE",
        "timestamp": "2026-07-11T10:00:00+00:00",
        "device_id": "sender",
        "payload": "mne1:" + base64.urlsafe_b64encode(b"x" * 64).decode("ascii"),
        "parent_event_ids": "[]",
        "importance": 0.5,
        "event_hash": None,
    }

    result = engine.push_changes([event])

    assert result["accepted"] == 0
    assert result["errors"] == 1
    assert "authenticated transport" in result["details"][0]
    assert result["acknowledged_event_ids"] == []
    assert engine.conn.execute(
        "SELECT COUNT(*) FROM memory_events WHERE event_id = 'unauthenticated-poison'"
    ).fetchone()[0] == 0


def test_existing_encrypted_events_reject_wrong_startup_key(memory):
    from cryptography.fernet import InvalidToken

    good = SyncEncryption.from_config(SyncEncryption.generate_key())
    wrong = SyncEncryption.from_config(SyncEncryption.generate_key())
    assert good is not None
    assert wrong is not None
    writer = SyncEngine(memory, device_id="writer", encryption=good)
    writer.log_event("existing-encrypted", "CREATE", {"content": "secret"})

    with pytest.raises(InvalidToken):
        SyncEngine(memory, device_id="reader", encryption=wrong)


def test_encrypted_engine_rewraps_legacy_plaintext_outbox(memory):
    plain = SyncEngine(memory, device_id="source")
    event = plain.log_event(
        "legacy-memory", "CREATE", {"content": "legacy", "source": "test"}
    )
    assert not event.payload.startswith("mne1:")
    encryption = SyncEncryption.from_config(SyncEncryption.generate_key())
    assert encryption is not None

    migrated = SyncEngine(
        memory,
        device_id="source",
        encryption=encryption,
        require_encryption=True,
    )
    stored = migrated.conn.execute(
        "SELECT payload FROM memory_events WHERE event_id = ?", (event.event_id,)
    ).fetchone()[0]

    assert stored.startswith("mne1:")
    pulled = migrated.pull_changes(limit=10)["events"]
    receiver_memory = Mnemosyne(db_path=memory.beam.db_path.parent / "legacy-receiver.db")
    receiver = SyncEngine(receiver_memory, device_id="receiver", encryption=encryption)
    result = receiver.push_changes(pulled)
    assert result["accepted"] == 1
    assert receiver_memory.get("legacy-memory")["content"] == "legacy"


def test_surface_marker_survives_beam_session_change(tmp_path):
    db_path = tmp_path / "durable-surface.db"
    first_memory = Mnemosyne(db_path=db_path, session_id="surface-session-a")
    first = SyncEngine(
        first_memory,
        device_id="first",
        surface_only=True,
        initialize_surface=True,
    )
    memory_id = first_memory.remember(
        "durable surface", source="test", scope="global"
    )
    assert first.discover_local_mutations()["created"] == 1

    second_memory = Mnemosyne(db_path=db_path, session_id="surface-session-b")
    second = SyncEngine(second_memory, device_id="second", surface_only=True)
    result = second.discover_local_mutations()

    assert result["deleted"] == 0
    assert second._working_payload(memory_id)["content"] == "durable surface"


def test_surface_only_initialization_rejects_private_session_rows(tmp_path):
    destination = Mnemosyne(db_path=tmp_path / "surface-boundary.db", session_id="surface")
    private_id = destination.remember("private", source="test", scope="session")

    with pytest.raises(ValueError, match="requires an empty working_memory"):
        SyncEngine(
            destination,
            device_id="receiver",
            surface_only=True,
            initialize_surface=True,
        )

    assert destination.get(private_id)["content"] == "private"
    marker = destination.beam.conn.execute(
        """SELECT value FROM sync_meta
           WHERE key = 'surface_db_id'"""
    ).fetchone()
    assert marker is None


def test_surface_initialization_rejects_existing_global_rows(tmp_path):
    memory = Mnemosyne(db_path=tmp_path / "existing-global.db", session_id="surface")
    memory_id = memory.remember("not implicitly shared", source="test", scope="global")

    with pytest.raises(ValueError, match="requires an empty working_memory"):
        SyncEngine(memory, surface_only=True, initialize_surface=True)

    assert memory.get(memory_id)["content"] == "not implicitly shared"
    marker = memory.beam.conn.execute(
        "SELECT value FROM sync_meta WHERE key = 'surface_db_id'"
    ).fetchone()
    assert marker is None


def test_surface_event_payload_scrubs_private_metadata(tmp_path):
    memory = Mnemosyne(db_path=tmp_path / "metadata-surface.db", session_id="surface")
    engine = SyncEngine(
        memory,
        device_id="surface",
        surface_only=True,
        initialize_surface=True,
    )
    memory.remember(
        "safe shared row",
        source="test",
        scope="global",
        metadata={
            "source_profile_session": "private-profile",
            "nested": {"author_id": "private-author", "safe": "kept"},
        },
    )

    event = engine.discover_local_mutations()["events"][0]
    payload = json.loads(event["payload"])
    metadata = json.loads(payload["metadata_json"])

    assert "source_profile_session" not in metadata
    assert "author_id" not in metadata["nested"]
    assert metadata["nested"]["safe"] == "kept"


def test_public_log_event_scrubs_private_payload_fields(memory):
    encryption = SyncEncryption.from_config(SyncEncryption.generate_key())
    assert encryption is not None
    engine = SyncEngine(memory, device_id="writer", encryption=encryption)

    event = engine.log_event(
        "manual-event",
        "CREATE",
        {
            "content": "safe",
            "session_id": "private-session",
            "author_id": "private-author",
            "metadata_json": json.dumps(
                {
                    "source_profile_session": "private-profile",
                    "nested": {"author_type": "private", "safe": "kept"},
                }
            ),
        },
    )
    assert event.payload is not None
    token = event.payload.removeprefix("mne1:")
    envelope = encryption.decrypt(token)
    payload = envelope["payload"]
    metadata = json.loads(payload["metadata_json"])

    assert "session_id" not in payload
    assert "author_id" not in payload
    assert "source_profile_session" not in metadata
    assert "author_type" not in metadata["nested"]
    assert metadata["nested"]["safe"] == "kept"


def test_encrypted_metadata_tampering_is_rejected(memory):
    encryption = SyncEncryption.from_config(SyncEncryption.generate_key())
    assert encryption is not None
    sender_memory = Mnemosyne(db_path=memory.beam.db_path.parent / "sender.db")
    sender = SyncEngine(sender_memory, device_id="sender", encryption=encryption)
    receiver = SyncEngine(memory, device_id="receiver", encryption=encryption)
    event = sender.log_event(
        "tamper-target", "CREATE", {"content": "authenticated", "source": "test"}
    ).to_dict()
    event["operation"] = "DELETE"

    result = receiver.push_changes([event])

    assert result["errors"] == 1
    assert result["accepted"] == 0
    assert memory.get("tamper-target") is None


def test_discovery_recovers_from_event_logged_before_shadow_state(memory):
    engine = SyncEngine(
        memory, device_id="device-a", allow_unscoped_sync=True
    )
    memory_id = memory.remember("already logged", source="test")
    payload = engine._working_payload(memory_id)
    assert payload is not None
    engine.log_event(memory_id, "CREATE", payload)

    result = engine.discover_local_mutations()

    assert result["created"] == 0
    assert engine.conn.execute(
        "SELECT COUNT(*) FROM memory_events WHERE memory_id = ?", (memory_id,)
    ).fetchone()[0] == 1
    assert engine.conn.execute(
        "SELECT COUNT(*) FROM sync_memory_state WHERE memory_id = ?", (memory_id,)
    ).fetchone()[0] == 1


def test_upgrade_bootstrap_emits_delete_for_legacy_logged_missing_row(memory):
    engine = SyncEngine(
        memory, device_id="device-a", allow_unscoped_sync=True
    )
    memory_id = memory.remember("deleted before shadow migration", source="test")
    payload = engine._working_payload(memory_id)
    assert payload is not None
    create_event = engine.log_event(memory_id, "CREATE", payload)
    assert memory.forget(memory_id)

    result = engine.discover_local_mutations()

    assert result["deleted"] == 1
    delete_event = result["events"][0]
    assert delete_event["operation"] == "DELETE"
    assert json.loads(delete_event["parent_event_ids"]) == [create_event.event_id]
    assert engine.conn.execute(
        "SELECT last_operation FROM sync_memory_state WHERE memory_id = ?", (memory_id,)
    ).fetchone()[0] == "DELETE"


def test_discovery_recovers_from_delete_event_logged_before_shadow_state(memory):
    engine = SyncEngine(
        memory, device_id="device-a", allow_unscoped_sync=True
    )
    memory_id = memory.remember("delete crash", source="test")
    engine.discover_local_mutations()
    previous = engine.conn.execute(
        "SELECT event_id FROM sync_memory_state WHERE memory_id = ?", (memory_id,)
    ).fetchone()[0]
    assert memory.forget(memory_id)
    engine.log_event(memory_id, "DELETE", {"deleted": True}, parent_event_ids=[previous])

    result = engine.discover_local_mutations()

    assert result["deleted"] == 0
    assert engine.conn.execute(
        "SELECT COUNT(*) FROM memory_events WHERE memory_id = ? AND operation = 'DELETE'",
        (memory_id,),
    ).fetchone()[0] == 1


def test_push_database_is_fail_closed_to_one_relay(memory):
    engine = SyncEngine(
        memory, device_id="device-a", allow_unscoped_sync=True
    )
    engine._meta_set("configured_push_remote", "https://relay-a.example")

    result = engine.sync_with("https://relay-b.example", mode="push")

    assert result["errors"] == [
        "this sync database is pinned to a different push relay"
    ]
    assert result["push"] is None


def test_empty_outbox_is_pinned_to_first_push_relay(memory):
    engine = SyncEngine(
        memory, device_id="device-a", allow_unscoped_sync=True
    )

    first = engine.sync_with("https://relay-a.example", mode="push")
    second = engine.sync_with("https://relay-b.example", mode="push")

    assert first["errors"] == []
    assert engine._meta_get("configured_push_remote") == "https://relay-a.example"
    assert second["errors"] == [
        "this sync database is pinned to a different push relay"
    ]
    assert second["push"] is None


def test_failed_push_keeps_local_event_in_outbox(memory):
    engine = SyncEngine(
        memory, device_id="device-a", allow_unscoped_sync=True
    )
    memory.remember("must retry", source="test")

    result = engine.sync_with("http://127.0.0.1:1", mode="push")

    assert result["errors"]
    assert engine.conn.execute(
        "SELECT COUNT(*) FROM memory_events WHERE device_id = ? AND synced_at IS NULL",
        (engine.device_id,),
    ).fetchone()[0] == 1


def test_push_rejects_acknowledgements_outside_current_batch(memory, monkeypatch):
    import urllib.request

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return json.dumps(
                {
                    "accepted": 1,
                    "duplicates": 0,
                    "conflicts": 0,
                    "errors": 0,
                    "acknowledged_event_ids": ["foreign-event-id"],
                }
            ).encode()

    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    engine = SyncEngine(
        memory, device_id="device-a", allow_unscoped_sync=True
    )
    memory.remember("ack boundary", source="test")

    result = engine.sync_with("https://relay.invalid", mode="push")

    assert "outside the current push batch" in result["errors"][0]
    assert engine.conn.execute(
        "SELECT COUNT(*) FROM memory_events WHERE device_id = ? AND synced_at IS NULL",
        (engine.device_id,),
    ).fetchone()[0] == 1


def test_partial_ack_atomically_pins_first_relay(memory, monkeypatch):
    import urllib.request

    requests = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return json.dumps(self.payload).encode()

    def fake_urlopen(request, **_kwargs):
        body = json.loads(request.data.decode())
        requests.append(body)
        first_id = body["events"][0]["event_id"]
        return FakeResponse(
            {
                "accepted": 1,
                "duplicates": 0,
                "conflicts": 0,
                "errors": 0,
                "acknowledged_event_ids": [first_id],
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    engine = SyncEngine(
        memory, device_id="device-a", allow_unscoped_sync=True
    )
    memory.remember("first", source="test")
    memory.remember("second", source="test")

    first = engine.sync_with("https://relay-a.invalid", mode="push")
    second = engine.sync_with("https://relay-b.invalid", mode="push")

    assert first["errors"] == ["remote only partially acknowledged a push batch"]
    assert engine._meta_get("configured_push_remote") == "https://relay-a.invalid"
    assert second["errors"] == [
        "this sync database is pinned to a different push relay"
    ]
    assert len(requests) == 1


def test_wrong_key_does_not_advance_pull_cursor_and_can_retry(tmp_path):
    from mnemosyne.core.sync_server import run_sync_server

    relay = Mnemosyne(db_path=tmp_path / "relay-retry.db")
    sender_memory = Mnemosyne(db_path=tmp_path / "sender-retry.db")
    receiver_memory = Mnemosyne(db_path=tmp_path / "receiver-retry.db")
    good_encryption = SyncEncryption.from_config(SyncEncryption.generate_key())
    wrong_encryption = SyncEncryption.from_config(SyncEncryption.generate_key())
    assert good_encryption is not None
    assert wrong_encryption is not None
    server = run_sync_server(
        host="127.0.0.1",
        port=0,
        beam_instance=relay,
        device_id="relay",
        api_key="test-api-key",
        daemon=True,
        initialize_surface=True,
    )
    remote = f"http://127.0.0.1:{server.server_address[1]}"
    sender = SyncEngine(
        sender_memory,
        device_id="sender",
        encryption=good_encryption,
        surface_only=True,
        initialize_surface=True,
    )
    receiver_wrong = SyncEngine(
        receiver_memory,
        device_id="receiver",
        encryption=wrong_encryption,
        surface_only=True,
        initialize_surface=True,
    )

    try:
        memory_id = sender_memory.remember(
            "retry after key fix", source="test", scope="global"
        )
        assert not sender.sync_with(remote, mode="push", api_key="test-api-key")["errors"]

        failed = receiver_wrong.sync_with(remote, mode="pull", api_key="test-api-key")
        assert failed["errors"]
        assert receiver_wrong._meta_get(f"last_pull_cursor_{remote}") is None
        assert receiver_memory.get(memory_id) is None
        assert receiver_wrong.conn.execute(
            "SELECT COUNT(*) FROM memory_events WHERE memory_id = ?", (memory_id,)
        ).fetchone()[0] == 0

        receiver_fixed = SyncEngine(
            receiver_memory,
            device_id="receiver",
            encryption=good_encryption,
            surface_only=True,
        )
        retried = receiver_fixed.sync_with(remote, mode="pull", api_key="test-api-key")
        assert not retried["errors"]
        assert receiver_memory.get(memory_id)["content"] == "retry after key fix"
    finally:
        server.shutdown()
        server.server_close()


def test_encrypted_e2e_propagates_create_update_delete(tmp_path):
    from mnemosyne.core.sync_server import run_sync_server

    relay = Mnemosyne(db_path=tmp_path / "relay.db")
    client_a = Mnemosyne(db_path=tmp_path / "client-a.db")
    client_b = Mnemosyne(db_path=tmp_path / "client-b.db")
    encryption = SyncEncryption.from_config(SyncEncryption.generate_key())
    assert encryption is not None
    server = run_sync_server(
        host="127.0.0.1",
        port=0,
        beam_instance=relay,
        device_id="relay",
        api_key="test-api-key",
        daemon=True,
        initialize_surface=True,
    )
    remote = f"http://127.0.0.1:{server.server_address[1]}"
    engine_a = SyncEngine(
        client_a,
        device_id="client-a",
        encryption=encryption,
        surface_only=True,
        initialize_surface=True,
    )
    engine_b = SyncEngine(
        client_b,
        device_id="client-b",
        encryption=encryption,
        surface_only=True,
        initialize_surface=True,
    )

    try:
        memory_id = client_a.remember(
            "shared version one", source="surface", importance=0.8, scope="global"
        )
        assert not engine_a.sync_with(remote, api_key="test-api-key")["errors"]
        first_pull = engine_b.sync_with(remote, api_key="test-api-key")
        assert not first_pull["errors"]
        assert client_b.get(memory_id)["content"] == "shared version one"

        assert client_a.update(memory_id, content="shared version two", importance=0.9)
        update_push = engine_a.sync_with(remote, api_key="test-api-key")
        update_pull = engine_b.sync_with(remote, api_key="test-api-key")
        assert update_push["push"]["discovered"]["updated"] == 1
        assert not update_push["errors"]
        assert not update_pull["errors"]
        assert client_b.get(memory_id)["content"] == "shared version two"

        assert client_a.forget(memory_id)
        delete_push = engine_a.sync_with(remote, api_key="test-api-key")
        delete_pull = engine_b.sync_with(remote, api_key="test-api-key")
        assert delete_push["push"]["discovered"]["deleted"] == 1
        assert not delete_push["errors"]
        assert not delete_pull["errors"]
        assert client_b.get(memory_id) is None

        assert relay.beam.conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == 0
        operations = {
            row[0]
            for row in relay.beam.conn.execute(
                "SELECT operation FROM memory_events WHERE memory_id = ?", (memory_id,)
            ).fetchall()
        }
        assert operations == {"CREATE", "UPDATE", "DELETE"}
    finally:
        server.shutdown()
        server.server_close()


def test_sync_with_drains_more_than_5000_opaque_events(tmp_path):
    from mnemosyne.core.sync_server import run_sync_server

    relay = Mnemosyne(db_path=tmp_path / "relay-pages.db")
    client = Mnemosyne(db_path=tmp_path / "client-pages.db")
    relay_engine = SyncEngine(relay, device_id="relay")
    shared_timestamp = datetime.now(timezone.utc).isoformat()
    shared_epoch = datetime.fromisoformat(shared_timestamp).timestamp()
    opaque_payload = "mne1:" + base64.urlsafe_b64encode(b"x" * 64).decode("ascii")
    relay_engine.conn.executemany(
        """INSERT INTO memory_events
           (event_id, memory_id, operation, timestamp, timestamp_epoch, device_id,
            surface_id, payload, parent_event_ids, importance, event_hash)
           VALUES (?, ?, 'CREATE', ?, ?, 'source', 'shared-surface-v1', ?, '[]', 0.5, ?)""",
        [
            (
                f"event-{index:05d}",
                f"page-{index:05d}",
                shared_timestamp,
                shared_epoch,
                opaque_payload,
                f"hash-{index:05d}",
            )
            for index in range(5001)
        ],
    )
    relay_engine.conn.commit()
    server = run_sync_server(
        host="127.0.0.1",
        port=0,
        beam_instance=relay,
        device_id="relay-server",
        api_key="test-api-key",
        daemon=True,
        initialize_surface=True,
    )
    remote = f"http://127.0.0.1:{server.server_address[1]}"
    client_engine = SyncEngine(
        client,
        device_id="client-relay",
        require_encryption=True,
        relay_mode=True,
        surface_only=True,
        initialize_surface=True,
        claim_surface_rows=False,
    )

    try:
        result = client_engine.sync_with(
            remote, mode="pull", api_key="test-api-key"
        )
        assert not result["errors"]
        assert result["pull"]["events_fetched"] == 5001
        assert result["pull"]["accepted"] == 5001
        assert result["pull"]["batches"] == 6
        assert client.beam.conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == 0
        assert client.beam.conn.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0] == 5001
    finally:
        server.shutdown()
        server.server_close()
