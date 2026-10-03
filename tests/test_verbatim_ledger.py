"""Best-effort boundary release: bounded evidence, no context-exactness claim."""
import json
import sqlite3
from types import SimpleNamespace

import pytest

from mnemosyne.core import verbatim_ledger as vl

ANCHOR = {"role": "assistant", "content": "Previous boundary tail response."}


def transcript(raw, role="user"):
    return [ANCHOR, {"role": role, "content": raw}]


@pytest.fixture
def store():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE working_memory (id TEXT PRIMARY KEY, session_id, source, content, metadata_json)")

    def remember(content, source="conversation", metadata=None, **kwargs):
        row = conn.execute("SELECT id FROM working_memory WHERE content = ?", (content,)).fetchone()
        if row:
            return row[0]
        mid = str(conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0])
        conn.execute("INSERT INTO working_memory VALUES (?, 'beam-session', ?, ?, ?)",
                     (mid, source, content, json.dumps(metadata or {})))
        conn.commit()
        return mid

    yield SimpleNamespace(conn=conn, session_id="beam-session", remember=remember)
    conn.close()


def armed():
    ledger = vl.VerbatimLedger(True)
    ledger.release("session", [ANCHOR])
    return ledger


def capture(ledger, store, raw, messages=None):
    ticket = ledger.begin("session", transcript(raw) if messages is None else messages)
    return ledger.capture("session", ticket, store, raw, content="[USER] " + raw, source="conversation")


def ids(ledger, store):
    return vl.resolve_exclusions(store.conn, ledger.snapshot_for("session"))


@pytest.mark.parametrize("value", [None, "", "0", "false", "off", "garbage"])
def test_default_off_and_explicit_false(monkeypatch, value):
    monkeypatch.delenv("MNEMOSYNE_SELF_ECHO_ENABLED", raising=False)
    if value is not None:
        monkeypatch.setenv("MNEMOSYNE_SELF_ECHO_ENABLED", value)
    assert not vl.VerbatimLedger().enabled


def test_optin_without_observed_hook_and_reconstruction_fail_open(store, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_SELF_ECHO_ENABLED", "1")
    ledger = vl.VerbatimLedger()
    capture(ledger, store, "First unsuppressed source")
    assert ids(ledger, store) == set()
    ledger.release("session", [ANCHOR])
    mid = capture(ledger, store, "New source after callback")
    assert ids(ledger, store) == {mid}
    assert vl.VerbatimLedger().snapshot_for("session") is None


@pytest.mark.parametrize("payload", [[], [ANCHOR], [{"role": "system", "content": "Only internal text"}]])
def test_every_hook_releases_all_and_revokes_cached_snapshots(store, payload):
    ledger = armed()
    mid = capture(ledger, store, "Retained or removed source is released equally")
    snapshot = ledger.snapshot_for("session")
    assert vl.resolve_exclusions(store.conn, snapshot) == {mid}
    ledger.release("session", payload)
    assert ids(ledger, store) == set()
    assert vl.resolve_exclusions(store.conn, snapshot) == set()
    ledger.release("session", payload)
    assert ids(ledger, store) == set()


def test_late_inflight_ticket_cannot_record_after_release(store):
    ledger = armed()
    raw = "Delayed completed write"
    ticket = ledger.begin("session", transcript(raw))
    ledger.release("session", transcript(raw))
    mid = ledger.capture("session", ticket, store, raw, content=raw)
    assert store.conn.execute("SELECT id FROM working_memory WHERE id=?", (mid,)).fetchone()
    assert ids(ledger, store) == set()


def test_queued_before_boundary_enters_after_it_and_same_text_repeat_abstain(store):
    ledger = armed()
    raw = "Queued original text never entered the plugin"
    old_messages = transcript(raw)
    ledger.release("session", old_messages)
    capture(ledger, store, raw, old_messages)
    assert ids(ledger, store) == set()
    new_messages = old_messages + [{"role": "assistant", "content": "A genuinely new response"}]
    new = "A genuinely new response"
    mid = capture(ledger, store, new, new_messages)
    assert ids(ledger, store) == {mid}
    # Even a same-text reassertion after the new anchor cannot revive old text.
    ledger.release("session", new_messages)
    capture(ledger, store, raw, new_messages + [{"role": "user", "content": raw}])
    assert ids(ledger, store) == set()


def test_rewritten_old_source_cannot_masquerade_as_new(store):
    ledger = armed()
    raw = "Original user source before skill expansion"
    queued = transcript(raw)
    ledger.release("session", transcript(raw + "\nInjected skill instructions"))
    capture(ledger, store, raw, queued)
    assert ids(ledger, store) == set()


def test_released_text_reintroduced_uniquely_after_old_source_evicted_abstains(store):
    ledger = armed()
    old = "Source released at a previous boundary"
    capture(ledger, store, old)
    ledger.release("session", transcript(old))
    # A subsequent boundary no longer contains the old text; only the
    # cumulative proof set remembers why its reassertion must fail open.
    ledger.release("session", [ANCHOR])
    # A new row after consolidation/deletion must not bypass source evidence.
    store.conn.execute("DELETE FROM working_memory")
    capture(ledger, store, old)
    assert ids(ledger, store) == set()


def test_missing_or_ambiguous_sync_projection_abstains(store):
    ledger = armed()
    assert ledger.begin("session", None) is None
    assert ledger.begin("session", [ANCHOR, ANCHOR]) is None
    capture(ledger, store, "raw input", [ANCHOR, {"role": "user", "content": "rewritten input"}])
    assert ids(ledger, store) == set()


def test_multimodal_text_projection_and_release(store):
    ledger = armed()
    raw = "Botanical catalogue\nImage annotations"
    multipart = {"role": "user", "content": [
        {"type": "text", "text": "Botanical catalogue"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        {"type": "text", "text": "Image annotations"},
    ]}
    mid = capture(ledger, store, raw, [ANCHOR, multipart])
    assert ids(ledger, store) == {mid}
    ledger.release("session", [ANCHOR, multipart])
    capture(ledger, store, raw, [ANCHOR, multipart])
    assert ids(ledger, store) == set()


@pytest.mark.parametrize("payload", [None, ["bad row"], [{"role": "user", "content": None}],
                                      [{"role": "user", "content": [{"type": "unknown"}]}]])
def test_unknown_hook_projection_releases_and_disables_until_reset(store, payload):
    ledger = armed()
    capture(ledger, store, "First source")
    ledger.release("session", payload)
    ledger.release("session", [ANCHOR])
    capture(ledger, store, "Later source")
    assert ids(ledger, store) == set()


def test_long_context_does_not_age_out_and_all_releases(store):
    ledger = armed()
    mids = {capture(ledger, store, f"Distinct long-context source number {i}") for i in range(80)}
    assert ids(ledger, store) == mids
    ledger.release("session", [ANCHOR])
    assert ids(ledger, store) == set()


@pytest.mark.parametrize("cap", ["MAX_CAPTURES", "MAX_SOURCE_HASHES", "MAX_PAYLOAD_CHARS", "MAX_MESSAGES"])
def test_cap_overflow_revokes_all_not_a_sorted_prefix(store, monkeypatch, cap):
    ledger = armed()
    capture(ledger, store, "Before overflow")
    cached = ledger.snapshot_for("session")
    assert ids(ledger, store)
    monkeypatch.setattr(vl, cap, 1)
    if cap == "MAX_CAPTURES":
        capture(ledger, store, "Overflow capture")
    else:
        ledger.release("session", transcript("overflow"))
    assert ids(ledger, store) == set()
    assert vl.resolve_exclusions(store.conn, cached) == set()
    ledger.release("session", [ANCHOR])
    assert ledger.begin("session", transcript("Later source")) is None


def test_session_cap_disables_instance_without_eviction(store, monkeypatch):
    ledger = armed()
    capture(ledger, store, "Owned source")
    cached = ledger.snapshot_for("session")
    monkeypatch.setattr(vl, "MAX_SESSIONS", 1)
    ledger.release("other-session", [ANCHOR])
    assert not ledger.enabled
    assert vl.resolve_exclusions(store.conn, cached) == set()


def test_reset_revokes_tickets_and_capability(store):
    ledger = armed()
    ticket = ledger.begin("session", transcript("Delayed source"))
    capture(ledger, store, "Earlier source")
    cached = ledger.snapshot_for("session")
    ledger.reset_session("session")
    assert ledger.begin("session", transcript("Unarmed source")) is None
    ledger.release("session", [ANCHOR])
    ledger.capture("session", ticket, store, "Delayed source", content="Delayed source")
    assert vl.resolve_exclusions(store.conn, cached) == set()
    assert ids(ledger, store) == set()


@pytest.mark.parametrize("column,value", [
    ("session_id", None), ("session_id", "other"), ("source", "import"),
    ("metadata_json", "{}"), ("metadata_json", "[]"), ("metadata_json", "broken"),
    ("content", "Edited captured text"), ("content", None),
])
def test_provenance_readback_rejects_mutations(store, column, value):
    ledger = armed()
    mid = capture(ledger, store, "Original captured text")
    assert ids(ledger, store) == {mid}
    store.conn.execute(f"UPDATE working_memory SET {column}=? WHERE id=?", (value, mid))
    assert ids(ledger, store) == set()


def test_duplicate_unmarked_row_does_not_acquire_nonce(store):
    raw = "Existing imported equal content"
    mid = store.remember("[USER] " + raw)
    ledger = armed()
    assert capture(ledger, store, raw) == mid
    assert ids(ledger, store) == set()
    assert store.conn.execute("SELECT metadata_json FROM working_memory").fetchone()[0] == "{}"


def test_failed_or_unreadable_write_never_gets_proof(store):
    ledger = armed()
    actual = store.remember
    store.remember = lambda **kw: None
    capture(ledger, store, "Skipped write source")
    assert ids(ledger, store) == set()
    store.remember = lambda **kw: "nonexistent"
    capture(ledger, store, "Nonexistent row source")
    assert ids(ledger, store) == set()
    store.remember = actual
    mid = capture(ledger, store, "Successful source")
    assert ids(ledger, store) == {mid}
