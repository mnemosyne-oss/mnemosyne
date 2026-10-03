"""Both real provider copies: ownership, ordering, missing-host capability."""
import importlib.util
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.verbatim_ledger import resolve_exclusions
from tests.test_verbatim_ledger import ANCHOR, transcript

ROOT = Path(__file__).resolve().parents[1]
PATHS = {
    "root": ROOT / "hermes_memory_provider" / "__init__.py",
    "packaged": ROOT / "integrations/hermes/src/mnemosyne_hermes/__init__.py",
}


def load_provider(name):
    spec = importlib.util.spec_from_file_location(f"boundary_provider_{name}", PATHS[name])
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == PATHS[name]
    return module


@pytest.fixture(params=list(PATHS))
def provider_mod(request):
    return load_provider(request.param)


@pytest.fixture
def provider(provider_mod, tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_SELF_ECHO_ENABLED", "1")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MNEMOSYNE_VOICE_GRAPH", "0")
    monkeypatch.setenv("MNEMOSYNE_VOICE_FACT", "0")
    obj = provider_mod.MnemosyneMemoryProvider()
    obj._beam = BeamMemory(db_path=tmp_path / "memory.db", session_id="custom-beam-session")
    obj._active_session_id = "session"
    obj._agent_context = "primary"
    obj._auto_sleep_enabled = False
    obj._sync_roles = {"user", "assistant"}
    obj._capture_identity_signals = lambda _: None
    return obj


def synced(provider, raw, messages=None, **kw):
    provider.sync_turn(raw, "", session_id="session", messages=transcript(raw) if messages is None else messages, **kw)


def exclusions(provider):
    return resolve_exclusions(provider._beam.conn, provider._verbatim_ledger.snapshot_for("session"))


def arm(provider):
    provider.on_pre_compress([ANCHOR])


def test_provider_default_off_and_no_v2_advertisement(provider_mod, monkeypatch):
    monkeypatch.delenv("MNEMOSYNE_SELF_ECHO_ENABLED", raising=False)
    provider = provider_mod.MnemosyneMemoryProvider()
    assert not provider._verbatim_ledger.enabled
    # The callback is bookkeeping only. No checkpoint implementation/marker.
    assert "api_version" not in provider.__class__.__dict__
    assert "API_VERSION" not in provider.__class__.__dict__


def test_optin_method_existence_is_not_observed_capability(provider):
    assert callable(provider.on_pre_compress)
    synced(provider, "orchard botanical archive before any hook")
    assert provider._beam.conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == 1
    assert not exclusions(provider)
    arm(provider)
    synced(provider, "orchard botanical archive newly captured after hook")
    assert exclusions(provider)


@pytest.mark.parametrize("engine", ["0", "1"], ids=["linear", "polyphonic"])
def test_both_engines_through_actual_provider_prefetch_and_release(provider, monkeypatch, engine):
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", engine)
    arm(provider)
    raw = "orchard botanical archive itinerary includes a greenhouse expedition"
    synced(provider, raw)
    mids = exclusions(provider)
    assert len(mids) == 1
    query = "recent orchard botanical archive itinerary greenhouse expedition"
    assert provider._beam.recall(query, _cross_session=True)
    real_recall = provider._beam.recall
    returned = []

    def observed_recall(**kw):
        results = real_recall(**kw)
        returned.append(results)
        return results

    monkeypatch.setattr(provider._beam, "recall", observed_recall)
    before = provider.prefetch(query, session_id="session")
    assert returned[-1] == []
    assert "greenhouse expedition" not in before
    provider.on_pre_compress(transcript(raw))
    assert not exclusions(provider)
    after = provider.prefetch(query, session_id="session")
    assert {r["id"] for r in returned[-1]} == mids
    if engine == "0":
        assert "greenhouse expedition" in after
    # Polyphonic temporal-only hits lack the lexical score fields required
    # by the existing provider relevance gate. They become eligible in real
    # recall after release; this optimization does not change that gate.


def test_user_and_assistant_successful_captures_share_boundary_contract(provider):
    from mnemosyne.core.verbatim_ledger import content_hash
    arm(provider)
    user = "User source about botanical archives"
    assistant = "Assistant response about botanical archives"
    messages = transcript(user) + [{"role": "assistant", "content": assistant}]
    provider.sync_turn(user, assistant, session_id="session", messages=messages)
    snapshot = provider._verbatim_ledger.snapshot_for("session")
    assert snapshot and len(snapshot.captures) == 2
    assert {p.stored_hash for p in snapshot.captures} == {
        content_hash("[USER] " + user), content_hash("[ASSISTANT] " + assistant)
    }
    assert len(exclusions(provider)) == 2
    provider.on_pre_compress(messages)
    assert not exclusions(provider)


def test_actual_beam_session_key_not_inferred_from_caller(provider):
    arm(provider)
    synced(provider, "botanical archives with mismatched beam key")
    snapshot = provider._verbatim_ledger.snapshot_for("session")
    assert snapshot and len(snapshot.captures) == 1
    proof = snapshot.captures[0]
    stored = provider._beam.conn.execute("SELECT session_id FROM working_memory WHERE id=?", (proof.memory_id,)).fetchone()[0]
    assert proof.session_id == stored
    assert proof.session_id != "session"


@pytest.mark.parametrize("payload", [[], [ANCHOR], [{"role": "system", "content": "internal"}]])
def test_noop_retained_failure_signals_release_everything(provider, payload):
    arm(provider)
    synced(provider, "Captured turn omitted from hook payload")
    cached = provider._verbatim_ledger.snapshot_for("session")
    assert exclusions(provider)
    provider.on_pre_compress(payload, require_checkpoint=False)
    assert not exclusions(provider)
    assert not resolve_exclusions(provider._beam.conn, cached)
    provider.on_pre_compress(payload)
    assert not exclusions(provider)


def test_multimodal_queued_sync_after_boundary_never_resurrects(provider):
    arm(provider)
    raw = "botanical archive records\norchard greenhouse map"
    multipart = {"role": "user", "content": [
        {"type": "text", "text": "botanical archive records"},
        {"type": "image_url", "image_url": {"url": "https://example.invalid/image.png"}},
        {"type": "text", "text": "orchard greenhouse map"},
    ]}
    queued_payload = [ANCHOR, multipart]
    provider.on_pre_compress(queued_payload)
    synced(provider, raw, queued_payload)
    assert not exclusions(provider)
    later = "genuinely new botanical catalogue entry"
    synced(provider, later, queued_payload + [{"role": "user", "content": later}])
    assert len(exclusions(provider)) == 1
    provider.on_pre_compress(queued_payload)
    assert not exclusions(provider)


@pytest.mark.parametrize("payload_covers_source", [False, True], ids=["empty-hook", "full-hook"])
def test_inflight_committed_write_after_boundary_cannot_rearm(provider, monkeypatch, payload_covers_source):
    arm(provider)
    committed, resume = threading.Event(), threading.Event()
    remember = provider._beam.remember

    def delayed(**kw):
        mid = remember(**kw)
        committed.set()
        assert resume.wait(10), "test rendezvous timed out"
        return mid

    monkeypatch.setattr(provider._beam, "remember", delayed)
    raw = "orchard botanical archive async committed source"
    with ThreadPoolExecutor(max_workers=1) as executor:
        task = executor.submit(synced, provider, raw)
        try:
            assert committed.wait(10)
            provider.on_pre_compress(transcript(raw) if payload_covers_source else [])
        finally:
            resume.set()
        task.result(timeout=10)
    assert provider._beam.conn.execute("SELECT content FROM working_memory").fetchone()[0] == "[USER] " + raw
    assert not exclusions(provider)


def test_queued_enters_after_hook_not_just_inflight(provider):
    arm(provider)
    raw = "queued orchard botanical archive text"
    queued_payload = transcript(raw)
    provider.on_pre_compress(queued_payload)
    # No plugin call/ticket existed when the hook fired.
    synced(provider, raw, queued_payload)
    assert not exclusions(provider)
    for i in range(30):
        later = f"Subsequent distinct orchard source {i}"
        synced(provider, later, queued_payload + [{"role": "user", "content": later}])
    ids = exclusions(provider)
    row = provider._beam.conn.execute("SELECT id FROM working_memory WHERE content=?", ("[USER] " + raw,)).fetchone()
    assert row and row[0] not in ids
    assert len(ids) == 30


def test_rewritten_and_missing_transcript_abstain(provider):
    arm(provider)
    raw = "Original orchard request"
    provider.on_pre_compress(transcript(raw + "\nexpanded skill scaffold"))
    synced(provider, raw, transcript(raw))
    assert not exclusions(provider)
    provider.sync_turn("new turn without transcript", "", session_id="session")
    assert not exclusions(provider)
    assert provider._beam.conn.execute(
        "SELECT 1 FROM working_memory WHERE content = ?",
        ("[USER] new turn without transcript",),
    ).fetchone() is not None


def test_repeat_after_release_and_duplicate_collision_are_unowned(provider):
    arm(provider)
    raw = "orchard botanical archive duplicate content"
    synced(provider, raw)
    assert exclusions(provider)
    provider.on_pre_compress(transcript(raw))
    arm(provider)
    synced(provider, raw)
    assert not exclusions(provider)
    fresh_provider_ledger = type(provider._verbatim_ledger)(True)
    provider._verbatim_ledger = fresh_provider_ledger
    arm(provider)
    # Existing row has the old instance's nonce; dedupe must not adopt it.
    synced(provider, raw)
    assert not exclusions(provider)


def test_reset_reconstruction_unknown_session_and_empty_reset(provider):
    arm(provider)
    synced(provider, "orchard botanical archive reset source")
    assert exclusions(provider)
    assert provider._verbatim_ledger.snapshot_for("unknown-session") is None
    provider.on_session_switch("", reset=True)
    assert not exclusions(provider)
    provider._active_session_id = "session"
    synced(provider, "After reset without new observed hook")
    assert not exclusions(provider)


def test_prefetch_retries_ordinary_recall_when_snapshot_revoked_inflight(provider, monkeypatch):
    arm(provider)
    synced(provider, "botanical archive prefetch source")
    real_recall = provider._beam.recall
    calls = []

    def recalling(**kw):
        calls.append(kw)
        result = real_recall(**kw)
        if kw.get("exclude_captures"):
            provider.on_pre_compress([ANCHOR])
        return result

    monkeypatch.setattr(provider._beam, "recall", recalling)
    provider.prefetch("recent botanical archive source", session_id="session")
    assert len(calls) == 2
    assert calls[0].get("exclude_captures") is not None
    assert "exclude_captures" not in calls[1]


def test_explicit_provider_tool_recall_never_receives_exclusions(provider, monkeypatch):
    arm(provider)
    synced(provider, "orchard botanical archive explicit recall source")
    assert exclusions(provider)
    actual = provider._beam.recall
    calls = []

    def observed(query, **kwargs):
        calls.append(kwargs)
        return actual(query, _cross_session=True, **kwargs)

    monkeypatch.setattr(provider._beam, "recall", observed)
    response = json.loads(provider._handle_recall({"query": "orchard botanical archive"}))
    assert response["results"]
    assert calls and all("exclude_captures" not in call for call in calls)


def test_readback_uses_actual_sanitized_content_and_rejects_later_edits(provider, monkeypatch):
    from mnemosyne.core import content_sanitizer
    from mnemosyne.core.verbatim_ledger import content_hash
    arm(provider)
    raw = "Original botanical archive source before sanitization"
    persisted = "Sanitized botanical archive source"
    monkeypatch.setattr(content_sanitizer, "sanitize_content", lambda _: (persisted, {"test": True}))
    synced(provider, raw)
    snapshot = provider._verbatim_ledger.snapshot_for("session")
    assert snapshot and len(snapshot.captures) == 1
    proof = snapshot.captures[0]
    assert proof.stored_hash == content_hash(persisted)
    assert exclusions(provider) == {proof.memory_id}
    provider._beam.conn.execute("UPDATE working_memory SET content=? WHERE id=?", ("Edited source", proof.memory_id))
    provider._beam.conn.commit()
    assert not exclusions(provider)


def test_interleaved_old_session_sync_cannot_change_hook_target(provider):
    arm(provider)
    synced(provider, "Current session botanical archive source")
    assert exclusions(provider)
    # A queued call from another session must not redirect the next v1 hook.
    provider.sync_turn("Other session delayed source", "", session_id="other-session", messages=transcript("Other session delayed source"))
    assert provider._active_session_id == "session"
    assert provider._verbatim_ledger.snapshot_for("other-session") is None
    provider.on_pre_compress([ANCHOR])
    assert not exclusions(provider)


def test_reset_tombstone_prevents_queued_calls_after_reobserved_hook(provider):
    arm(provider)
    raw = "Released botanical archive text before reset"
    synced(provider, raw)
    provider._verbatim_ledger.reset_session("session")
    arm(provider)
    # Lost safety evidence cannot be reconstructed by reusing an old anchor.
    synced(provider, raw)
    assert not exclusions(provider)
    assert provider._verbatim_ledger.begin("session", transcript("New text")) is None


def test_failed_store_or_unreadable_beam_never_suppresses(provider):
    arm(provider)
    provider._beam = MagicMock()
    provider._beam.remember.return_value = None
    synced(provider, "unsuccessfully stored orchard source")
    assert provider._verbatim_ledger.snapshot_for("session") is None
