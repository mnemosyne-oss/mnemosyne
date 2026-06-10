"""Tests for always-inject, query-independent identity prefetch.

Per-contact identity memories (source='identity') answer "who am I talking
to?" and must surface on EVERY turn, regardless of the user's message. Routing
them through semantic recall is a latent bug: a short/generic opener ("Hi", a
nickname) does not match the identity text, so it never enters recall's top_k
window and the importance filter never sees it -- the agent then loses track of
who it is talking to. These tests prove identity injection is:

  (a) deterministic on a generic query that does NOT semantically match it,
  (b) strictly session-scoped (session A's identity never leaks into session B),
  (c) deduplicated (a query that DOES match the identity yields no duplicate).
"""
from __future__ import annotations

import os
import tempfile

import pytest

from hermes_memory_provider import MnemosyneMemoryProvider


def _insert_identity(beam, content, session_id, importance=0.95):
    """Insert an identity row directly into working_memory for *session_id*."""
    beam.conn.execute(
        "INSERT INTO working_memory "
        "(id, content, source, timestamp, session_id, importance) "
        "VALUES (?, ?, 'identity', ?, ?, ?)",
        (
            f"id-{session_id}-{abs(hash(content)) % 10**9}",
            content,
            "2026-05-14T12:00:00Z",
            session_id,
            importance,
        ),
    )
    beam.conn.commit()


@pytest.fixture
def provider_factory():
    """Yield a factory that builds a provider bound to a beam for a session_id,
    sharing one temp DB so cross-session isolation can be exercised."""
    from mnemosyne.core.beam import BeamMemory

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "mnemosyne.db")
        beams = []

        def make(session_id):
            beam = BeamMemory(session_id=session_id, db_path=db_path)
            beams.append(beam)
            p = MnemosyneMemoryProvider()
            p._beam = beam
            p._prefetch_profile = "general"
            p._agent_context = "test"
            p._skip_contexts = set()
            return p

        yield make

        for b in beams:
            try:
                b.conn.close()
            except Exception:
                pass


def test_identity_surfaces_on_non_matching_generic_query(provider_factory):
    """An identity row is injected even when the query does NOT match it."""
    p = provider_factory("session-A")
    _insert_identity(
        p._beam,
        "Contact A is the lead engineer on the payments team.",
        "session-A",
    )
    # A short, generic opener with zero semantic overlap with the identity.
    block = p.prefetch("Hi")
    assert "[IDENTITY]" in block
    assert "Contact A is the lead engineer on the payments team." in block


def test_identity_does_not_leak_across_sessions(provider_factory):
    """Identity stored under session A must never appear in session B prefetch."""
    pa = provider_factory("session-A")
    _insert_identity(
        pa._beam,
        "Contact A is the lead engineer on the payments team.",
        "session-A",
    )

    pb = provider_factory("session-B")
    block_b = pb.prefetch("Hi")
    assert "Contact A" not in block_b
    assert "[IDENTITY]" not in block_b

    # Sanity: session A still sees its own identity.
    block_a = pa.prefetch("Hi")
    assert "Contact A is the lead engineer on the payments team." in block_a


def test_no_duplicate_when_query_matches_identity(provider_factory):
    """When recall ALSO surfaces the identity, it must not be injected twice."""
    p = provider_factory("session-A")
    identity_text = "Contact A is the lead engineer on the payments team."
    _insert_identity(p._beam, identity_text, "session-A")

    # Force recall to return the very same identity content (simulating a query
    # that DOES semantically match), so the dedup path is exercised.
    def fake_recall(**kwargs):
        return [{
            "content": identity_text,
            "timestamp": "2026-05-14T12:00:00Z",
            "importance": 0.95,
            "score": 0.9,
            "trust_tier": "STATED",
        }]

    p._beam.recall = fake_recall
    block = p.prefetch("who is the lead engineer on payments")
    # The identity content appears exactly once across the whole block.
    assert block.count(identity_text) == 1


def test_no_identity_rows_is_a_noop(provider_factory):
    """With no identity rows, behavior is unchanged (no [IDENTITY] block)."""
    p = provider_factory("session-A")
    p._beam.recall = lambda **kwargs: []
    block = p.prefetch("Hi")
    assert "[IDENTITY]" not in block
