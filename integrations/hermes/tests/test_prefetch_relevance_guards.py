from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mnemosyne_hermes import (
    MnemosyneMemoryProvider,
    _canonical_prefetch_rows,
    _is_operational_log,
    _is_prompt_echo,
    _prefetch_age_label,
    _prefetch_tokens,
)


@pytest.fixture(autouse=True)
def clear_prefetch_configuration(monkeypatch):
    for key in (
        "MNEMOSYNE_PREFETCH_CANONICAL_GENERIC_TOKENS",
        "MNEMOSYNE_PREFETCH_CANONICAL_EXTRA_GENERIC_TOKENS",
        "MNEMOSYNE_PREFETCH_MIN_DISTINCTIVE_TOKENS",
        "MNEMOSYNE_PREFETCH_MIN_QUERY_COVERAGE",
        "MNEMOSYNE_PREFETCH_CANONICAL_RARE_TOKEN_MAX_FREQUENCY",
    ):
        monkeypatch.delenv(key, raising=False)


def test_stopwords_drop_function_words_keep_content_words():
    assert "was" not in _prefetch_tokens("how was the race?")
    assert "any" not in _prefetch_tokens("any news on the job?")
    tokens = _prefetch_tokens("open the shutters")
    assert "open" in tokens
    assert "shutters" in tokens


def test_is_operational_log_detects_cron_rows():
    assert _is_operational_log(
        "IBKR challenge run #324 (2026-09-21 14:30Z, intraday scan of candidates)"
    )
    assert _is_operational_log(
        "Xi/Memory-deal monitor tick 2026-09-13, nothing to act on"
    )
    assert _is_operational_log("HEALTH FEED 2026-10-15: all green")
    assert _is_operational_log("NIGHTLY INTROSPECTION 2026-09-10 — reflections")
    assert _is_operational_log(
        "[USER] [Home Assistant] [Home Assistant] Tapparella (cover.tapparella_bedroom) opened"
    )
    assert not _is_operational_log("Alex prefers metric units for shutters.")
    assert not _is_operational_log("")


class FakeCanonicalStore:
    def __init__(self, rows):
        self.rows = rows

    def list(self, owner_id):
        return list(self.rows)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _now():
    return datetime.now(timezone.utc)


def test_canonical_prefetch_drops_weak_and_stale_task_progress_rows():
    now = _now()
    shared_token = "kesselrrundweg"
    rows = [
        {
            "category": "task:progress",
            "name": "single-weak-overlap",
            "body": f"status update on the {shared_token} errand.",
            "owner_id": "default",
            "valid_from": _iso(now - timedelta(days=1)),
            "valid_until": "",
            "created_at": "",
        },
        {
            "category": "task:progress",
            "name": "single-weak-overlap-two",
            "body": f"another log mentioning {shared_token} briefly.",
            "owner_id": "default",
            "valid_from": _iso(now - timedelta(days=1)),
            "valid_until": "",
            "created_at": "",
        },
        {
            "category": "task:progress",
            "name": "old-row",
            "body": f"{shared_token} errand status updated earlier in the season.",
            "owner_id": "default",
            "valid_from": _iso(now - timedelta(days=10)),
            "valid_until": "",
            "created_at": "",
        },
        {
            "category": "task:progress",
            "name": "fresh-strong-row",
            "body": f"{shared_token} errand resumed at westgate.",
            "owner_id": "default",
            "valid_from": _iso(now - timedelta(days=1)),
            "valid_until": "",
            "created_at": "",
        },
    ]
    # The shared token appears in all four bodies, so its document frequency
    # is 4 and even the multi-token branch needs another genuinely rare token.
    # Historical (task:progress) rows can never qualify on a single token, so
    # the two one-token rows always drop; the stale row drops on age; only the
    # fresh row sharing two rare tokens survives.
    rendered = _canonical_prefetch_rows(
        FakeCanonicalStore(rows), "default",
        f"{shared_token} resumed westgate",
    )
    names = {r.get("canonical_name") for r in rendered}
    assert "single-weak-overlap" not in names
    assert "single-weak-overlap-two" not in names
    assert "old-row" not in names
    assert any(r.get("canonical_name") == "fresh-strong-row" for r in rendered)
    kept = next(
        r for r in rendered if r.get("canonical_name") == "fresh-strong-row"
    )
    assert kept["importance"] == 0.55
    assert kept["trust_tier"] == "RECORD"


def test_canonical_prefetch_keeps_identity_single_rare_token():
    rows = [
        {
            "category": "identity",
            "name": "surname",
            "body": "Zoraya is the family surname on record.",
            "owner_id": "default",
            "valid_from": "",
            "valid_until": "",
            "created_at": "",
        },
    ]
    rendered = _canonical_prefetch_rows(
        FakeCanonicalStore(rows), "default", "zoraya"
    )
    assert len(rendered) == 1
    assert rendered[0]["trust_tier"] == "CANONICAL"
    assert rendered[0]["importance"] == 0.95


def test_canonical_single_token_needs_half_the_query():
    """A generic noun that is rare only inside the canonical surface must not
    drag an unrelated identity slot in (live regression: 'gbrain port number'
    pulled the salary slot via the shared word 'number')."""
    rows = [
        {
            "category": "identity",
            "name": "salary",
            "body": "Compensation is 125000 euro, fourteen monthly payments.",
            "owner_id": "default",
            "valid_from": "",
            "valid_until": "",
            "created_at": "",
        },
    ]
    assert _canonical_prefetch_rows(
        FakeCanonicalStore(rows), "default", "gbrain port number"
    ) == []
    # One shared token covering half of a two-token query is still decisive.
    kept = _canonical_prefetch_rows(
        FakeCanonicalStore(rows), "default", "compensation details"
    )
    assert len(kept) == 1


def test_is_prompt_echo_blocks_consolidation_boilerplate():
    assert _is_prompt_echo("The following memories are concise and brief.")
    assert _is_prompt_echo("1. Memories: Alex prefers metric units.")
    assert _is_prompt_echo("Memories: Alex bought a GPU.")
    assert _is_prompt_echo("Summarizes the conversation into three sentences: x")
    assert _is_prompt_echo("Summary: Summary: - No action needed.")
    assert _is_prompt_echo("1. Preserve facts, names, preference, and decision by summarizing the conversation.")
    assert _is_prompt_echo("Memorize the conversation and update the skill library")
    assert _is_prompt_echo("A concise summary of the conversation is: A brief update.")
    assert not _is_prompt_echo("Remember to prune old backups")
    assert not _is_prompt_echo("reminder: buy milk")
    assert not _is_prompt_echo("Alex prefers metric units for shutters.")
    assert not _is_prompt_echo("")


def test_prefetch_age_label_units():
    now = datetime.now()
    assert _prefetch_age_label(now.isoformat()) == "today"
    old = (now - timedelta(days=47)).isoformat()
    assert _prefetch_age_label(old) == "47d old"
    assert _prefetch_age_label("") == ""
    assert _prefetch_age_label("not a timestamp") == ""


def test_prefetch_blocks_operational_and_foreign_global_raw_rows(monkeypatch):
    class FakeBeam:
        def __init__(self, results):
            self._results = results

        def recall(self, *args, **kwargs):
            return self._results

    p = MnemosyneMemoryProvider()
    p._beam = FakeBeam([
        {
            "content": "IBKR challenge run #324 (2026-09-21 14:30Z, intraday)",
            "scope": "global", "source": "conversation",
            "timestamp": "2026-09-21T14:30:00+00:00",
            "importance": 1.0, "score": 1.0, "keyword_score": 1.0,
            "trust_tier": "STATED",
        },
        {
            "content": "[USER] the shutters in the bedroom still need greasing",
            "scope": "global", "source": "conversation",
            "timestamp": "2026-08-01T09:00:00+00:00",
            "importance": 1.0, "score": 1.0, "keyword_score": 1.0,
            "trust_tier": "STATED",
        },
        {
            "content": "Owner prefers metric units for shutters and curtains.",
            "scope": "session", "source": "preference",
            "timestamp": "2026-10-20T09:00:00+00:00",
            "importance": 0.9, "score": 0.8, "keyword_score": 0.8,
            "trust_tier": "STATED",
        },
    ])
    p._agent_context = "primary"

    block = p.prefetch("shutters metric preferences")
    assert block
    assert "metric units" in block
    assert "IBKR challenge" not in block
    assert "greasing" not in block
