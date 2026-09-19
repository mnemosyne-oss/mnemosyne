"""
Tests for persistent recall provenance logging (mnemosyne/core/recall_provenance.py).

When MNEMOSYNE_RECALL_PROVENANCE=1, BeamMemory.recall() appends one JSONL
line per call to <db>.recall_provenance.jsonl (one file per database),
recording query + returned ids/scores. The flag is read per call; default
OFF means no file is ever created.
"""

import json
import tempfile
import threading
from collections import Counter
from pathlib import Path

import pytest

from mnemosyne.core import recall_provenance as rp
from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.recall_provenance import (
    append_recall_provenance,
    cleanup_orphaned_provenance,
    read_recall_provenance,
)


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        yield db_path


def _provenance_file(db_path: Path) -> Path:
    return Path(str(db_path) + ".recall_provenance.jsonl")


def _read_last_record(db_path: Path) -> dict:
    lines = _provenance_file(db_path).read_text(encoding="utf-8").splitlines()
    return json.loads(lines[-1])


def test_flag_on_writes_provenance_line(temp_db, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_RECALL_PROVENANCE", "1")
    beam = BeamMemory(session_id="prov-a", db_path=temp_db)
    id1 = beam.remember("pluto alpha provenance fact one", source="test")
    id2 = beam.remember("pluto beta provenance fact two", source="test")

    # Long query (>200 chars) with the shared token early, so the
    # truncated stored query still matches both memories via FTS.
    query = "pluto " + "padding " * 30
    assert len(query) > 200
    results = beam.recall(query, top_k=5)

    prov_file = _provenance_file(temp_db)
    assert prov_file.exists()
    record = _read_last_record(temp_db)
    assert record["query"] == query[:200]
    assert record["top_k"] == 5
    assert isinstance(record["ts"], str)
    returned_ids = {r["id"] for r in record["results"]}
    assert {id1, id2} <= returned_ids
    for entry in record["results"]:
        assert "id" in entry
        assert "tier" in entry
        assert "score" in entry
        assert "importance" in entry
        assert entry["importance"] is not None
        assert "timestamp" in entry
        assert entry["timestamp"] is not None
    # Ids recorded are exactly what recall returned, in order.
    assert [r["id"] for r in record["results"]] == [r["id"] for r in results]


def test_flag_off_by_default_writes_no_file(temp_db, monkeypatch):
    monkeypatch.delenv("MNEMOSYNE_RECALL_PROVENANCE", raising=False)
    beam = BeamMemory(session_id="prov-b", db_path=temp_db)
    beam.remember("pluto gamma provenance fact three", source="test")
    beam.recall("pluto", top_k=5)

    assert not _provenance_file(temp_db).exists()


def test_flag_garbage_value_treated_as_off(temp_db, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_RECALL_PROVENANCE", "yes")
    beam = BeamMemory(session_id="prov-c", db_path=temp_db)
    beam.remember("pluto delta provenance fact four", source="test")
    beam.recall("pluto", top_k=5)

    assert not _provenance_file(temp_db).exists()


def test_read_recall_provenance_newest_first_and_limit(temp_db, monkeypatch):
    monkeypatch.delenv("MNEMOSYNE_RECALL_PROVENANCE", raising=False)

    assert read_recall_provenance(temp_db) == []

    for i in range(4):
        append_recall_provenance(
            str(temp_db), f"query-{i}", [{"id": f"m{i}", "tier": "working",
                                          "score": 1.0}], top_k=5,
        )

    records = read_recall_provenance(temp_db, limit=20)
    assert [r["query"] for r in records] == [
        "query-3", "query-2", "query-1", "query-0",
    ]

    limited = read_recall_provenance(temp_db, limit=2)
    assert len(limited) == 2
    assert [r["query"] for r in limited] == ["query-3", "query-2"]


def test_read_recall_provenance_zero_or_negative_limit_returns_empty(
        temp_db, monkeypatch):
    """limit<=0 must return [], not the whole file (-0 == 0 slicing bug)."""
    monkeypatch.delenv("MNEMOSYNE_RECALL_PROVENANCE", raising=False)
    for i in range(3):
        append_recall_provenance(
            str(temp_db), f"query-{i}", [{"id": f"m{i}", "tier": "working",
                                          "score": 1.0}], top_k=5,
        )

    assert read_recall_provenance(temp_db, limit=0) == []
    assert read_recall_provenance(temp_db, limit=-3) == []


def test_read_recall_provenance_skips_malformed_lines(temp_db, monkeypatch):
    """Malformed JSONL lines are skipped; valid neighbors still returned."""
    monkeypatch.delenv("MNEMOSYNE_RECALL_PROVENANCE", raising=False)
    append_recall_provenance(
        str(temp_db), "good-1", [{"id": "m1", "tier": "working",
                                  "score": 1.0}], top_k=5,
    )
    with open(_provenance_file(temp_db), "a", encoding="utf-8") as f:
        f.write("{not valid json\n")
    append_recall_provenance(
        str(temp_db), "good-2", [{"id": "m2", "tier": "working",
                                  "score": 1.0}], top_k=5,
    )

    records = read_recall_provenance(temp_db, limit=10)
    assert [r["query"] for r in records] == ["good-2", "good-1"]


def test_explain_recall_writes_no_provenance_line(temp_db, monkeypatch):
    """explain=True returns via the explain-trace early return, before the
    provenance hook, so no JSONL line is written even with the flag on."""
    monkeypatch.setenv("MNEMOSYNE_RECALL_PROVENANCE", "1")
    beam = BeamMemory(session_id="prov-e", db_path=temp_db)
    beam.remember("pluto zeta provenance fact six", source="test")

    explained = beam.recall("pluto", top_k=5, explain=True)
    assert explained["engine"] == "linear"
    assert not _provenance_file(temp_db).exists()


def test_enhanced_recall_delegation_writes_no_provenance_line(
        temp_db, monkeypatch):
    """Under MNEMOSYNE_ENHANCED_RECALL=1, the recall_enhanced() cache-miss
    delegation into self.recall() must not log provenance: its expanded
    query + doubled top_k are internal, not the caller's request."""
    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")
    monkeypatch.setenv("MNEMOSYNE_RECALL_PROVENANCE", "1")
    beam = BeamMemory(session_id="prov-f", db_path=temp_db)
    beam.remember("pluto eta provenance fact seven", source="test")

    # First call is always a cache miss: the linear path really runs.
    results = beam.recall_enhanced(
        "pluto", top_k=5,
        use_intent=False, use_synonyms=False, use_weibull=False, use_mmr=False,
    )
    assert results

    assert not _provenance_file(temp_db).exists()


def test_enhanced_recall_passthrough_writes_provenance_line(
        temp_db, monkeypatch):
    """With MNEMOSYNE_ENHANCED_RECALL unset, recall_enhanced() is a plain
    passthrough into recall() WITHOUT _skip_provenance -- the caller's
    request flows through the linear path, so provenance IS logged."""
    monkeypatch.delenv("MNEMOSYNE_ENHANCED_RECALL", raising=False)
    monkeypatch.setenv("MNEMOSYNE_RECALL_PROVENANCE", "1")
    beam = BeamMemory(session_id="prov-g", db_path=temp_db)
    beam.remember("pluto eta provenance fact seven", source="test")

    results = beam.recall_enhanced(
        "pluto", top_k=5,
        use_intent=False, use_synonyms=False, use_weibull=False, use_mmr=False,
    )
    assert results
    assert _provenance_file(temp_db).exists()
    records = read_recall_provenance(temp_db, limit=1)
    assert records and records[0]["query"] == "pluto"


def test_flag_read_per_call(temp_db, monkeypatch):
    """The flag is consulted on every recall call, not cached at init."""
    monkeypatch.delenv("MNEMOSYNE_RECALL_PROVENANCE", raising=False)
    beam = BeamMemory(session_id="prov-d", db_path=temp_db)
    beam.remember("pluto epsilon provenance fact five", source="test")

    beam.recall("pluto", top_k=5)
    assert not _provenance_file(temp_db).exists()

    monkeypatch.setenv("MNEMOSYNE_RECALL_PROVENANCE", "1")
    beam.recall("pluto", top_k=5)
    assert _provenance_file(temp_db).exists()

    monkeypatch.setenv("MNEMOSYNE_RECALL_PROVENANCE", "0")
    beam.recall("pluto", top_k=5)
    lines = _provenance_file(temp_db).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_provenance_files_isolated_per_database(monkeypatch):
    """Two databases in the SAME directory get separate audit files:
    each file holds only its own store's queries (regression for the
    shared recall_provenance.jsonl-per-directory collision)."""
    monkeypatch.delenv("MNEMOSYNE_RECALL_PROVENANCE", raising=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        a, b = Path(tmpdir) / "a.db", Path(tmpdir) / "b.db"
        append_recall_provenance(
            str(a), "query-alpha", [{"id": "ma", "tier": "working",
                                     "score": 1.0}], top_k=5,
        )
        append_recall_provenance(
            str(b), "query-beta", [{"id": "mb", "tier": "working",
                                    "score": 1.0}], top_k=5,
        )

        pa, pb = _provenance_file(a), _provenance_file(b)
        assert pa.exists() and pb.exists()
        assert pa != pb
        assert [r["query"] for r in read_recall_provenance(a)] == ["query-alpha"]
        assert [r["query"] for r in read_recall_provenance(b)] == ["query-beta"]


def test_provenance_file_created_with_restrictive_permissions(
        temp_db, monkeypatch):
    """The file is created 0600: no group/other bits, under any umask."""
    monkeypatch.delenv("MNEMOSYNE_RECALL_PROVENANCE", raising=False)
    append_recall_provenance(str(temp_db), "perm-check", [], top_k=5)

    st = _provenance_file(temp_db).stat()
    assert st.st_mode & 0o077 == 0


def test_rotation_keeps_single_generation(temp_db, monkeypatch):
    """At the size cap the current file rotates to `.1` (overwriting any
    previous generation); reads cover the current file only."""
    monkeypatch.delenv("MNEMOSYNE_RECALL_PROVENANCE", raising=False)
    # Rotate on every append so the policy is exercised deterministically.
    monkeypatch.setattr(rp, "_MAX_FILE_BYTES", 1)
    for i in range(5):
        append_recall_provenance(
            str(temp_db), f"rot-{i}", [{"id": f"m{i}", "tier": "working",
                                        "score": 1.0}], top_k=5,
        )

    current = _provenance_file(temp_db)
    rotated = Path(str(current) + ".1")
    assert current.exists() and rotated.exists()
    # Single generation only: no .2, .3, ... ever accumulates.
    assert not Path(str(current) + ".2").exists()
    # The current generation holds only post-rotation records; the
    # rotated `.1` is not read back.
    assert [r["query"] for r in read_recall_provenance(temp_db)] == ["rot-4"]


def test_read_recall_provenance_reads_only_tail_window(temp_db, monkeypatch):
    """Reads examine a bounded tail of the file, newest first, with the
    possibly-partial first window line dropped."""
    monkeypatch.delenv("MNEMOSYNE_RECALL_PROVENANCE", raising=False)
    monkeypatch.setattr(rp, "_MAX_TAIL_BYTES", 500)
    for i in range(50):
        append_recall_provenance(
            str(temp_db), f"tail-{i}", [{"id": f"m{i}", "tier": "working",
                                         "score": 1.0}], top_k=5,
        )

    records = read_recall_provenance(temp_db, limit=100)
    numbers = [int(r["query"].split("-")[1]) for r in records]
    assert numbers, "tail read returned nothing"
    assert numbers[0] == 49  # newest record first
    assert numbers == sorted(numbers, reverse=True)
    assert len(numbers) < 50  # window truncated, not the whole file


def test_cleanup_orphaned_provenance_removes_files_when_db_gone(
        temp_db, monkeypatch):
    monkeypatch.delenv("MNEMOSYNE_RECALL_PROVENANCE", raising=False)
    append_recall_provenance(str(temp_db), "orphan-check", [], top_k=5)
    rotated = Path(str(_provenance_file(temp_db)) + ".1")
    rotated.write_text("{}\n", encoding="utf-8")

    # DB still present: nothing removed.
    temp_db.write_bytes(b"")
    assert cleanup_orphaned_provenance(temp_db) is False
    assert _provenance_file(temp_db).exists()

    temp_db.unlink()
    assert cleanup_orphaned_provenance(temp_db) is True
    assert not _provenance_file(temp_db).exists()
    assert not rotated.exists()
    # Idempotent: nothing left to clean.
    assert cleanup_orphaned_provenance(temp_db) is False


def test_enabled_recall_with_no_matches_writes_empty_results(
        temp_db, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_RECALL_PROVENANCE", "1")
    beam = BeamMemory(session_id="prov-h", db_path=temp_db)

    results = beam.recall("zzz-nothing-matches-this", top_k=5)
    assert results == []

    assert _provenance_file(temp_db).exists()
    record = _read_last_record(temp_db)
    assert record["query"] == "zzz-nothing-matches-this"
    assert record["results"] == []


def test_write_failure_does_not_break_recall(temp_db, monkeypatch):
    """A provenance file path that cannot be written (a directory sits
    there) must fail open: recall returns normally with its results."""
    monkeypatch.setenv("MNEMOSYNE_RECALL_PROVENANCE", "1")
    _provenance_file(temp_db).mkdir(parents=True)
    beam = BeamMemory(session_id="prov-i", db_path=temp_db)
    beam.remember("pluto iota provenance fact nine", source="test")

    results = beam.recall("pluto", top_k=5)
    assert results


def test_concurrent_appends_never_interleave(temp_db, monkeypatch):
    """8 threads x 25 appends to one store: exactly 200 lines, every
    line intact JSON, every record present exactly once."""
    monkeypatch.delenv("MNEMOSYNE_RECALL_PROVENANCE", raising=False)

    def work(worker: int) -> None:
        for i in range(25):
            append_recall_provenance(
                str(temp_db), f"w{worker}-{i}",
                [{"id": f"m{worker}-{i}", "tier": "working", "score": 1.0}],
                top_k=5,
            )

    threads = [
        threading.Thread(target=work, args=(w,)) for w in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = _provenance_file(temp_db).read_text(
        encoding="utf-8").splitlines()
    assert len(lines) == 200
    queries = [json.loads(line)["query"] for line in lines]
    assert Counter(queries) == Counter(
        f"w{w}-{i}" for w in range(8) for i in range(25)
    )


def test_polyphonic_recall_writes_no_provenance_line(temp_db, monkeypatch):
    """Under MNEMOSYNE_POLYPHONIC_RECALL=1, recall() returns the engine's
    poly_results before the linear provenance hook runs. The documented
    exclusion must hold even with the provenance flag enabled. Uses an
    injected fake engine (same idiom as test_e3a3_cross_tier_dedup) so
    the test never depends on embeddings backends."""
    from mnemosyne.core.polyphonic_recall import PolyphonicResult

    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
    monkeypatch.setenv("MNEMOSYNE_RECALL_PROVENANCE", "1")
    beam = BeamMemory(session_id="prov-j", db_path=temp_db)
    real_id = beam.remember("pluto kappa provenance fact ten", source="test")

    engine_calls: list[dict] = []

    class _FakePolyEngine:
        def recall(self, **kwargs):
            engine_calls.append(kwargs)
            return [
                PolyphonicResult(
                    # _recall_polyphonic re-fetches every returned id from
                    # the db and silently drops unknown ones, so the fake
                    # must return the real seeded row's id.
                    memory_id=real_id,
                    combined_score=0.9,
                    voice_scores={"vector": 0.9},
                    metadata={},
                )
            ]

    monkeypatch.setattr(
        beam, "_get_polyphonic_engine", lambda: _FakePolyEngine()
    )

    results = beam.recall("pluto", top_k=5)

    assert engine_calls, "polyphonic engine path must have run"
    assert results, "polyphonic results must be returned to caller"
    assert not _provenance_file(temp_db).exists()
