"""Regression coverage for accepted issue #821 review blockers."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
import types
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERMES_SRC = ROOT / "integrations" / "hermes" / "src"


def _run(script: str, env: dict[str, str]) -> dict:
    process_env = os.environ.copy()
    process_env.update(env)
    process_env["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(HERMES_SRC)))
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=process_env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert env.get("SECRET", "not-present") not in result.stderr
    return json.loads(result.stdout)


def test_public_sdk_cannot_claim_internal_write_exemption(tmp_path: Path):
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core.canonical import CanonicalStore
    from mnemosyne.core.filters import WritePolicySnapshot, write_policy_operation
    from mnemosyne.core.memory import Mnemosyne

    facade = Mnemosyne(session_id="public-facade", db_path=tmp_path / "facade.db")
    beam = BeamMemory(session_id="public-beam", db_path=tmp_path / "beam.db")
    canonical = CanonicalStore(db_path=tmp_path / "canonical.db")
    media_beam = BeamMemory(session_id="public-media", db_path=tmp_path / "media.db")
    strict = WritePolicySnapshot((r"^ISSUE821",), "strict")
    try:
        with write_policy_operation(strict):
            for claimed_kind in ("restore", "system_derived"):
                assert facade.remember(
                    "ISSUE821 facade exemption claim", _write_kind=claimed_kind
                ) is None
                assert beam.remember(
                    "ISSUE821 beam exemption claim", _write_kind=claimed_kind
                ) is None
                assert canonical.remember(
                    "owner", "identity", "name",
                    "ISSUE821 canonical exemption claim",
                    _write_kind=claimed_kind,
                ) is None
                media_result = media_beam.remember_media(
                    "ISSUE821 media exemption claim", _write_kind=claimed_kind
                )
                assert media_result.status == "filtered"
                assert media_result.asset_id == ""

        assert facade.conn.execute(
            "SELECT COUNT(*) FROM working_memory"
        ).fetchone()[0] == 0
        assert facade.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0
        assert beam.conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == 0
        assert canonical.conn.execute(
            "SELECT COUNT(*) FROM canonical_facts"
        ).fetchone()[0] == 0
        assert media_beam.conn.execute(
            "SELECT COUNT(*) FROM media_assets"
        ).fetchone()[0] == 0
        assert media_beam.conn.execute(
            "SELECT COUNT(*) FROM working_memory"
        ).fetchone()[0] == 0
    finally:
        facade.conn.close()
        beam.conn.close()
        canonical.conn.close()
        media_beam.conn.close()


def test_direct_remember_canonical_rejects_before_store_initialization(
    tmp_path: Path,
):
    from mnemosyne.core.canonical import remember_canonical
    from mnemosyne.core.filters import WritePolicySnapshot, write_policy_operation

    parent = tmp_path / "new-parent"
    db_path = parent / "canonical.db"
    with write_policy_operation(WritePolicySnapshot((r"^ISSUE821",), "strict")):
        result = remember_canonical(
            "owner", "identity", "name", "ISSUE821 rejected canonical body",
            db_path=db_path,
        )

    assert result is None
    assert not db_path.exists()
    assert not parent.exists()


def test_file_import_restore_exemption_and_null_accounting(tmp_path: Path):
    from mnemosyne.core.filters import WritePolicySnapshot, write_policy_operation
    from mnemosyne.core.importers.base import import_from_file
    from mnemosyne.core.memory import Mnemosyne

    source = tmp_path / "restore.json"
    source.write_text(json.dumps([
        {"content": ""},
        {"content": "allowed restored row"},
        {"content": "ISSUE821 restored row"},
    ]))
    memory = Mnemosyne(session_id="restore", db_path=tmp_path / "restore.db")
    strict = WritePolicySnapshot((r"^ISSUE821",), "strict")
    try:
        with write_policy_operation(strict):
            result = import_from_file(str(source), memory)
        assert (result.total, result.imported, result.skipped, result.failed) == (3, 2, 0, 0)
        assert len(result.memory_ids) == 2 and all(result.memory_ids)
        assert {
            memory.beam.get(memory_id)["content"] for memory_id in result.memory_ids
        } == {"allowed restored row", "ISSUE821 restored row"}
    finally:
        memory.conn.close()

    class RejectingMemory:
        def remember(self, **kwargs):
            return None if kwargs["content"].startswith("ISSUE821") else "allowed-id"

    rejected = import_from_file(str(source), RejectingMemory())
    assert (rejected.total, rejected.imported, rejected.skipped, rejected.failed) == (
        3, 1, 1, 0,
    )
    assert rejected.memory_ids == ["allowed-id"]


def test_update_boundary_provider_batch_and_importance_only(tmp_path: Path):
    import hermes_memory_provider
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core.filters import WritePolicySnapshot, write_policy_operation

    beam = BeamMemory(session_id="updates", db_path=tmp_path / "updates.db")
    provider = hermes_memory_provider.MnemosyneMemoryProvider.__new__(
        hermes_memory_provider.MnemosyneMemoryProvider
    )
    provider._beam = beam
    provider._default_scope = "session"
    provider._audit_event = lambda *_args, **_kwargs: None
    strict = WritePolicySnapshot((r"^ISSUE821",), "strict")
    try:
        memory_id = beam.remember("allowed original")
        with write_policy_operation(strict):
            direct = json.loads(provider._handle_update({
                "memory_id": memory_id, "content": "ISSUE821 provider secret"
            }))
            batch = json.loads(provider._handle_batch({"operations": [{
                "action": "update", "memory_id": memory_id,
                "content": "ISSUE821 batch secret",
            }]}))
            assert beam.update_working(memory_id, importance=0.91) is True
        assert direct == {"status": "filtered", "memory_id": memory_id}
        assert batch["results"] == [
            {"index": 0, "action": "update", "status": "filtered"}
        ]
        assert "ISSUE821" not in json.dumps((direct, batch))
        row = beam.get(memory_id)
        assert row["content"] == "allowed original"
        assert row["importance"] == pytest.approx(0.91)
    finally:
        beam.conn.close()


def test_facade_update_rejects_before_both_sql_stores(tmp_path: Path, monkeypatch):
    from mnemosyne.core import filters
    from mnemosyne.core.filters import WritePolicySnapshot
    from mnemosyne.core.memory import Mnemosyne

    memory = Mnemosyne(session_id="facade-update", db_path=tmp_path / "facade-update.db")
    memory_id = memory.remember("allowed facade original")
    strict = WritePolicySnapshot((r"^ISSUE821",), "strict")
    resolutions = 0

    def resolve_once():
        nonlocal resolutions
        resolutions += 1
        return strict

    monkeypatch.setattr(filters, "resolve_write_policy", resolve_once)
    try:
        assert memory.update(memory_id, content="ISSUE821 facade replacement") is None
        assert resolutions == 1
        assert memory.beam.get(memory_id)["content"] == "allowed facade original"
        legacy = memory.conn.execute(
            "SELECT content FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        assert legacy[0] == "allowed facade original"
    finally:
        memory.conn.close()


def test_direct_mcp_rejections_are_content_free(tmp_path: Path, monkeypatch, caplog):
    from mnemosyne import mcp_tools
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core.filters import WritePolicySnapshot, write_policy_operation
    from mnemosyne.core.memory import Mnemosyne

    private = Mnemosyne(session_id="mcp_default", db_path=tmp_path / "private.db")
    surface = BeamMemory(session_id="mcp_shared_surface", db_path=tmp_path / "surface.db")
    monkeypatch.setattr(mcp_tools, "_create_instance", lambda **_kwargs: private)
    monkeypatch.setattr(mcp_tools, "_create_surface_instance", lambda: surface)
    secret = "ISSUE821 sk-proj-abcdefghijklmnopqrstuvwxyz1234567890"
    strict = WritePolicySnapshot((r"^ISSUE821",), "strict")
    try:
        with write_policy_operation(strict), caplog.at_level("DEBUG"):
            normal = mcp_tools._handle_remember({"content": secret})
            shared = mcp_tools._handle_shared_remember({"content": secret, "kind": "meta"})
        assert normal == {"status": "filtered", "bank": "default"}
        assert shared == {"status": "filtered_shared", "kind": "meta"}
        assert "memory_id" not in json.dumps((normal, shared))
        assert secret not in caplog.text
        assert private.conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == 0
        assert surface.conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == 0
    finally:
        private.conn.close()
        surface.conn.close()


def test_sleep_proposal_is_system_derived_exempt(tmp_path: Path, monkeypatch):
    from mnemosyne.core import model_refresh
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core.filters import WritePolicySnapshot, write_policy_operation

    beam = BeamMemory(session_id="derived", db_path=tmp_path / "derived.db")
    old = (datetime.now() - timedelta(hours=200)).isoformat()
    for index in range(2):
        beam.conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
            "VALUES (?, ?, 'conversation', ?, 'derived')",
            (f"source-{index}", f"evidence {index}", old),
        )
    beam.conn.commit()
    monkeypatch.setattr(model_refresh, "infer_model_update_proposals", lambda _items: [{
        "category": "model:workflow", "name": "issue821",
        "body": "ISSUE821 derived proposal", "confidence": 0.5,
        "evidence_ids": ["source-0", "source-1"], "action": "update",
        "reason": "derived",
    }])
    try:
        with write_policy_operation(WritePolicySnapshot(("ISSUE821",), "strict")):
            result = beam.sleep(dry_run=False)
        proposals = model_refresh.list_model_refresh_proposals(beam, status="all", limit=10)
        assert result["model_refresh"]["proposals"] == 1
        assert len(proposals) == 1 and proposals[0]["id"]
    finally:
        beam.conn.close()


def test_consolidate_rejects_raw_summary_before_all_mutation(
    tmp_path: Path, monkeypatch, caplog
):
    from mnemosyne.core import beam as beam_module
    from mnemosyne.core import filters
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core.filters import WritePolicySnapshot

    marker = "ISSUE821 rejected episodic summary"
    strict = WritePolicySnapshot((r"^ISSUE821",), "strict")
    resolutions = 0
    embedding_calls = 0
    events = []

    def resolve_once():
        nonlocal resolutions
        resolutions += 1
        return strict

    def embedding_available():
        nonlocal embedding_calls
        embedding_calls += 1
        return True

    monkeypatch.setattr(filters, "resolve_write_policy", resolve_once)
    monkeypatch.setattr(beam_module._embeddings, "available", embedding_available)
    beam = BeamMemory(
        session_id="episodic-admission",
        db_path=tmp_path / "episodic-admission.db",
        event_emitter=events.append,
    )
    try:
        with caplog.at_level("DEBUG"):
            result = beam.consolidate_to_episodic(
                marker,
                source_wm_ids=[],
                source="sleep_consolidation",
            )
        assert result is None
        assert resolutions == 1
        assert embedding_calls == 0
        assert events == []
        assert beam.conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0] == 0
        assert beam.conn.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()[0] == 0
        assert marker not in caplog.text
    finally:
        beam.conn.close()


def test_sleep_consolidation_is_system_derived_exempt(tmp_path: Path, monkeypatch):
    from mnemosyne.core import beam as beam_module
    from mnemosyne.core import filters
    from mnemosyne.core import local_llm, model_refresh
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core.filters import WritePolicySnapshot

    beam = BeamMemory(session_id="derived-summary", db_path=tmp_path / "derived-summary.db")
    old = (datetime.now() - timedelta(hours=200)).isoformat()
    for index in range(2):
        beam.conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
            "VALUES (?, ?, 'conversation', ?, 'derived-summary')",
            (f"summary-source-{index}", f"allowed evidence {index}", old),
        )
    beam.conn.commit()
    monkeypatch.setattr(local_llm, "llm_available", lambda: False)
    monkeypatch.setattr(model_refresh, "infer_model_update_proposals", lambda _items: [])
    monkeypatch.setattr(beam_module._embeddings, "available", lambda: False)
    strict = WritePolicySnapshot((r"^\[conversation\]",), "strict")
    resolutions = 0

    def resolve_once():
        nonlocal resolutions
        resolutions += 1
        return strict

    monkeypatch.setattr(filters, "resolve_write_policy", resolve_once)
    try:
        result = beam.sleep(dry_run=False)
        rows = beam.conn.execute(
            "SELECT content FROM episodic_memory ORDER BY rowid"
        ).fetchall()
        assert resolutions == 1
        assert result["items_consolidated"] == 2
        assert result["summaries_created"] == 1
        assert len(rows) == 1
        assert rows[0][0].startswith("[conversation]")
    finally:
        beam.conn.close()


def test_direct_mcp_triple_add_admits_annotation_and_triple_objects(
    tmp_path: Path, monkeypatch, caplog
):
    from mnemosyne import mcp_tools
    from mnemosyne.core.annotations import AnnotationStore
    from mnemosyne.core.filters import WritePolicySnapshot, write_policy_operation
    from mnemosyne.core.memory import Mnemosyne
    from mnemosyne.core.triples import TripleStore

    memory = Mnemosyne(session_id="mcp-triples", db_path=tmp_path / "mcp-triples.db")
    monkeypatch.setattr(mcp_tools, "_create_instance", lambda **_kwargs: memory)
    annotations = AnnotationStore(db_path=memory.beam.db_path, conn=memory.beam.conn)
    triples = TripleStore(db_path=memory.beam.db_path)
    existing_id = triples.add("user", "prefers", "allowed old value")
    marker = "ISSUE821 rejected triple object"
    strict = WritePolicySnapshot((r"^ISSUE821",), "strict")
    try:
        with write_policy_operation(strict), caplog.at_level("DEBUG"):
            annotation = mcp_tools._handle_triple_add({
                "subject": "memory-1", "predicate": "mentions", "object": marker,
            })
            triple = mcp_tools._handle_triple_add({
                "subject": "user", "predicate": "prefers", "object": marker,
            })
            allowed = mcp_tools._handle_triple_add({
                "subject": "memory-1", "predicate": "mentions", "object": "Alice",
            })
        assert annotation == {"status": "filtered", "store": "annotations"}
        assert triple == {"status": "filtered", "store": "triples"}
        assert "ISSUE821" not in json.dumps((annotation, triple))
        assert marker not in caplog.text
        annotation_rows = annotations.query_by_kind("mentions", memory_id="memory-1")
        assert [row["value"] for row in annotation_rows] == ["Alice"]
        rows = triples.conn.execute(
            "SELECT id, object, valid_until FROM triples WHERE subject = ? AND predicate = ?",
            ("user", "prefers"),
        ).fetchall()
        assert [(row[0], row[1], row[2]) for row in rows] == [
            (existing_id, "allowed old value", None)
        ]
        assert allowed["status"] == "added" and allowed["store"] == "annotations"
    finally:
        triples.conn.close()
        memory.conn.close()


def test_direct_add_triple_admits_object_before_supersede(tmp_path: Path):
    from mnemosyne.core.filters import WritePolicySnapshot, write_policy_operation
    from mnemosyne.core.triples import TripleStore, add_triple

    db_path = tmp_path / "direct-add-triple.db"
    marker = "ISSUE821 rejected add_triple object"

    with write_policy_operation(WritePolicySnapshot((r"^ISSUE821",), "strict")):
        existing_id = add_triple(
            "user", "prefers", "allowed old value", db_path=db_path
        )
        rejected = add_triple("user", "prefers", marker, db_path=db_path)

    triples = TripleStore(db_path=db_path)
    try:
        assert rejected is None
        rows = triples.conn.execute(
            "SELECT id, object, valid_until FROM triples WHERE subject = ? AND predicate = ?",
            ("user", "prefers"),
        ).fetchall()
        assert [(row[0], row[1], row[2]) for row in rows] == [
            (existing_id, "allowed old value", None)
        ]
        assert marker not in str([tuple(row) for row in rows])
    finally:
        triples.conn.close()


def test_direct_triple_store_add_admits_object_before_supersede(tmp_path: Path):
    from mnemosyne.core.filters import WritePolicySnapshot, write_policy_operation
    from mnemosyne.core.triples import TripleStore

    triples = TripleStore(db_path=tmp_path / "direct-store-add.db")
    marker = "ISSUE821 rejected TripleStore.add object"
    try:
        with write_policy_operation(WritePolicySnapshot((r"^ISSUE821",), "strict")):
            existing_id = triples.add("user", "prefers", "allowed old value")
            rejected = triples.add("user", "prefers", marker)

        assert rejected is None
        rows = triples.conn.execute(
            "SELECT id, object, valid_until FROM triples WHERE subject = ? AND predicate = ?",
            ("user", "prefers"),
        ).fetchall()
        assert [(row[0], row[1], row[2]) for row in rows] == [
            (existing_id, "allowed old value", None)
        ]
        assert marker not in str([tuple(row) for row in rows])
    finally:
        triples.conn.close()


def test_direct_sdk_annotation_routes_admit_raw_values_before_mutation(
    tmp_path: Path, caplog
):
    from mnemosyne.core.annotations import AnnotationStore, add_annotation
    from mnemosyne.core.filters import WritePolicySnapshot, write_policy_operation
    from mnemosyne.core.triples import TripleStore

    db_path = tmp_path / "direct-annotations.db"
    annotations = AnnotationStore(db_path=db_path)
    triples = TripleStore(db_path=db_path)
    rejected = "ISSUE821 rejected annotation value"
    allowed = "allowed annotation value"
    try:
        with write_policy_operation(
            WritePolicySnapshot((r"^ISSUE821",), "strict")
        ), caplog.at_level("DEBUG"):
            add_result = annotations.add("memory-add-rejected", "fact", rejected)
            add_allowed = annotations.add("memory-add", "fact", allowed)
            many_result = annotations.add_many(
                "memory-many", "fact", [rejected, allowed]
            )
            helper_result = add_annotation(
                "memory-helper-rejected", "fact", rejected, db_path=db_path
            )
            helper_allowed = add_annotation(
                "memory-helper", "fact", allowed, db_path=db_path
            )
            with pytest.warns(DeprecationWarning):
                facts_result = triples.add_facts(
                    "memory-facts", [rejected, allowed]
                )

        assert add_result is None
        assert isinstance(add_allowed, int)
        assert many_result == 1
        assert helper_result is None
        assert isinstance(helper_allowed, int)
        assert facts_result == 1
        rows = annotations.conn.execute(
            "SELECT memory_id, value FROM annotations ORDER BY memory_id"
        ).fetchall()
        assert [tuple(row) for row in rows] == [
            ("memory-add", allowed),
            ("memory-facts", allowed),
            ("memory-helper", allowed),
            ("memory-many", allowed),
        ]
        assert rejected not in str([tuple(row) for row in rows])
        assert rejected not in caplog.text
    finally:
        triples.conn.close()
        annotations.conn.close()


def test_admitted_memory_persists_system_derived_entity_mentions(tmp_path: Path):
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core.filters import WritePolicySnapshot, write_policy_operation

    strict = WritePolicySnapshot((r"^Alice",), "strict")
    beam = BeamMemory(session_id="entity-enrichment", db_path=tmp_path / "entities.db")
    try:
        with write_policy_operation(strict):
            memory_id = beam.remember(
                "I met Alice yesterday", extract_entities=True
            )
            direct_result = beam.annotations.add_many(
                "direct-annotation",
                "mentions",
                ["Alice"],
            )

        rows = beam.annotations.query_by_kind("mentions", filter_noise=False)
        assert memory_id is not None
        assert direct_result == 0
        assert [(row["memory_id"], row["value"]) for row in rows] == [
            (memory_id, "Alice")
        ]
    finally:
        beam.conn.close()


@pytest.mark.parametrize("provider_name", ["hermes_memory_provider", "mnemosyne_hermes"])
def test_provider_triple_add_admits_object_before_supersede(
    tmp_path: Path, provider_name: str, caplog
):
    import importlib

    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core.filters import WritePolicySnapshot
    from mnemosyne.core.triples import TripleStore

    module = importlib.import_module(provider_name)
    provider = module.MnemosyneMemoryProvider.__new__(module.MnemosyneMemoryProvider)
    provider._beam = BeamMemory(
        session_id=f"provider-triples-{provider_name}",
        db_path=tmp_path / f"{provider_name}.db",
    )
    provider._write_policy = WritePolicySnapshot((r"^ISSUE821",), "strict")
    triples = TripleStore(db_path=provider._beam.db_path)
    existing_id = triples.add("user", "prefers", "allowed old value")
    marker = "ISSUE821 rejected provider triple object"
    try:
        with caplog.at_level("DEBUG"):
            rejected = json.loads(provider._handle_triple_add({
                "subject": "user", "predicate": "prefers", "object": marker,
            }))
        assert rejected == {"status": "filtered"}
        assert marker not in caplog.text
        rows = triples.conn.execute(
            "SELECT id, object, valid_until FROM triples WHERE subject = ? AND predicate = ?",
            ("user", "prefers"),
        ).fetchall()
        assert [(row[0], row[1], row[2]) for row in rows] == [
            (existing_id, "allowed old value", None)
        ]
    finally:
        triples.conn.close()
        provider._beam.conn.close()


_SYNC_SCRIPT = r"""
import importlib, json, os, sys, types
from pathlib import Path
h = types.ModuleType("hermes_constants")
h.get_hermes_home = lambda: Path(os.environ["HERMES_HOME"])
sys.modules.setdefault("hermes_constants", h)
m = importlib.import_module(os.environ["PROVIDER"])
p = m.MnemosyneMemoryProvider()
before = {k: os.environ.get(k) for k in ("MNEMOSYNE_IGNORE_PATTERNS", "MNEMOSYNE_WRITE_CLASSIFIER")}
kwargs = dict(hermes_home=os.environ["HERMES_HOME"], sync_roles=["user", "assistant"], auto_sleep=False)
if os.environ.get("INIT_PATTERN"): kwargs["ignore_patterns"] = [os.environ["INIT_PATTERN"]]
p.initialize("issue821", **kwargs)
p.sync_turn(os.environ["SECRET"], "allowed assistant response")
rows = [r[0] for r in p._beam.conn.execute("SELECT content FROM working_memory ORDER BY rowid")]
after = {k: os.environ.get(k) for k in before}
print(json.dumps({"rows": rows, "same_env": before == after, "mode": p._write_policy.classifier_mode,
                  "patterns": p._write_policy.ignore_patterns}))
"""


@pytest.mark.parametrize("provider", ["hermes_memory_provider", "mnemosyne_hermes"])
@pytest.mark.parametrize("case", ["hermes", "initialize", "builtin"])
def test_provider_sync_effective_config_and_raw_admission(
    tmp_path: Path, provider: str, case: str
):
    home = tmp_path / "hermes"
    data = tmp_path / "data"
    home.mkdir(); data.mkdir()
    pattern = "^ISSUE821"
    if case == "hermes":
        config = "memory:\n  mnemosyne:\n    ignore_patterns: ['^ISSUE821']\n    write_classifier: strict\n"
        init_pattern, secret, expected, expected_mode = "", "ISSUE821 raw anchored", [pattern], "strict"
    elif case == "initialize":
        config = "memory:\n  mnemosyne: {}\n"
        init_pattern, secret, expected, expected_mode = pattern, "ISSUE821 kwargs anchored", [pattern], "off"
    else:
        config = "memory:\n  mnemosyne:\n    write_classifier: strict\n"
        init_pattern, secret, expected, expected_mode = "", "$ pip install requests --quiet", ["CONFLICT"], "strict"
    (home / "config.yaml").write_text(config)
    payload = _run(_SYNC_SCRIPT, {
        "PROVIDER": provider, "HERMES_HOME": str(home), "MNEMOSYNE_DATA_DIR": str(data),
        "MNEMOSYNE_IGNORE_PATTERNS": "CONFLICT", "MNEMOSYNE_WRITE_CLASSIFIER": "off",
        "MNEMOSYNE_NO_EMBEDDINGS": "1", "MNEMOSYNE_HOST_LLM_ENABLED": "0",
        "INIT_PATTERN": init_pattern, "SECRET": secret,
    })
    assert payload == {
        "rows": ["[ASSISTANT] allowed assistant response"], "same_env": True,
        "mode": expected_mode, "patterns": expected,
    }


_APPROVAL_SCRIPT = r"""
import importlib, json, logging, os, sys, types
from pathlib import Path

h = types.ModuleType("hermes_constants")
h.get_hermes_home = lambda: Path(os.environ["HERMES_HOME"])
sys.modules.setdefault("hermes_constants", h)
hc = types.ModuleType("hermes_cli.config")
hc.load_config = lambda: {"memory": {"write_approval": True}}
hc.cfg_get = lambda cfg, *keys, default=None: (
    cfg.get(keys[0], {}).get(keys[1], default) if len(keys) == 2 else default
)
hp = types.ModuleType("hermes_cli")
hp.__path__ = []
hp.config = hc
sys.modules.setdefault("hermes_cli", hp)
sys.modules.setdefault("hermes_cli.config", hc)

module = importlib.import_module(os.environ["PROVIDER"])
provider = module.MnemosyneMemoryProvider()
provider.initialize(
    "issue821-approval",
    hermes_home=os.environ["HERMES_HOME"],
    ignore_patterns=[r"^ISSUE821"],
    write_classifier="strict",
    auto_sleep=False,
)
allowed_id = provider._beam.remember(
    "allowed original", _write_policy=type(provider._write_policy)((), "off")
)
invalidate_id = provider._beam.remember(
    "allowed invalidate target", _write_policy=type(provider._write_policy)((), "off")
)
replacement_id = provider._beam.remember(
    "allowed replacement", _write_policy=type(provider._write_policy)((), "off")
)

class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []
    def emit(self, record):
        self.messages.append(self.format(record))

capture = Capture()
logging.getLogger().addHandler(capture)
marker = os.environ["SECRET"]
remember = provider.handle_tool_call("mnemosyne_remember", {"content": marker})
batch = provider.handle_tool_call("mnemosyne_batch", {"operations": [
    {"action": "remember", "content": marker + " remember"},
    {"action": "update", "memory_id": allowed_id, "content": marker + " update"},
]})
pending_root = Path(os.environ["HERMES_HOME"]) / "pending"
pending_files = list(pending_root.rglob("*")) if pending_root.exists() else []
pending_text = "\n".join(
    path.read_text(errors="replace") for path in pending_files if path.is_file()
)
approved_update_batch = json.loads(provider.handle_tool_call(
    "mnemosyne_batch", {"operations": [
        {"action": "remember", "content": "allowed approved batch remember"},
        {
            "action": "update", "memory_id": allowed_id,
            "content": "allowed approved update", "importance": 0.83,
        },
        {
            "action": "invalidate", "memory_id": invalidate_id,
            "replacement_id": replacement_id,
        },
    ]}
))
approved_pending_ids = approved_update_batch["pending_ids"]
unsupported_pending = module._stage_pending_write({
    "tool": "mnemosyne_batch", "action": "unsupported",
    "content": "allowed but unsupported",
})
approved_update_apply = json.loads(provider.handle_tool_call(
    "mnemosyne_apply_pending",
    {"pending_ids": approved_pending_ids + [unsupported_pending]},
))
non_content = json.loads(provider.handle_tool_call("mnemosyne_batch", {"operations": [
    {"action": "update", "memory_id": allowed_id, "importance": 0.9},
    {"action": "forget", "memory_id": allowed_id},
]}))
non_content_records = []
for result in non_content["results"]:
    record_path = pending_root / "memory" / (result["pending_id"] + ".json")
    non_content_records.append(json.loads(record_path.read_text())["payload"])
    record_path.unlink()
rejected_pending = module._stage_pending_write({
    "tool": "mnemosyne_remember", "content": marker + " apply"
})
allowed_pending = module._stage_pending_write({
    "tool": "mnemosyne_remember", "content": "allowed pending content"
})
apply_response = provider.handle_tool_call(
    "mnemosyne_apply_pending", {"pending_ids": [rejected_pending, allowed_pending]}
)
print(json.dumps({
    "remember": json.loads(remember),
    "batch": json.loads(batch),
    "pending_exists": pending_root.exists(),
    "pending_entries": [str(path.relative_to(pending_root)) for path in pending_files],
    "pending_text": pending_text,
    "non_content": non_content,
    "non_content_records": non_content_records,
    "apply_response": json.loads(apply_response),
    "approved_update_batch": approved_update_batch,
    "approved_update_apply": approved_update_apply,
    "approved_pending_ids": approved_pending_ids,
    "unsupported_pending": unsupported_pending,
    "allowed_id": allowed_id,
    "invalidate_id": invalidate_id,
    "replacement_id": replacement_id,
    "pending_after_apply": [str(path) for path in pending_root.rglob("*.json")],
    "marker_rows": provider._beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE content LIKE '%ISSUE821%'"
    ).fetchone()[0],
    "allowed_rows": provider._beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE content = 'allowed pending content'"
    ).fetchone()[0],
    "approved_batch_remember_rows": provider._beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory "
        "WHERE content = 'allowed approved batch remember'"
    ).fetchone()[0],
    "approved_batch_remember_id": provider._beam.conn.execute(
        "SELECT id FROM working_memory "
        "WHERE content = 'allowed approved batch remember'"
    ).fetchone()[0],
    "working_rows": provider._beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory"
    ).fetchone()[0],
    "logs": capture.messages,
    "original": provider._beam.get(allowed_id)["content"],
    "updated_importance": provider._beam.get(allowed_id)["importance"],
    "invalidation": dict(provider._beam.conn.execute(
        "SELECT valid_until, superseded_by FROM working_memory WHERE id = ?",
        (invalidate_id,),
    ).fetchone()),
}))
"""


@pytest.mark.parametrize("provider", ["hermes_memory_provider", "mnemosyne_hermes"])
def test_write_approval_rejects_before_pending_persistence(
    tmp_path: Path, provider: str
):
    home = tmp_path / "hermes"
    data = tmp_path / "data"
    home.mkdir(); data.mkdir()
    (home / "config.yaml").write_text("memory:\n  write_approval: true\n")
    marker = "ISSUE821 pending persistence marker"
    payload = _run(_APPROVAL_SCRIPT, {
        "PROVIDER": provider,
        "HERMES_HOME": str(home),
        "MNEMOSYNE_DATA_DIR": str(data),
        "MNEMOSYNE_NO_EMBEDDINGS": "1",
        "MNEMOSYNE_HOST_LLM_ENABLED": "0",
        "SECRET": marker,
    })
    assert payload["remember"] == {"status": "filtered"}
    assert payload["batch"]["status"] == "filtered"
    assert payload["batch"]["results"] == [
        {"index": 0, "action": "remember", "status": "filtered"},
        {"index": 1, "action": "update", "status": "filtered"},
    ]
    assert payload["pending_entries"] == []
    assert payload["original"] == "allowed approved update"
    assert payload["updated_importance"] == pytest.approx(0.83)
    assert payload["approved_update_batch"]["status"] == "staged"
    assert payload["approved_update_batch"]["count"] == 3
    assert payload["approved_update_batch"]["filtered_count"] == 0
    assert "staged" not in payload["approved_update_batch"]
    assert "staged_count" not in payload["approved_update_batch"]
    assert payload["approved_update_batch"]["message"] == (
        "3 writes staged for approval. Use mnemosyne_apply_pending to commit."
    )
    assert payload["approved_update_batch"]["pending_ids"] == [
        result["pending_id"] for result in payload["approved_update_batch"]["results"]
    ]
    assert all(
        isinstance(pending_id, str)
        for pending_id in payload["approved_update_batch"]["pending_ids"]
    )
    assert payload["approved_update_apply"] == {
        "applied": [
            {
                "id": payload["approved_pending_ids"][0],
                "memory_id": payload["approved_batch_remember_id"],
            },
            {
                "id": payload["approved_pending_ids"][1],
                "memory_id": payload["allowed_id"],
            },
            {
                "id": payload["approved_pending_ids"][2],
                "memory_id": payload["invalidate_id"],
            },
        ],
        "failed": [{"id": payload["unsupported_pending"],
                    "error": "unsupported action"}],
        "applied_count": 3,
        "failed_count": 1,
    }
    assert payload["approved_batch_remember_rows"] == 1
    assert payload["invalidation"]["valid_until"] is not None
    assert payload["invalidation"]["superseded_by"] == payload["replacement_id"]
    assert payload["non_content"]["status"] == "staged"
    assert [result["status"] for result in payload["non_content"]["results"]] == [
        "staged", "staged",
    ]
    assert [record["memory_id"] for record in payload["non_content_records"]] == [
        payload["allowed_id"],
        payload["allowed_id"],
    ]
    assert payload["apply_response"]["applied_count"] == 1
    assert payload["apply_response"]["failed_count"] == 1
    assert payload["apply_response"]["failed"][0]["error"] == "filtered"
    assert payload["pending_after_apply"] == []
    assert payload["marker_rows"] == 0
    assert payload["allowed_rows"] == 1
    assert payload["working_rows"] == 5
    assert marker not in json.dumps(payload)


_PENDING_RETRY_SCRIPT = r"""
import importlib, json, os, sys, types
from pathlib import Path

h = types.ModuleType("hermes_constants")
h.get_hermes_home = lambda: Path(os.environ["HERMES_HOME"])
sys.modules.setdefault("hermes_constants", h)
hc = types.ModuleType("hermes_cli.config")
hc.load_config = lambda: {"memory": {"write_approval": True}}
hc.cfg_get = lambda cfg, *keys, default=None: (
    cfg.get(keys[0], {}).get(keys[1], default) if len(keys) == 2 else default
)
hp = types.ModuleType("hermes_cli")
hp.__path__ = []
hp.config = hc
sys.modules.setdefault("hermes_cli", hp)
sys.modules.setdefault("hermes_cli.config", hc)

module = importlib.import_module(os.environ["PROVIDER"])
provider = module.MnemosyneMemoryProvider()
provider.initialize(
    "issue821-pending-retry",
    hermes_home=os.environ["HERMES_HOME"],
    auto_sleep=False,
)
pending_dir = Path(os.environ["HERMES_HOME"]) / "pending" / "memory"


def stage(action, memory_id, **payload):
    return module._stage_pending_write({
        "tool": "mnemosyne_batch",
        "action": action,
        "memory_id": memory_id,
        **payload,
    })


def apply(pending_id):
    return json.loads(provider.handle_tool_call(
        "mnemosyne_apply_pending", {"pending_ids": [pending_id]}
    ))


def exists(pending_id):
    return (pending_dir / f"{pending_id}.json").is_file()


update_target = "pendingupdatetarget"
update_pending = stage("update", update_target, content="updated after retry")
update_missing = apply(update_pending)
update_retained = exists(update_pending)
provider._beam.remember(
    "original before retry", memory_id=update_target, dedupe=False
)
update_retried = apply(update_pending)
update_cleaned = not exists(update_pending)
update_content = provider._beam.get(update_target)["content"]

forget_target = "pendingforgettarget"
provider._beam.remember("forget once", memory_id=forget_target, dedupe=False)
forget_pending = stage("forget", forget_target)
forget_applied = apply(forget_pending)
forget_cleaned = not exists(forget_pending)
forget_again_pending = stage("forget", forget_target)
forget_already_satisfied = apply(forget_again_pending)
forget_terminal_cleaned = not exists(forget_again_pending)

forget_transient_target = "pendingforgettransient"
provider._beam.remember(
    "forget after transient failure", memory_id=forget_transient_target, dedupe=False
)
forget_transient_pending = stage("forget", forget_transient_target)
original_forget = provider._beam.forget_working

def temporary_forget_failure(_memory_id):
    raise RuntimeError("temporary forget failure")

provider._beam.forget_working = temporary_forget_failure
forget_transient = apply(forget_transient_pending)
forget_transient_retained = exists(forget_transient_pending)
provider._beam.forget_working = original_forget
forget_retried = apply(forget_transient_pending)
forget_retry_cleaned = not exists(forget_transient_pending)

invalidate_missing_pending = stage("invalidate", "pendinginvalidatemissing")
invalidate_missing = apply(invalidate_missing_pending)
invalidate_terminal_cleaned = not exists(invalidate_missing_pending)

invalidate_target = "pendinginvalidatetarget"
invalidate_replacement = "pendinginvalidatereplacement"
provider._beam.remember(
    "invalidate after replacement appears", memory_id=invalidate_target, dedupe=False
)
invalidate_pending = stage(
    "invalidate", invalidate_target, replacement_id=invalidate_replacement
)
invalidate_replacement_missing = apply(invalidate_pending)
invalidate_retained = exists(invalidate_pending)
provider._beam.remember(
    "replacement appears", memory_id=invalidate_replacement, dedupe=False
)
invalidate_retried = apply(invalidate_pending)
invalidate_retry_cleaned = not exists(invalidate_pending)
invalidate_row = dict(provider._beam.conn.execute(
    "SELECT valid_until, superseded_by FROM working_memory WHERE id = ?",
    (invalidate_target,),
).fetchone())

print(json.dumps({
    "update_missing": update_missing,
    "update_retained": update_retained,
    "update_retried": update_retried,
    "update_cleaned": update_cleaned,
    "update_content": update_content,
    "forget_applied": forget_applied,
    "forget_cleaned": forget_cleaned,
    "forget_already_satisfied": forget_already_satisfied,
    "forget_terminal_cleaned": forget_terminal_cleaned,
    "forget_transient": forget_transient,
    "forget_transient_retained": forget_transient_retained,
    "forget_retried": forget_retried,
    "forget_retry_cleaned": forget_retry_cleaned,
    "invalidate_missing": invalidate_missing,
    "invalidate_terminal_cleaned": invalidate_terminal_cleaned,
    "invalidate_replacement_missing": invalidate_replacement_missing,
    "invalidate_retained": invalidate_retained,
    "invalidate_retried": invalidate_retried,
    "invalidate_retry_cleaned": invalidate_retry_cleaned,
    "invalidate_row": invalidate_row,
    "pending_files": sorted(path.name for path in pending_dir.glob("*.json")),
}))
"""


@pytest.mark.parametrize("provider", ["hermes_memory_provider", "mnemosyne_hermes"])
def test_pending_mutation_retry_and_terminal_cleanup(
    tmp_path: Path, provider: str
):
    home = tmp_path / "hermes"
    data = tmp_path / "data"
    home.mkdir(); data.mkdir()
    payload = _run(_PENDING_RETRY_SCRIPT, {
        "PROVIDER": provider,
        "HERMES_HOME": str(home),
        "MNEMOSYNE_DATA_DIR": str(data),
        "MNEMOSYNE_NO_EMBEDDINGS": "1",
        "MNEMOSYNE_HOST_LLM_ENABLED": "0",
    })

    update_pending_id = payload["update_missing"]["failed"][0]["id"]
    assert payload["update_missing"] == {
        "applied": [],
        "failed": [{"id": update_pending_id, "error": "memory not found"}],
        "applied_count": 0,
        "failed_count": 1,
    }
    assert payload["update_retained"] is True
    assert payload["update_retried"] == {
        "applied": [{"id": update_pending_id, "memory_id": "pendingupdatetarget"}],
        "failed": [],
        "applied_count": 1,
        "failed_count": 0,
    }
    assert payload["update_cleaned"] is True
    assert payload["update_content"] == "updated after retry"

    assert payload["forget_applied"]["applied_count"] == 1
    assert payload["forget_cleaned"] is True
    assert payload["forget_already_satisfied"]["applied"] == []
    assert payload["forget_already_satisfied"]["failed"][0]["error"] == (
        "memory not found"
    )
    assert payload["forget_terminal_cleaned"] is True
    assert payload["forget_transient"]["failed"][0]["error"] == (
        "temporary forget failure"
    )
    assert payload["forget_transient_retained"] is True
    assert payload["forget_retried"]["applied_count"] == 1
    assert payload["forget_retry_cleaned"] is True

    assert payload["invalidate_missing"]["applied"] == []
    assert payload["invalidate_missing"]["failed"][0]["error"] == (
        "memory not found"
    )
    assert payload["invalidate_terminal_cleaned"] is True
    assert payload["invalidate_replacement_missing"]["applied"] == []
    assert payload["invalidate_replacement_missing"]["failed"][0]["error"] == (
        "memory not found"
    )
    assert payload["invalidate_retained"] is True
    assert payload["invalidate_retried"]["applied_count"] == 1
    assert payload["invalidate_retry_cleaned"] is True
    assert payload["invalidate_row"]["valid_until"] is not None
    assert payload["invalidate_row"]["superseded_by"] == (
        "pendinginvalidatereplacement"
    )
    assert payload["pending_files"] == []


def test_facade_data_uri_is_admitted_before_blob_or_sql_mutation(
    tmp_path: Path, monkeypatch
):
    from mnemosyne.core import filters
    from mnemosyne.core.filters import WritePolicySnapshot
    from mnemosyne.core.memory import Mnemosyne

    blob_dir = tmp_path / "blobs"
    monkeypatch.setenv("MNEMOSYNE_BLOB_DIR", str(blob_dir))
    raw = b"issue 821 binary"
    content = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    strict = WritePolicySnapshot((r"^data:",), "strict")
    resolutions = 0

    def resolve_once():
        nonlocal resolutions
        resolutions += 1
        return strict

    monkeypatch.setattr(filters, "resolve_write_policy", resolve_once)
    memory = Mnemosyne(session_id="facade-data-uri", db_path=tmp_path / "facade.db")
    try:
        assert memory.remember(content) is None
        assert resolutions == 1
        assert memory.conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == 0
        assert memory.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0
        assert not blob_dir.exists()
    finally:
        memory.conn.close()


def test_remember_media_data_uri_is_admitted_before_blob_or_sql_mutation(
    tmp_path: Path, monkeypatch
):
    from mnemosyne.core import filters
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core.filters import WritePolicySnapshot

    blob_dir = tmp_path / "media-blobs"
    monkeypatch.setenv("MNEMOSYNE_BLOB_DIR", str(blob_dir))
    content = "data:image/png;base64," + base64.b64encode(
        b"issue 821 media binary"
    ).decode("ascii")
    strict = WritePolicySnapshot((r"^data:",), "strict")
    resolutions = 0

    def resolve_once():
        nonlocal resolutions
        resolutions += 1
        return strict

    monkeypatch.setattr(filters, "resolve_write_policy", resolve_once)
    beam = BeamMemory(session_id="media-data-uri", db_path=tmp_path / "media.db")
    try:
        result = beam.remember_media(content)
        assert result.status == "filtered"
        assert result.asset_id == ""
        assert resolutions == 1
        assert beam.conn.execute("SELECT COUNT(*) FROM media_assets").fetchone()[0] == 0
        assert beam.conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == 0
        assert not blob_dir.exists()
    finally:
        beam.conn.close()


def test_remember_media_allowed_data_uri_reuses_operation_snapshot(
    tmp_path: Path, monkeypatch
):
    from mnemosyne.core import filters
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core.filters import WritePolicySnapshot

    blob_dir = tmp_path / "allowed-media-blobs"
    monkeypatch.setenv("MNEMOSYNE_BLOB_DIR", str(blob_dir))
    content = "data:image/png;base64," + base64.b64encode(
        b"allowed issue 821 media binary"
    ).decode("ascii")
    allowed = WritePolicySnapshot((), "off")
    resolutions = 0

    def resolve_once():
        nonlocal resolutions
        resolutions += 1
        return allowed

    monkeypatch.setattr(filters, "resolve_write_policy", resolve_once)
    beam = BeamMemory(
        session_id="allowed-media-data-uri", db_path=tmp_path / "allowed-media.db"
    )
    try:
        result = beam.remember_media(content)
        assert result.status == "unavailable"
        assert resolutions == 1
        asset = beam.media.get_asset(result.asset_id)
        assert asset is not None
        assert asset["ref_value"].startswith("blob://sha256/")
        assert "anchor_memory_id" in json.loads(asset["metadata"])
        assert beam.conn.execute("SELECT COUNT(*) FROM media_assets").fetchone()[0] == 1
        assert beam.conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == 1
        assert any(path.is_file() for path in blob_dir.rglob("*"))
    finally:
        beam.conn.close()


def test_facade_allowed_data_uri_preserves_blob_extraction(
    tmp_path: Path, monkeypatch
):
    from mnemosyne.core.filters import WritePolicySnapshot, write_policy_operation
    from mnemosyne.core.memory import Mnemosyne

    blob_dir = tmp_path / "allowed-blobs"
    monkeypatch.setenv("MNEMOSYNE_BLOB_DIR", str(blob_dir))
    content = "data:image/png;base64," + base64.b64encode(b"allowed binary").decode("ascii")
    memory = Mnemosyne(session_id="facade-data-uri-allowed", db_path=tmp_path / "allowed.db")
    try:
        with write_policy_operation(WritePolicySnapshot((), "off")):
            memory_id = memory.remember(content)
        assert memory_id is not None
        row = memory.beam.get(memory_id)
        assert row is not None
        assert row["content"].startswith("[Binary content extracted")
        metadata = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
        assert metadata["_blob"]["blob_ref"].startswith("blob://sha256/")
        assert any(path.is_file() for path in blob_dir.rglob("*"))
    finally:
        memory.conn.close()


def test_lazy_tool_initialization_precedes_policy_snapshot():
    from contextlib import contextmanager

    import mnemosyne_hermes
    from mnemosyne.core.filters import WritePolicySnapshot, current_write_policy

    provider = mnemosyne_hermes.MnemosyneMemoryProvider.__new__(
        mnemosyne_hermes.MnemosyneMemoryProvider
    )
    provider._write_policy = None
    provider._reflect_disabled_for_cron = False
    provider._agent_context = ""
    provider.has_tool = lambda _name: True
    strict = WritePolicySnapshot((r"^ISSUE821",), "strict")
    calls = []

    def initialize_policy():
        calls.append("retry")
        provider._write_policy = strict

    provider._maybe_retry_init = initialize_policy
    provider._ensure_initialized_for_tools = lambda: calls.append("ensure")

    def resolve_policy():
        calls.append("resolve")
        return provider._write_policy

    provider._resolve_effective_write_policy = resolve_policy

    @contextmanager
    def beam_scope(_session_id):
        yield object()

    provider._beam_session_scope = beam_scope

    def dispatch(_tool_name, _args):
        calls.append("dispatch")
        return json.dumps({"mode": current_write_policy().classifier_mode})

    provider._dispatch_tool_call_locked = dispatch

    result = json.loads(provider.handle_tool_call("mnemosyne_remember", {}))

    assert result == {"mode": "strict"}
    assert calls == ["retry", "ensure", "resolve", "dispatch"]


@pytest.mark.parametrize(
    ("provider_module", "dispatch_name"),
    [
        ("hermes_memory_provider", "_dispatch_tool_call"),
        ("mnemosyne_hermes", "_dispatch_tool_call_locked"),
    ],
)
def test_provider_write_dispatch_policy_and_session_share_lifecycle_lock(
    provider_module: str, dispatch_name: str
):
    """A lifecycle transition cannot split policy resolution from dispatch."""
    import importlib

    from mnemosyne.core.filters import WritePolicySnapshot, current_write_policy

    module = importlib.import_module(provider_module)
    provider = module.MnemosyneMemoryProvider()
    old_policy = WritePolicySnapshot((), "off")
    new_policy = WritePolicySnapshot((r"^LIFECYCLE821",), "strict")
    provider._write_policy = old_policy
    provider._session_id = "old-session"
    provider._beam = types.SimpleNamespace(
        session_id="old-session", channel_id="old-session"
    )
    provider.has_tool = lambda _name: True
    lifecycle_started = threading.Event()
    release_lifecycle = threading.Event()
    policy_resolved = threading.Event()
    errors: list[BaseException] = []
    result: list[str] = []

    def initialize_locked(_session_id: str, **_kwargs) -> None:
        lifecycle_started.set()
        if not release_lifecycle.wait(timeout=5):
            raise AssertionError("timed out waiting to finish lifecycle transition")
        provider._write_policy = new_policy
        provider._session_id = "new-session"
        provider._beam.session_id = "new-session"
        provider._beam.channel_id = "new-session"

    provider._initialize_locked = initialize_locked
    if provider_module == "mnemosyne_hermes":
        provider._maybe_retry_init = lambda: None
        provider._ensure_initialized_for_tools = lambda: None

    def resolve_policy():
        policy_resolved.set()
        return provider._write_policy

    provider._resolve_effective_write_policy = resolve_policy

    def dispatch(_tool_name: str, _args: dict, **_kwargs) -> str:
        return json.dumps(
            {
                "mode": current_write_policy().classifier_mode,
                "session_id": provider._session_id,
                "beam_session_id": provider._beam.session_id,
            }
        )

    setattr(provider, dispatch_name, dispatch)

    def run_initialize() -> None:
        try:
            provider.initialize("new-session")
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def run_dispatch() -> None:
        try:
            result.append(provider.handle_tool_call("mnemosyne_remember", {}))
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    lifecycle = threading.Thread(target=run_initialize)
    dispatch_thread = threading.Thread(target=run_dispatch)
    lifecycle.start()
    assert lifecycle_started.wait(timeout=5)
    dispatch_thread.start()
    try:
        assert not policy_resolved.wait(timeout=0.1)
    finally:
        release_lifecycle.set()
        lifecycle.join(timeout=5)
        dispatch_thread.join(timeout=5)

    assert not lifecycle.is_alive()
    assert not dispatch_thread.is_alive()
    assert not errors
    assert policy_resolved.is_set()
    assert [json.loads(item) for item in result] == [
        {
            "mode": "strict",
            "session_id": "new-session",
            "beam_session_id": "new-session",
        }
    ]


@pytest.mark.parametrize(
    "provider_module", ["hermes_memory_provider", "mnemosyne_hermes"]
)
def test_provider_write_policy_reloads_between_sync_operations(
    tmp_path: Path, monkeypatch, provider_module: str
):
    import importlib

    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setenv("MNEMOSYNE_HOST_LLM_ENABLED", "0")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    hermes_home = tmp_path / provider_module
    hermes_home.mkdir()
    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        "memory:\n"
        "  mnemosyne:\n"
        "    ignore_patterns: ['^REFRESH821']\n"
        "    write_classifier: strict\n"
        "    sync_roles: [user, assistant]\n"
    )
    module = importlib.import_module(provider_module)
    provider = module.MnemosyneMemoryProvider()
    provider.initialize(
        "policy-refresh",
        hermes_home=str(hermes_home),
        auto_sleep=False,
    )
    assert provider._beam is not None

    real_remember = provider._beam.remember
    calls = 0

    def remember_while_config_changes(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            config_path.write_text(
                "memory:\n"
                "  mnemosyne:\n"
                "    ignore_patterns: []\n"
                "    write_classifier: off\n"
                "    sync_roles: [user, assistant]\n"
            )
        return real_remember(*args, **kwargs)

    provider._beam.remember = remember_while_config_changes
    try:
        provider.sync_turn(
            "REFRESH821 first user content",
            "REFRESH821 first assistant content",
            session_id="policy-refresh",
        )
        assert calls == 2
        assert provider._beam.conn.execute(
            "SELECT COUNT(*) FROM working_memory WHERE content LIKE '%REFRESH821%'"
        ).fetchone()[0] == 0

        response = json.loads(provider.handle_tool_call(
            "mnemosyne_batch",
            {"operations": [
                {"action": "remember", "content": "REFRESH821 second user content"},
                {
                    "action": "remember",
                    "content": "REFRESH821 second assistant content",
                },
            ]},
        ))
        assert [item["status"] for item in response["results"]] == [
            "stored",
            "stored",
        ]
        assert calls == 4
        rows = provider._beam.conn.execute(
            "SELECT content FROM working_memory WHERE content LIKE '%REFRESH821%' "
            "ORDER BY rowid"
        ).fetchall()
        assert [row[0] for row in rows] == [
            "REFRESH821 second user content",
            "REFRESH821 second assistant content",
        ]
        assert provider._write_policy.classifier_mode == "off"
        assert provider._write_policy.ignore_patterns == ()
    finally:
        provider.shutdown()


@pytest.mark.parametrize(
    "provider_module", ["hermes_memory_provider", "mnemosyne_hermes"]
)
def test_provider_initialize_policy_overrides_remain_sticky(
    tmp_path: Path, monkeypatch, provider_module: str
):
    import importlib

    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setenv("MNEMOSYNE_HOST_LLM_ENABLED", "0")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    hermes_home = tmp_path / provider_module
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "memory:\n"
        "  mnemosyne:\n"
        "    ignore_patterns: []\n"
        "    write_classifier: off\n"
        "    sync_roles: [user, assistant]\n"
    )
    module = importlib.import_module(provider_module)
    provider = module.MnemosyneMemoryProvider()
    provider.initialize(
        "sticky-policy",
        hermes_home=str(hermes_home),
        auto_sleep=False,
        ignore_patterns=[r"^STICKY821"],
        write_classifier="strict",
    )
    assert provider._beam is not None
    try:
        # Re-initializing without policy kwargs must retain explicit overrides.
        provider.initialize(
            "sticky-policy",
            hermes_home=str(hermes_home),
            auto_sleep=False,
        )
        provider.sync_turn(
            "STICKY821 user content",
            "STICKY821 assistant content",
            session_id="sticky-policy",
        )
        assert provider._beam.conn.execute(
            "SELECT COUNT(*) FROM working_memory WHERE content LIKE '%STICKY821%'"
        ).fetchone()[0] == 0
        assert provider._write_policy.classifier_mode == "strict"
        assert provider._write_policy.ignore_patterns == (r"^STICKY821",)
        provider._resolve_effective_write_policy = lambda: pytest.fail(
            "read-only tools must not refresh write policy"
        )
        assert "error" not in json.loads(
            provider.handle_tool_call("mnemosyne_stats", {})
        )
    finally:
        provider.shutdown()


def test_fact_enrichment_reuses_one_policy_snapshot(tmp_path: Path, monkeypatch):
    from mnemosyne.core import extraction, filters
    from mnemosyne.core.filters import WritePolicySnapshot
    from mnemosyne.core.memory import Mnemosyne

    strict = WritePolicySnapshot((r"^ISSUE821",), "strict")
    resolutions = 0

    def resolve_once():
        nonlocal resolutions
        resolutions += 1
        return strict

    monkeypatch.setattr(filters, "resolve_write_policy", resolve_once)
    monkeypatch.setattr(
        extraction,
        "extract_facts_safe",
        lambda _content: ["ISSUE821 generated fact must not persist"],
    )
    memory = Mnemosyne(session_id="fact-snapshot", db_path=tmp_path / "facts.db")
    try:
        memory_id = memory.remember(
            "Allowed note about Alice",
            source="document",
            extract=True,
            extract_entities=True,
        )
        assert memory_id is not None
        assert resolutions == 1
        assert memory.beam.annotations.query_by_memory(memory_id, kind="fact") == []
        mentions = memory.beam.annotations.query_by_memory(memory_id, kind="mentions")
        assert "Alice" in [row["value"] for row in mentions]
        assert len(memory.beam.annotations.query_by_memory(
            memory_id, kind="occurred_on"
        )) == 1
        assert [row["value"] for row in memory.beam.annotations.query_by_memory(
            memory_id, kind="has_source"
        )] == ["document"]
        assert memory.conn.execute(
            "SELECT COUNT(*) FROM facts WHERE object LIKE 'ISSUE821%'"
        ).fetchone()[0] == 0
    finally:
        memory.conn.close()


def test_system_derived_temporal_annotations_preserve_exemption(
    tmp_path: Path, monkeypatch
):
    from mnemosyne.core import filters
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core.filters import (
        _SYSTEM_DERIVED_WRITE_CAPABILITY,
        WritePolicySnapshot,
    )

    strict = WritePolicySnapshot((r".*",), "strict")
    monkeypatch.setattr(
        filters,
        "resolve_write_policy",
        lambda: pytest.fail("derived remember must not resolve another policy"),
    )
    beam = BeamMemory(session_id="derived-annotations", db_path=tmp_path / "derived.db")
    try:
        memory_id = beam.remember(
            "derived content",
            source="derived-source",
            _write_kind=_SYSTEM_DERIVED_WRITE_CAPABILITY,
            _write_policy=strict,
        )
        assert memory_id is not None
        assert len(beam.annotations.query_by_memory(memory_id, kind="occurred_on")) == 1
        assert [row["value"] for row in beam.annotations.query_by_memory(
            memory_id, kind="has_source"
        )] == ["derived-source"]
    finally:
        beam.conn.close()


def test_fact_ids_keep_original_extraction_position(tmp_path: Path):
    from mnemosyne.core.beam import BeamMemory, _store_facts_in_table
    from mnemosyne.core.filters import WritePolicySnapshot

    beam = BeamMemory(session_id="stable-facts", db_path=tmp_path / "facts.db")
    facts = ["ISSUE821 rejected earlier fact", "allowed later fact"]
    try:
        memory_id = beam.remember("allowed source memory")
        _store_facts_in_table(
            beam, memory_id, "allowed source memory", "test", facts,
            write_policy=WritePolicySnapshot((r"^ISSUE821",), "strict"),
        )
        first_rows = beam.conn.execute(
            "SELECT fact_id, object FROM facts WHERE source_msg_id = ? ORDER BY object",
            (memory_id,),
        ).fetchall()
        assert len(first_rows) == 1
        later_id = first_rows[0]["fact_id"]

        _store_facts_in_table(
            beam, memory_id, "allowed source memory", "test", facts,
            write_policy=WritePolicySnapshot((), "off"),
        )
        rows = beam.conn.execute(
            "SELECT fact_id, object FROM facts WHERE source_msg_id = ? ORDER BY object",
            (memory_id,),
        ).fetchall()
        assert len(rows) == 2
        assert {row["object"]: row["fact_id"] for row in rows}["allowed later fact"] == later_id
    finally:
        beam.conn.close()


def test_media_moments_are_admitted_before_store_and_bind(tmp_path: Path, monkeypatch):
    from mnemosyne.core import filters, media
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core.filters import WritePolicySnapshot
    from mnemosyne.core.modality_backends import DescribedMoment, DescribeResult

    strict = WritePolicySnapshot((r"^ISSUE821",), "strict")
    resolutions = 0

    def resolve_once():
        nonlocal resolutions
        resolutions += 1
        return strict

    monkeypatch.setattr(filters, "resolve_write_policy", resolve_once)
    monkeypatch.setattr(
        media,
        "_describe",
        lambda *_args, **_kwargs: DescribeResult(
            provider="stub",
            moments=[
                DescribedMoment(kind="caption", text="ISSUE821 blocked caption"),
                DescribedMoment(
                    kind="ocr", text="allowed OCR", bbox=[0.1, 0.2, 0.3, 0.4]
                ),
            ],
        ),
    )
    beam = BeamMemory(session_id="media-snapshot", db_path=tmp_path / "media.db")
    try:
        result = beam.remember_media("https://example.test/allowed.png")
        assert result.status == "partial"
        assert resolutions == 1
        moments = beam.media.get_moments(result.asset_id)
        assert [row["text"] for row in moments] == ["allowed OCR"]
        assert [row["memory_id"] for row in moments] == result.memory_ids
        assert len(result.moment_ids) == len(result.memory_ids) == 1
        assert beam.conn.execute(
            "SELECT COUNT(*) FROM media_moments WHERE text LIKE 'ISSUE821%' "
            "OR (text = 'allowed OCR' AND memory_id IS NULL)"
        ).fetchone()[0] == 0
        assert beam.conn.execute(
            "SELECT COUNT(*) FROM working_memory WHERE content LIKE 'ISSUE821%'"
        ).fetchone()[0] == 0
        assert any("write policy" in warning for warning in result.warnings)
    finally:
        beam.conn.close()


def test_media_summary_filtered_by_policy_is_unavailable(tmp_path: Path, monkeypatch):
    from mnemosyne.core import media
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core.filters import WritePolicySnapshot
    from mnemosyne.core.modality_backends import DescribeResult

    strict = WritePolicySnapshot((r"^ISSUE821",), "strict")
    monkeypatch.setattr(
        media,
        "_describe",
        lambda *_args, **_kwargs: DescribeResult(
            provider="stub",
            summary="ISSUE821 blocked summary",
        ),
    )
    beam = BeamMemory(session_id="filtered-summary", db_path=tmp_path / "summary.db")
    try:
        result = beam.remember_media(
            "https://example.test/allowed-summary.png",
            _write_policy=strict,
        )
        assert result.status == "unavailable"
        assert result.moment_ids == []
        assert result.memory_ids == []
        assert beam.media.get_moments(result.asset_id) == []
        assert any("write policy" in warning for warning in result.warnings)
    finally:
        beam.conn.close()


def test_system_derived_media_moments_preserve_exemption(tmp_path: Path, monkeypatch):
    from mnemosyne.core import media
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core.filters import (
        _SYSTEM_DERIVED_WRITE_CAPABILITY,
        WritePolicySnapshot,
    )
    from mnemosyne.core.modality_backends import DescribedMoment, DescribeResult

    strict = WritePolicySnapshot((r"^ISSUE821",), "strict")
    monkeypatch.setattr(
        media,
        "_describe",
        lambda *_args, **_kwargs: DescribeResult(
            provider="stub",
            moments=[
                DescribedMoment(kind="caption", text="ISSUE821 exempt caption"),
                DescribedMoment(kind="ocr", text="allowed exempt OCR"),
            ],
        ),
    )
    beam = BeamMemory(session_id="derived-media", db_path=tmp_path / "derived-media.db")
    try:
        result = beam.remember_media(
            "https://example.test/allowed-derived.png",
            _write_kind=_SYSTEM_DERIVED_WRITE_CAPABILITY,
            _write_policy=strict,
        )
        assert result.status == "ok"
        moments = beam.media.get_moments(result.asset_id)
        assert [row["text"] for row in moments] == [
            "ISSUE821 exempt caption", "allowed exempt OCR",
        ]
        assert [row["memory_id"] for row in moments] == result.memory_ids
        assert len(result.moment_ids) == len(result.memory_ids) == 2
    finally:
        beam.conn.close()


def test_update_filtered_result_typing_and_user_surfaces(
    tmp_path: Path, monkeypatch, capsys
):
    import typing

    from mnemosyne import cli, mcp_tools
    from mnemosyne.core import memory as memory_module
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core.filters import WritePolicySnapshot, write_policy_operation
    from mnemosyne.core.memory import Mnemosyne

    optional_bool = typing.Optional[bool]
    optional_str = typing.Optional[str]
    assert typing.get_type_hints(BeamMemory.remember)["return"] == optional_str
    assert (
        typing.get_type_hints(BeamMemory.consolidate_to_episodic)["return"]
        == optional_str
    )
    assert typing.get_type_hints(Mnemosyne.remember)["return"] == optional_str
    assert typing.get_type_hints(memory_module.remember)["return"] == optional_str
    assert typing.get_type_hints(BeamMemory.update_working)["return"] == optional_bool
    assert typing.get_type_hints(Mnemosyne.update)["return"] == optional_bool
    assert typing.get_type_hints(memory_module.update)["return"] == optional_bool

    memory = Mnemosyne(session_id="update-surfaces", db_path=tmp_path / "updates.db")
    marker = "ISSUE821 rejected update content"
    strict = WritePolicySnapshot((r"^ISSUE821",), "strict")
    try:
        memory_id = memory.remember("allowed original")
        monkeypatch.setattr(mcp_tools, "_create_instance", lambda **_kwargs: memory)
        monkeypatch.setattr(cli, "_get_memory", lambda: memory)

        with write_policy_operation(strict):
            mcp_result = mcp_tools._handle_update(
                {"memory_id": memory_id, "content": marker}
            )
            with pytest.raises(SystemExit) as exc:
                cli.cmd_update([memory_id, marker])

        assert mcp_result == {"status": "filtered", "memory_id": memory_id}
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert f"Update filtered by write policy: {memory_id}" in captured.err
        assert marker not in captured.err

        with pytest.raises(SystemExit) as missing_exc:
            cli.cmd_update(["missing-id", "allowed update"])
        assert missing_exc.value.code == 1
        missing_output = capsys.readouterr()
        assert missing_output.out == ""
        assert "Memory not found: missing-id" in missing_output.err
        assert memory.beam.get(memory_id)["content"] == "allowed original"
    finally:
        memory.conn.close()


def test_wrapper_batch_adapter_does_not_retry_filtered_update():
    from types import SimpleNamespace

    from mnemosyne import mcp_tools

    class FilteredMemory:
        def __init__(self):
            self.beam = SimpleNamespace(conn=object(), update_working=self.fail_fallback)
            self.conn = object()
            self._emit_wrapper = lambda *_args, **_kwargs: None

        def update(self, *_args, **_kwargs):
            return None

        @staticmethod
        def fail_fallback(*_args, **_kwargs):
            raise AssertionError("filtered updates must not fall through to Beam")

    adapter = mcp_tools._WrapperBatchAdapter(FilteredMemory())
    assert adapter.update_working("memory", content="blocked") is None


def test_batch_fact_enrichment_reuses_batch_policy_snapshot(
    tmp_path: Path, monkeypatch
):
    from mnemosyne.core import beam as beam_module
    from mnemosyne.core import extraction, filters
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core.filters import WritePolicySnapshot

    strict = WritePolicySnapshot((r"^ISSUE821",), "strict")
    permissive = WritePolicySnapshot((), "off")
    resolutions = 0
    fact_policies = []

    def changing_policy():
        nonlocal resolutions
        resolutions += 1
        return strict if resolutions == 1 else permissive

    monkeypatch.setattr(filters, "resolve_write_policy", changing_policy)
    monkeypatch.setattr(
        extraction,
        "extract_facts_safe",
        lambda _content: ["ISSUE821 generated batch fact must not persist"],
    )
    beam = BeamMemory(session_id="batch-fact-snapshot", db_path=tmp_path / "batch.db")
    real_annotation_add_many = beam.annotations.add_many
    real_table_store = beam_module._store_facts_in_table

    def capture_annotation_policy(*args, **kwargs):
        result = real_annotation_add_many(*args, **kwargs)
        fact_policies.append(("annotations", kwargs.get("_write_policy")))
        return result

    def capture_table_policy(*args, **kwargs):
        result = real_table_store(*args, **kwargs)
        fact_policies.append(("facts", kwargs.get("write_policy")))
        return result

    monkeypatch.setattr(beam.annotations, "add_many", capture_annotation_policy)
    monkeypatch.setattr(beam_module, "_store_facts_in_table", capture_table_policy)
    try:
        memory_ids = beam.remember_batch(
            [{"content": "Allowed batch note", "source": "test"}],
            extract=True,
        )
        assert len(memory_ids) == 1
        assert fact_policies == [("annotations", strict), ("facts", strict)]
        assert beam.annotations.query_by_memory(memory_ids[0], kind="fact") == []
        assert beam.conn.execute(
            "SELECT COUNT(*) FROM facts WHERE object LIKE 'ISSUE821%'"
        ).fetchone()[0] == 0
    finally:
        beam.conn.close()


@pytest.mark.parametrize(
    ("predicate", "expected_store", "id_key"),
    [("mentions", "annotations", "annotation_id"), ("prefers", "triples", "triple_id")],
)
def test_mcp_triple_add_reuses_operation_policy_snapshot(
    tmp_path: Path, monkeypatch, predicate: str, expected_store: str, id_key: str
):
    from mnemosyne import mcp_tools
    from mnemosyne.core import filters
    from mnemosyne.core.filters import WritePolicySnapshot
    from mnemosyne.core.memory import Mnemosyne

    permissive = WritePolicySnapshot((), "off")
    strict = WritePolicySnapshot((r"^ISSUE821",), "strict")
    resolutions = 0

    def changing_policy():
        nonlocal resolutions
        resolutions += 1
        return permissive if resolutions == 1 else strict

    monkeypatch.setattr(filters, "resolve_write_policy", changing_policy)
    memory = Mnemosyne(session_id="mcp-triple-snapshot", db_path=tmp_path / "triple.db")
    monkeypatch.setattr(mcp_tools, "_create_instance", lambda **_kwargs: memory)
    try:
        result = mcp_tools._handle_triple_add({
            "subject": "memory-1",
            "predicate": predicate,
            "object": "ISSUE821 admitted by the operation snapshot",
        })
        assert result["status"] == "added"
        assert result["store"] == expected_store
        assert result[id_key]
        assert resolutions == 1
        if expected_store == "annotations":
            rows = memory.beam.annotations.query_by_memory("memory-1", kind="mentions")
            assert [row["value"] for row in rows] == [
                "ISSUE821 admitted by the operation snapshot"
            ]
        else:
            from mnemosyne.core.triples import TripleStore

            triples = TripleStore(db_path=memory.beam.db_path)
            try:
                rows = triples.query(subject="memory-1", predicate="prefers")
                assert [row["object"] for row in rows] == [
                    "ISSUE821 admitted by the operation snapshot"
                ]
            finally:
                triples.conn.close()
    finally:
        memory.conn.close()
