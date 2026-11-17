"""Target-aware previews agree with action adapters without making writes."""

import json
import pytest
from mnemosyne.core.memory import Mnemosyne
from mnemosyne import mcp_tools
from mnemosyne import batch_tool
from mnemosyne.batch_tool import (
    apply_beam_batch,
    dry_run_batch,
    validate_batch_operations,
)


@pytest.mark.parametrize(
    "surface", ["beam", "mcp", "root_provider", "packaged_provider"]
)
@pytest.mark.parametrize(
    "target", ["working", "global", "foreign", "episodic", "legacy", "missing"]
)
@pytest.mark.parametrize("action", ["update", "forget", "invalidate"])
def test_preview_vs_apply(tmp_path, monkeypatch, surface, target, action):
    mem = Mnemosyne(db_path=str(tmp_path / "mnemosyne.db"), session_id="owner")
    mid = "not-stored"
    if target in ("working", "global", "foreign"):
        mid = mem.remember(
            "The project meeting is on Friday afternoon.", scope="session"
        )
        if target in ("global", "foreign"):
            mem.beam.conn.execute(
                "UPDATE working_memory SET session_id=?, scope=? WHERE id=?",
                ("other", "global" if target == "global" else "session", mid),
            )
            mem.beam.conn.execute(
                "UPDATE memories SET session_id=? WHERE id=?", ("other", mid)
            )
    elif target == "episodic":
        mid = "episode-only"
        mem.beam.conn.execute(
            "INSERT INTO episodic_memory (id,content,session_id,scope) VALUES (?,?,?,?)",
            (mid, "Recorded episode", "owner", "session"),
        )
    elif target == "legacy":
        mid = "legacy-only"
        mem.beam.conn.execute(
            "INSERT INTO memories (id,content,session_id) VALUES (?,?,?)",
            (mid, "Legacy record", "owner"),
        )
    mem.beam.conn.commit()
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    if surface == "mcp":
        # Same caller session used by the normal MCP constructor.
        for table in ("working_memory", "episodic_memory", "memories"):
            mem.beam.conn.execute(
                f"UPDATE {table} SET session_id='mcp_default' WHERE session_id='owner'"
            )
        mem.beam.session_id = "mcp_default"
        mem.session_id = "mcp_default"
        mem.beam.conn.commit()
    op = {"action": action, "memory_id": mid}
    if action == "update":
        op["content"] = "The project meeting is now on Monday afternoon."
    operations = [op]
    normalized = validate_batch_operations(operations)
    provider = None
    if surface.endswith("provider"):
        if surface == "root_provider":
            from hermes_memory_provider import MnemosyneMemoryProvider
        else:
            from mnemosyne_hermes import MnemosyneMemoryProvider
        provider = MnemosyneMemoryProvider()
        provider._beam = mem.beam
        provider._session_id = "owner"
        provider._agent_context = "primary"
    before = list(mem.beam.conn.iterdump())
    traced = []
    mem.beam.conn.set_trace_callback(traced.append)
    if surface == "beam":
        preview = batch_tool.snapshot_preview_batch(
            normalized, target=batch_tool.BatchPreviewTarget(mem.beam.conn, "owner")
        )
    elif provider is not None:
        preview = json.loads(
            provider._handle_batch({"operations": operations, "dry_run": True})
        )
    else:
        monkeypatch.setattr(mcp_tools, "_create_instance", lambda **kwargs: mem)
        preview = mcp_tools._handle_batch(
            {"operations": operations, "dry_run": True, "bank": "default"}
        )
    mem.beam.conn.set_trace_callback(None)
    assert all(
        sql.split()[0].upper()
        in {"SELECT", "PRAGMA", "SAVEPOINT", "RELEASE", "ROLLBACK"}
        for sql in traced
    ), traced
    assert list(mem.beam.conn.iterdump()) == before
    succeeds = (
        target in ("working", "global")
        or (target == "episodic" and action != "update")
        or (surface == "mcp" and target == "legacy" and action == "update")
    )
    assert preview["validation"] == "target_eligibility_snapshot"
    assert preview["results"][0]["status"] == (
        {
            "update": "would_update",
            "forget": "would_delete",
            "invalidate": "would_invalidate",
        }[action]
        if succeeds
        else "not_found"
    )
    if surface == "beam":
        result = apply_beam_batch(mem.beam, normalized)
    elif provider is not None:
        result = json.loads(provider._handle_batch({"operations": operations}))
    else:
        result = mcp_tools._handle_batch({"operations": operations, "bank": "default"})
    succeeds = (
        target in ("working", "global")
        or (target == "episodic" and action != "update")
        or (surface == "mcp" and target == "legacy" and action == "update")
    )
    assert result["status"] == ("ok" if succeeds else "error"), (
        surface,
        target,
        action,
        result,
    )
    if not succeeds:
        assert list(mem.beam.conn.iterdump()) == before


def _target_ops(action="forget", memory_id="id", **payload):
    """Use public validation, including the same ID normalization as apply."""
    return validate_batch_operations(
        [{"action": action, "memory_id": memory_id, **payload}]
    )


def test_syntax_helper_remains_contextless_and_compatible():
    assert dry_run_batch(_target_ops()) == {
        "status": "dry_run",
        "operations_count": 1,
        "results": [
            {
                "index": 0,
                "action": "forget",
                "status": "would_delete",
                "memory_id": "id",
            }
        ],
    }


@pytest.mark.parametrize(
    "replacement", ["valid", "missing", "foreign", "expired", "superseded", "self"]
)
def test_preview_replacement_eligibility_and_normalized_id(
    tmp_path, monkeypatch, replacement
):
    """Supersession needs two visible live records, not merely two IDs."""
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION", "1")
    mem = Mnemosyne(db_path=str(tmp_path / "db"), session_id="owner")
    mid = mem.beam.remember("The office opens at nine in the morning.")
    rid = mem.beam.remember("The office opens at ten in the morning.")
    if replacement == "foreign":
        mem.beam.conn.execute(
            "UPDATE working_memory SET session_id='other',scope='session' WHERE id=?",
            (rid,),
        )
    elif replacement == "expired":
        mem.beam.conn.execute(
            "UPDATE working_memory SET valid_until='2000-01-01' WHERE id=?", (rid,)
        )
    elif replacement == "superseded":
        mem.beam.conn.execute(
            "UPDATE working_memory SET superseded_by='elsewhere' WHERE id=?", (rid,)
        )
    elif replacement == "missing":
        rid = "not-stored"
    elif replacement == "self":
        rid = mid
    mem.beam.conn.commit()
    ops = _target_ops("invalidate", " " + mid + " ", replacement_id=rid)
    result = batch_tool.snapshot_preview_batch(
        ops, target=batch_tool.BatchPreviewTarget(mem.beam.conn, "owner")
    )
    assert result["results"][0]["memory_id"] == mid
    assert result["results"][0]["status"] == (
        "would_invalidate" if replacement == "valid" else "not_found"
    )


def test_preview_preserves_callers_transaction_and_does_not_simulate_order(tmp_path):
    mem = Mnemosyne(db_path=str(tmp_path / "db"), session_id="owner")
    mem.beam.conn.execute(
        "INSERT INTO working_memory(id,content,session_id,scope) VALUES('id','Pending row','owner','session')"
    )
    assert mem.beam.conn.in_transaction
    ops = validate_batch_operations([{"action": "forget", "memory_id": "id"}] * 2)
    result = batch_tool.snapshot_preview_batch(
        ops, target=batch_tool.BatchPreviewTarget(mem.beam.conn, "owner")
    )
    assert [r["status"] for r in result["results"]] == ["would_delete", "would_delete"]
    assert mem.beam.conn.in_transaction
    mem.beam.conn.rollback()
    assert (
        mem.beam.conn.execute("SELECT 1 FROM working_memory WHERE id='id'").fetchone()
        is None
    )


def test_mcp_missing_bank_preview_creates_nothing(tmp_path, monkeypatch):
    root = tmp_path / "not-created"
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(root))
    for bank in ("default", "missing", "../escape"):
        result = mcp_tools._handle_batch(
            {
                "bank": bank,
                "dry_run": True,
                "operations": [{"action": "forget", "memory_id": "x"}],
            }
        )
        assert result == batch_tool.preview_unavailable()
    assert not root.exists()


@pytest.mark.parametrize("kind", ["empty_schema", "closed", "read_error"])
def test_unavailable_targets_fail_closed_without_error_details(tmp_path, kind):
    import sqlite3

    conn = sqlite3.connect(str(tmp_path / "db"))
    if kind == "closed":
        conn.close()
    elif kind == "read_error":
        conn.set_authorizer(lambda *args: sqlite3.SQLITE_DENY)
    result = batch_tool.snapshot_preview_batch(
        _target_ops(), target=batch_tool.BatchPreviewTarget(conn, "owner")
    )
    assert result == batch_tool.preview_unavailable()
    if kind != "closed":
        conn.close()


def test_mcp_preview_uses_requested_bank_without_initializing_wrapper(
    tmp_path, monkeypatch
):
    """Select the right bank, preserving files except SQLite reader marks."""
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    (tmp_path / "banks" / "work").mkdir(parents=True)
    work = Mnemosyne(
        db_path=str(tmp_path / "banks" / "work" / "mnemosyne.db"), session_id="mcp_work"
    )
    mid = work.beam.remember("A project target stored in the work bank.")
    Mnemosyne(db_path=str(tmp_path / "mnemosyne.db"), session_id="mcp_default")

    def forbid_initialization(**kwargs):
        raise AssertionError("preview constructed mutation-capable wrapper")

    monkeypatch.setattr(mcp_tools, "_create_instance", forbid_initialization)
    before = {
        str(p.relative_to(tmp_path)): p.read_bytes()
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    op = {"action": "forget", "memory_id": mid}
    for bank, want in [("work", "would_delete"), ("default", "not_found")]:
        result = mcp_tools._handle_batch(
            {"bank": bank, "dry_run": True, "operations": [op]}
        )
        assert result["results"][0]["status"] == want
    after = {
        str(p.relative_to(tmp_path)): p.read_bytes()
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    assert after.keys() == before.keys()
    for name, contents in before.items():
        if name.endswith("-shm"):
            # SQLite may update read coordination in existing shared memory.
            # This exception does not permit creating or truncating sidecars.
            assert len(after[name]) == len(contents)
        else:
            # Includes database, WAL, configuration and every other file.
            assert after[name] == contents, name


@pytest.mark.parametrize("package", ["hermes_memory_provider", "mnemosyne_hermes"])
def test_provider_preview_with_old_core_does_not_claim_syntax_is_target_checked(
    monkeypatch, package
):
    import importlib

    module = importlib.import_module(package)
    provider = module.MnemosyneMemoryProvider()
    provider._beam = object()
    monkeypatch.delattr(batch_tool, "snapshot_preview_batch")
    result = json.loads(
        provider._handle_batch(
            {"dry_run": True, "operations": [{"action": "forget", "memory_id": "x"}]}
        )
    )
    assert result == batch_tool.preview_unavailable()


def test_preview_keeps_one_snapshot_when_another_connection_deletes(tmp_path):
    import sqlite3

    mem = Mnemosyne(db_path=str(tmp_path / "db"), session_id="owner")
    mid = mem.beam.remember("The reception desk closes at five in the afternoon.")
    writer = sqlite3.connect(str(tmp_path / "db"))
    reads = []

    def interleave(sql):
        if sql.startswith("SELECT 1 FROM working_memory"):
            reads.append(sql)
            if len(reads) == 2:
                writer.execute("DELETE FROM working_memory WHERE id=?", (mid,))
                writer.commit()

    mem.beam.conn.set_trace_callback(interleave)
    try:
        ops = validate_batch_operations([{"action": "forget", "memory_id": mid}] * 2)
        result = batch_tool.snapshot_preview_batch(
            ops, target=batch_tool.BatchPreviewTarget(mem.beam.conn, "owner")
        )
    finally:
        mem.beam.conn.set_trace_callback(None)
        writer.close()
    assert len(reads) == 2
    assert [r["status"] for r in result["results"]] == ["would_delete", "would_delete"]
    assert apply_beam_batch(mem.beam, _target_ops("forget", mid))["status"] == "error"


@pytest.mark.parametrize("adapter", ["beam", "wrapper"])
@pytest.mark.parametrize("action", ["update", "forget", "invalidate"])
def test_collision_with_private_working_does_not_hide_visible_episode(
    tmp_path, adapter, action
):
    mem = Mnemosyne(db_path=str(tmp_path / "db"), session_id="owner")
    mem.beam.conn.execute(
        "INSERT INTO working_memory(id,content,session_id,scope) VALUES('same','Private working','other','session')"
    )
    mem.beam.conn.execute(
        "INSERT INTO episodic_memory(id,content,session_id,scope) VALUES('same','Own episode','owner','session')"
    )
    mem.beam.conn.commit()
    payload = {"content": "Corrected episode."} if action == "update" else {}
    result = batch_tool.snapshot_preview_batch(
        _target_ops(action, "same", **payload),
        target=batch_tool.BatchPreviewTarget(mem.beam.conn, "owner", adapter),
    )
    assert (
        result["results"][0]["status"]
        == {
            "update": "not_found",
            "forget": "would_delete",
            "invalidate": "would_invalidate",
        }[action]
    )


@pytest.mark.parametrize("package", ["hermes_memory_provider", "mnemosyne_hermes"])
def test_public_preview_never_initializes_or_captures_write_policy(
    tmp_path, monkeypatch, package
):
    import importlib

    provider = importlib.import_module(package).MnemosyneMemoryProvider()
    provider._beam = None
    touched = []
    for name in (
        "_maybe_retry_init",
        "_ensure_initialized_for_tools",
        "_resolve_effective_write_policy",
    ):
        if hasattr(provider, name):
            monkeypatch.setattr(provider, name, lambda name=name: touched.append(name))
    result = json.loads(
        provider.handle_tool_call(
            "mnemosyne_batch",
            {"dry_run": True, "operations": [{"action": "forget", "memory_id": "x"}]},
        )
    )
    assert touched == []
    expected = batch_tool.preview_unavailable()
    if package == "mnemosyne_hermes":
        expected["reason_code"] = "never_initialized"
    assert result == expected


def test_mcp_closed_wal_preview_does_not_create_sidecars(tmp_path, monkeypatch):
    import sqlite3

    path = tmp_path / "mnemosyne.db"
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE marker(id)")
    conn.commit()
    conn.close()
    assert set(p.name for p in tmp_path.iterdir()) == {"mnemosyne.db"}
    before = path.read_bytes()
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    result = mcp_tools._handle_batch(
        {"dry_run": True, "operations": [{"action": "forget", "memory_id": "missing"}]}
    )
    assert result == batch_tool.preview_unavailable()
    assert set(p.name for p in tmp_path.iterdir()) == {"mnemosyne.db"}
    assert path.read_bytes() == before


@pytest.mark.parametrize("package", ["hermes_memory_provider", "mnemosyne_hermes"])
def test_public_preview_keeps_changed_tool_configuration_errors_in_json(
    tmp_path, package
):
    """A post-startup exposure typo is a JSON error, not an escaping exception."""
    import importlib

    provider = importlib.import_module(package).MnemosyneMemoryProvider()
    provider._hermes_home = str(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("memory:\n  mnemosyne:\n    tools: [mnemosyne_batch]\n")
    assert provider.has_tool("mnemosyne_batch")
    config.write_text("memory:\n  mnemosyne:\n    tools: [mnemosyne_not_real]\n")

    result = json.loads(
        provider.handle_tool_call(
            "mnemosyne_batch",
            {
                "dry_run": True,
                "operations": [{"action": "forget", "memory_id": "x"}],
            },
        )
    )

    assert "Unknown Mnemosyne tool(s) in memory.mnemosyne.tools" in result["error"]
    assert "mnemosyne_not_real" in result["error"]


@pytest.mark.parametrize("package", ["hermes_memory_provider", "mnemosyne_hermes"])
@pytest.mark.parametrize(
    "action, expected",
    [
        ("update", "would_update"),
        ("forget", "would_delete"),
        ("invalidate", "would_invalidate"),
    ],
)
def test_public_active_preview_selects_targets_without_write_hooks(
    tmp_path, monkeypatch, package, action, expected
):
    """Both public dispatchers reach read-only eligibility on an active Beam."""
    import importlib
    from mnemosyne.core.beam import BeamMemory

    module = importlib.import_module(package)
    beam = BeamMemory(session_id="owner", db_path=tmp_path / "active.db")
    owned = beam.remember(
        "The project meeting starts at nine on Friday.", scope="session"
    )
    beam.conn.execute(
        "INSERT INTO working_memory(id,content,session_id,scope) VALUES(?,?,?,?)",
        ("foreign", "A private meeting starts at ten on Monday.", "other", "session"),
    )
    beam.conn.commit()
    provider = module.MnemosyneMemoryProvider()
    provider._beam = beam
    provider._session_id = "owner"
    provider._agent_context = "primary"

    def forbidden(*args, **kwargs):
        pytest.fail(
            "a preview invoked an initialization/write-policy/approval/audit hook"
        )

    for name in (
        "_maybe_retry_init",
        "_ensure_initialized_for_tools",
        "_resolve_effective_write_policy",
        "_audit_event",
    ):
        if hasattr(provider, name):
            monkeypatch.setattr(provider, name, forbidden)
    monkeypatch.setattr(module, "_write_approval_enabled", lambda: True)
    monkeypatch.setattr(module, "_stage_pending_write", forbidden)
    before = list(beam.conn.iterdump())
    operations = [{"action": action, "memory_id": mid} for mid in (owned, "foreign")]
    if action == "update":
        for operation in operations:
            operation["content"] = "The project meeting now starts at noon."

    result = json.loads(
        provider.handle_tool_call(
            "mnemosyne_batch",
            {
                "dry_run": True,
                "operations": operations,
            },
        )
    )

    assert result["validation"] == "target_eligibility_snapshot"
    assert [row["status"] for row in result["results"]] == [expected, "not_found"]
    assert list(beam.conn.iterdump()) == before


def test_mcp_preview_last_connection_close_only_recreates_reader_coordination(
    tmp_path, monkeypatch
):
    """After another connection checkpoints/closes, preview remains data-read-only."""
    import sqlite3

    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    memory = Mnemosyne(db_path=str(tmp_path / "mnemosyne.db"), session_id="mcp_default")
    mid = memory.remember("The temporary project meeting is on Friday afternoon.")
    path = tmp_path / "mnemosyne.db"
    handles = {memory.conn, memory.beam.conn}
    assert path.with_name(path.name + "-wal").exists()
    assert path.with_name(path.name + "-shm").exists()
    real_connect = sqlite3.connect
    baseline = {}
    opened = []
    statements = []

    def close_last_connection_then_open(*args, **kwargs):
        # Interleave after the real sidecar preflight, at the actual read-only open.
        assert kwargs.get("uri") is True
        assert str(args[0]).endswith("?mode=ro")
        assert "immutable" not in str(args[0])
        for handle in handles:
            handle.close()
        assert not path.with_name(path.name + "-wal").exists()
        assert not path.with_name(path.name + "-shm").exists()
        # Closing the external writer checkpoints. Attribute that separately
        # from whatever the preview does after this boundary.
        baseline.update(
            {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
        )
        conn = real_connect(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        opened.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", close_last_connection_then_open)
    result = mcp_tools._handle_batch(
        {
            "dry_run": True,
            "operations": [{"action": "forget", "memory_id": mid}],
        }
    )

    assert len(opened) == 1
    assert result["results"][0]["status"] == "would_delete"
    assert statements
    assert all(
        sql.split()[0].upper() in {"SELECT", "PRAGMA", "SAVEPOINT", "RELEASE"}
        for sql in statements
    )
    after = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    for name, contents in baseline.items():
        assert after[name] == contents, name
    assert set(after) - set(baseline) == {"mnemosyne.db-wal", "mnemosyne.db-shm"}
    assert after["mnemosyne.db-wal"] == b""  # no WAL header or user-data frames
    assert len(after["mnemosyne.db-shm"]) > 0
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0].execute("SELECT 1")
    # Inspect stored content independently after the preview, not via its target helper.
    with real_connect(str(path)) as reader:
        assert (
            reader.execute(
                "SELECT content FROM working_memory WHERE id=?", (mid,)
            ).fetchone()[0]
            == "The temporary project meeting is on Friday afternoon."
        )


@pytest.mark.parametrize("context", ["subagent", "cron"])
@pytest.mark.parametrize("was_live", [False, True], ids=["direct-skip", "live-reinit"])
def test_public_skip_preview_preserves_lifecycle_reason_without_initialization(
    tmp_path, monkeypatch, context, was_live
):
    """Preview preserves real lifecycle diagnostics without reviving a skipped Beam."""
    import importlib

    module = importlib.import_module("mnemosyne_hermes")
    provider = module.MnemosyneMemoryProvider()
    try:
        if was_live:
            provider.initialize(
                "primary", hermes_home=str(tmp_path), profile_isolation=False,
                agent_context="primary", auto_sleep=False,
            )
            assert provider._beam is not None
        provider.initialize(
            "skipped", hermes_home=str(tmp_path), profile_isolation=False,
            agent_context=context, auto_sleep=False,
        )
        assert provider._beam is None
        expected = "reset_by_reinit" if was_live else "skipped_context"
        memory_result = json.loads(
            provider.handle_tool_call("mnemosyne_recall", {"query": "marker"})
        )
        assert memory_result["reason_code"] == expected

        def forbidden(*args, **kwargs):
            pytest.fail("a skipped preview must not initialize, open SQLite, or capture a write")

        for name in (
            "_maybe_retry_init", "_ensure_initialized_for_tools",
            "_resolve_effective_write_policy", "_audit_event",
        ):
            if hasattr(provider, name):
                monkeypatch.setattr(provider, name, forbidden)
        monkeypatch.setattr(module, "_get_beam_class", forbidden)
        import sqlite3
        monkeypatch.setattr(sqlite3, "connect", forbidden)

        result = json.loads(provider.handle_tool_call("mnemosyne_batch", {
            "dry_run": True,
            "operations": [{"action": "forget", "memory_id": "marker"}],
        }))
        assert result == {**batch_tool.preview_unavailable(), "reason_code": expected}
        assert provider._beam is None
        assert provider._unavailable_reason_code == expected
    finally:
        provider.shutdown()
