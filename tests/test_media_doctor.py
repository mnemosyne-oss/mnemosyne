"""Doctor coverage for the media sidecar tables (RFC 0003 §2.4).

The load-bearing behaviour: **both orphan kinds are counted, only one warns.**
A moment whose asset is gone is real corruption. A moment whose memory row is
gone is the expected steady state after consolidation -- warning about those
would make the check cry wolf at every healthy media user after their first
``sleep()``, which trains people to ignore the half that catches real damage.
"""

import sqlite3

from mnemosyne.doctor import (
    ReferenceContractRegistry,
    STATUS_UNKNOWN,
    open_readonly_doctor_db,
)

_BASE_DDL = """
CREATE TABLE working_memory (id TEXT PRIMARY KEY);
CREATE TABLE memory_embeddings (memory_id TEXT PRIMARY KEY, embedding_json TEXT);
CREATE TABLE media_assets (asset_id TEXT PRIMARY KEY, ref_kind TEXT, ref_value TEXT);
CREATE TABLE media_moments (
  moment_id TEXT PRIMARY KEY, asset_id TEXT, memory_id TEXT,
  kind TEXT, text TEXT, span_kind TEXT, span_key TEXT
);
INSERT INTO working_memory VALUES ('mem-live');
INSERT INTO media_assets VALUES ('asset-live', 'url', 'http://x/1');
"""


def _fixture(tmp_path, ddl):
    db_path = tmp_path / "media-doctor.db"
    writable = sqlite3.connect(db_path)
    writable.executescript(ddl)
    writable.commit()
    writable.close()
    return open_readonly_doctor_db(db_path)


def _moment(moment_id, asset_id, memory_id):
    value = "NULL" if memory_id is None else f"'{memory_id}'"
    return (
        f"INSERT INTO media_moments VALUES "
        f"('{moment_id}', '{asset_id}', {value}, 'caption', 'x', 'whole', 'v1:whole');"
    )


def _inspect(tmp_path, ddl):
    conn = _fixture(tmp_path, ddl)
    try:
        return ReferenceContractRegistry(conn).inspect()
    finally:
        conn.close()


def test_absent_tables_report_not_configured(tmp_path):
    """The doctor connection is read-only, so the tables-absent case must take
    this branch and never attempt DDL."""
    result = _inspect(tmp_path, "CREATE TABLE working_memory (id TEXT PRIMARY KEY);")
    assert result.metrics["media_moments"] == {
        "status": "not_configured", "asset_orphans": 0, "memory_orphans": 0,
    }


def test_a_half_present_schema_reports_not_configured(tmp_path):
    result = _inspect(
        tmp_path,
        "CREATE TABLE working_memory (id TEXT PRIMARY KEY);"
        "CREATE TABLE media_moments (moment_id TEXT PRIMARY KEY, asset_id TEXT, memory_id TEXT);",
    )
    assert result.metrics["media_moments"]["status"] == "not_configured"


def test_a_healthy_media_install_is_clean(tmp_path):
    result = _inspect(tmp_path, _BASE_DDL + _moment("m1", "asset-live", "mem-live"))
    assert result.metrics["media_moments"] == {
        "status": "checked", "asset_orphans": 0, "memory_orphans": 0,
    }
    assert [f for f in result.findings if "media" in f.code] == []


def test_a_consolidated_memory_is_counted_without_a_warning(tmp_path):
    """The expected steady state: media_moments.text remains authoritative
    after sleep() summarizes the memory row away."""
    result = _inspect(tmp_path, _BASE_DDL + _moment("m1", "asset-live", "mem-gone"))
    assert result.metrics["media_moments"] == {
        "status": "checked", "asset_orphans": 0, "memory_orphans": 1,
    }
    assert [f for f in result.findings if "media" in f.code] == [], (
        "memory-orphans must not warn -- doing so trains users to ignore the "
        "asset-orphan finding that catches real corruption"
    )


def test_an_orphaned_asset_reference_raises_a_finding(tmp_path):
    result = _inspect(tmp_path, _BASE_DDL + _moment("m1", "asset-gone", "mem-live"))
    assert result.metrics["media_moments"] == {
        "status": "checked", "asset_orphans": 1, "memory_orphans": 0,
    }
    codes = [f.code for f in result.findings]
    assert "references.media_moment_orphan_asset" in codes


def test_an_unbound_moment_is_not_an_orphan(tmp_path):
    """A NULL memory_id is a moment awaiting binding, not a dangling one."""
    result = _inspect(tmp_path, _BASE_DDL + _moment("m1", "asset-live", None))
    assert result.metrics["media_moments"]["memory_orphans"] == 0


def test_media_findings_never_become_repair_candidates(tmp_path):
    """Deletion and cascade semantics are unspecified in v1 (RFC 0003 §5.4);
    doctor reports, it does not offer to delete."""
    result = _inspect(tmp_path, _BASE_DDL + _moment("m1", "asset-gone", "mem-gone"))
    assert result.repair_candidates == []


def test_media_metrics_leak_no_content(tmp_path):
    import json

    ddl = _BASE_DDL + (
        "INSERT INTO media_moments VALUES "
        "('m1', 'asset-live', 'mem-live', 'caption', 'private caption text', "
        "'whole', 'v1:whole');"
    )
    result = _inspect(tmp_path, ddl)
    assert "private caption text" not in json.dumps(result.metrics)


def test_unknown_catalog_includes_the_media_key(tmp_path):
    """Every metric name the adapter can emit must also appear in its unknown
    fallback, or a corrupt catalog produces a report missing a key that
    downstream renderers expect."""
    conn = _fixture(tmp_path, _BASE_DDL)
    try:
        metrics = ReferenceContractRegistry._unknown_metrics("DatabaseError")
    finally:
        conn.close()
    assert metrics["media_moments"] == {
        "status": STATUS_UNKNOWN, "error_class": "DatabaseError",
    }


def test_counts_stay_bounded_by_the_scan_limit(tmp_path):
    """A pathological database must not produce an unbounded report. The limit
    has to clear media_moments' column count (8) so the column probe still
    succeeds -- below that the adapter correctly reports it cannot tell."""
    scan_limit = 8
    ddl = _BASE_DDL + "".join(
        _moment(f"m{i}", "asset-gone", "mem-gone") for i in range(20)
    )
    conn = _fixture(tmp_path, ddl)
    try:
        metrics = ReferenceContractRegistry(conn, scan_limit=scan_limit).inspect().metrics
    finally:
        conn.close()
    media = metrics["media_moments"]
    assert media["status"] == "scan_limited"
    assert media["asset_orphans"] == scan_limit
    assert media["asset_orphans_truncated"] is True


def test_a_truncated_column_probe_reports_unknown_not_healthy(tmp_path):
    """Below the column count the adapter cannot verify the contract, and must
    say so rather than report zero orphans."""
    conn = _fixture(tmp_path, _BASE_DDL + _moment("m1", "asset-gone", "mem-gone"))
    try:
        metrics = ReferenceContractRegistry(conn, scan_limit=1).inspect().metrics
    finally:
        conn.close()
    assert metrics["media_moments"] == {
        "status": STATUS_UNKNOWN, "columns_truncated": True,
    }
