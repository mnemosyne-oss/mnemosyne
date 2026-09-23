"""Focused contract tests for read-only Doctor embedding status (#1017)."""

import builtins
from dataclasses import fields
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from mnemosyne import doctor
from mnemosyne.doctor import (
    DoctorReport,
    EmbeddingsStatusAdapter,
    build_doctor_report,
    doctor_report_payload,
    open_readonly_doctor_db,
    render_doctor_json,
    render_doctor_markdown,
)


@pytest.fixture(autouse=True)
def _isolated_embedding_config(monkeypatch, tmp_path):
    """Isolate live embedding resolution from the ambient config.

    ``doctor._embedding_runtime_status`` reads the model/endpoint live
    (config.yaml > env), so an ambient ~/.hermes config would otherwise
    shadow the test doubles. An empty temp config disables seeding and
    leaves every key unset.
    """
    from mnemosyne.core.config import MnemosyneConfig

    for key in (
        "MNEMOSYNE_EMBEDDING_API_URL",
        "MNEMOSYNE_EMBEDDING_API_KEY",
        "MNEMOSYNE_EMBEDDING_MODEL",
        "MNEMOSYNE_EMBEDDING_DIM",
        "MNEMOSYNE_EMBEDDINGS_VIA_API",
    ):
        monkeypatch.delenv(key, raising=False)
    cfg_dir = tmp_path / "iso-config"
    cfg_dir.mkdir(exist_ok=True)
    (cfg_dir / "config.yaml").write_text("")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(cfg_dir))
    # Isolate the shared-HOME fallback (profile YAML > env > shared YAML).
    monkeypatch.setenv("HOME", str(tmp_path))
    MnemosyneConfig.reset_instance()
    yield
    MnemosyneConfig.reset_instance()


def _runtime(**overrides):
    values = {
        "disabled": False,
        "backend": "fastembed_local",
        "backend_available": True,
        "fastembed_installed": True,
        "fastembed_version": "0.7.3",
        "configured_model_raw": "BAAI/bge-small-en-v1.5",
        "configured_model": "BAAI/bge-small-en-v1.5",
        "configured_dimension": 3,
    }
    values.update(overrides)
    return values


def _embedding_db(tmp_path: Path, rows=(), *, ddl=None) -> Path:
    path = tmp_path / "doctor-embeddings.db"
    conn = sqlite3.connect(path)
    conn.execute(
        ddl
        or "CREATE TABLE memory_embeddings ("
        "memory_id TEXT PRIMARY KEY, embedding_json TEXT NOT NULL, model TEXT)"
    )
    if rows:
        conn.executemany(
            "INSERT INTO memory_embeddings (memory_id, embedding_json, model) VALUES (?, ?, ?)",
            rows,
        )
    conn.commit()
    conn.close()
    return path


def _inspect(path: Path, *, runtime=None, scan_limit=200):
    conn = open_readonly_doctor_db(path)
    try:
        return (
            EmbeddingsStatusAdapter(
                conn, scan_limit=scan_limit, runtime=runtime or _runtime()
            )
            .inspect()
            .metrics
        )
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("runtime", "expected_state"),
    [
        (_runtime(disabled=True), "disabled"),
        (_runtime(backend_available=False, fastembed_version=None), "unavailable"),
        (_runtime(), "available"),
        ({"error_class": "runtime_error"}, "unknown"),
    ],
)
def test_embedding_state_without_persisted_vectors(tmp_path, runtime, expected_state):
    status = _inspect(_embedding_db(tmp_path), runtime=runtime)

    assert status["state"] == expected_state
    assert status["activity_evidence"] is None
    assert status["runtime_scope"] == "current_process"
    assert status["coverage"]["persisted"]["status"] == "no_vectors"


def test_matching_persisted_vector_is_active_operational_evidence(tmp_path):
    status = _inspect(
        _embedding_db(
            tmp_path,
            [("memory-1", "[0.1, 0.2, 0.3]", "BAAI/bge-small-en-v1.5")],
        )
    )

    assert status["state"] == "active"
    assert status["activity_evidence"] == "persisted_matching_vectors"
    assert status["runtime_scope"] == "current_process"
    assert status["observed_model"] == "BAAI/bge-small-en-v1.5"
    assert status["observed_dimension"] == 3
    assert status["coverage"]["persisted"] == {
        "status": "complete",
        "total_vectors": 1,
        "scanned_vectors": 1,
        "matching_model_vectors": 1,
        "matching_dimension_vectors": 1,
        "scan_limited": False,
    }
    for unprovable in (
        "pending_vectors",
        "failed_vectors",
        "model_revision",
        "distance_metric",
        "index_last_updated",
    ):
        assert status[unprovable] is None


@pytest.mark.parametrize(
    "embedding_json",
    [
        "[0, -1, 2.5]",
        "[1e-3, 2E2, -3.0]",
    ],
)
def test_finite_numeric_scalar_vectors_remain_valid(tmp_path, embedding_json):
    status = _inspect(
        _embedding_db(
            tmp_path,
            [("memory-1", embedding_json, "BAAI/bge-small-en-v1.5")],
        )
    )

    persisted = status["coverage"]["persisted"]
    assert status["state"] == "active"
    assert persisted["status"] == "complete"
    assert persisted["matching_model_vectors"] == 1
    assert persisted["matching_dimension_vectors"] == 1


@pytest.mark.parametrize(
    ("rows", "scan_limit", "coverage_state", "state"),
    [
        (
            [
                ("one", "[1, 2, 3]", "BAAI/bge-small-en-v1.5"),
                ("two", "[1, 2, 3]", "stale/model"),
            ],
            3,
            "partial",
            "active",
        ),
        (
            [("one", "[1, 2, 3]", "stale/model")],
            3,
            "model_mismatch",
            "available",
        ),
        (
            [("one", "[1, 2]", "BAAI/bge-small-en-v1.5")],
            3,
            "dimension_mismatch",
            "active",
        ),
        (
            [("one", "not-json", "BAAI/bge-small-en-v1.5")],
            3,
            "unknown",
            "unknown",
        ),
        (
            [
                ("one", "[1, 2, 3]", "BAAI/bge-small-en-v1.5"),
                ("two", "[1, 2, 3]", "BAAI/bge-small-en-v1.5"),
                ("three", "[1, 2, 3]", "BAAI/bge-small-en-v1.5"),
                ("four", "[1, 2, 3]", "BAAI/bge-small-en-v1.5"),
            ],
            3,
            "scan_limited",
            "active",
        ),
    ],
)
def test_persisted_coverage_states(tmp_path, rows, scan_limit, coverage_state, state):
    status = _inspect(_embedding_db(tmp_path, rows), scan_limit=scan_limit)

    assert status["coverage"]["persisted"]["status"] == coverage_state
    assert status["state"] == state
    if coverage_state == "scan_limited":
        assert status["coverage"]["persisted"]["total_vectors"] is None
        assert status["coverage"]["persisted"]["scanned_vectors"] == scan_limit


def test_matching_vector_beyond_bounded_sample_is_active_evidence(tmp_path):
    scan_limit = 3
    stale_rows = [
        (f"stale-{index}", "[1, 2, 3]", "stale/model") for index in range(scan_limit)
    ]
    matching_row = ("matching", "[1, 2, 3]", "BAAI/bge-small-en-v1.5")

    status = _inspect(
        _embedding_db(tmp_path, [*stale_rows, matching_row]),
        scan_limit=scan_limit,
    )

    assert status["state"] == "active"
    assert status["activity_evidence"] == "persisted_matching_vectors"
    assert status["coverage"]["persisted"]["status"] == "scan_limited"
    assert status["coverage"]["persisted"]["total_vectors"] is None
    assert status["coverage"]["persisted"]["scanned_vectors"] == scan_limit
    assert status["coverage"]["persisted"]["matching_model_vectors"] == 1
    assert status["coverage"]["persisted"]["matching_dimension_vectors"] == 1


@pytest.mark.parametrize(
    "embedding_json",
    [
        "not-json",
        "[]",
        '[1, 2, "3"]',
        "[1, 2, null]",
        "[1, 2, [3]]",
        "[1, 2, true]",
        "[1, 2, NaN]",
        "[1, 2, Infinity]",
        "[1, 2, -Infinity]",
        "[1, 2, 1e999]",
    ],
)
def test_invalid_matching_sentinel_is_not_active_evidence(tmp_path, embedding_json):
    scan_limit = 3
    stale_rows = [
        (f"stale-{index}", "[1, 2, 3]", "stale/model") for index in range(scan_limit)
    ]
    invalid_matching_row = (
        "matching",
        embedding_json,
        "BAAI/bge-small-en-v1.5",
    )

    status = _inspect(
        _embedding_db(tmp_path, [*stale_rows, invalid_matching_row]),
        scan_limit=scan_limit,
    )

    persisted = status["coverage"]["persisted"]
    assert status["state"] == "unknown"
    assert status["activity_evidence"] is None
    assert persisted["status"] == "scan_limited"
    assert persisted["total_vectors"] is None
    assert persisted["scanned_vectors"] == scan_limit
    assert persisted["matching_model_vectors"] == 0
    assert persisted["matching_dimension_vectors"] == 0


@pytest.mark.parametrize(
    "embedding_json",
    [
        "not-json",
        "[]",
        '[1, 2, "3"]',
        "[1, 2, null]",
        "[1, 2, [3]]",
        "[1, 2, false]",
        "[1, 2, NaN]",
        "[1, 2, Infinity]",
        "[1, 2, -Infinity]",
        "[1, 2, -1e999]",
    ],
)
def test_invalid_matching_sample_is_not_active_evidence_under_truncation(
    tmp_path, embedding_json
):
    scan_limit = 3
    rows = [
        ("invalid-match", embedding_json, "BAAI/bge-small-en-v1.5"),
        ("stale-one", "[1, 2, 3]", "stale/model"),
        ("stale-two", "[1, 2, 3]", "stale/model"),
        ("overflow", "[1, 2, 3]", "another/model"),
    ]

    status = _inspect(_embedding_db(tmp_path, rows), scan_limit=scan_limit)

    persisted = status["coverage"]["persisted"]
    assert status["state"] == "unknown"
    assert status["activity_evidence"] is None
    assert persisted["status"] == "scan_limited"
    assert persisted["total_vectors"] is None
    assert persisted["scanned_vectors"] == scan_limit
    assert persisted["matching_model_vectors"] == 0
    assert persisted["matching_dimension_vectors"] == 0


@pytest.mark.parametrize("first_embedding", ["[1, 2]", "not-json", "[]"])
def test_valid_matching_sentinel_is_counted_after_nonmatching_sample_evidence(
    tmp_path, first_embedding
):
    scan_limit = 3
    rows = [
        ("sample-match", first_embedding, "BAAI/bge-small-en-v1.5"),
        ("stale-one", "[1, 2, 3]", "stale/model"),
        ("stale-two", "[1, 2, 3]", "stale/model"),
        ("overflow-match", "[1, 2, 3]", "BAAI/bge-small-en-v1.5"),
    ]

    status = _inspect(_embedding_db(tmp_path, rows), scan_limit=scan_limit)

    persisted = status["coverage"]["persisted"]
    assert status["state"] == "active"
    assert status["activity_evidence"] == "persisted_matching_vectors"
    assert persisted["status"] == "scan_limited"
    assert persisted["total_vectors"] is None
    assert persisted["scanned_vectors"] == scan_limit
    assert persisted["matching_model_vectors"] == (
        2 if first_embedding == "[1, 2]" else 1
    )
    assert persisted["matching_dimension_vectors"] == 1


def test_unknown_schema_metadata_does_not_invent_counts(tmp_path):
    status = _inspect(
        _embedding_db(
            tmp_path,
            ddl="CREATE TABLE memory_embeddings (memory_id TEXT PRIMARY KEY)",
        )
    )

    persisted = status["coverage"]["persisted"]
    assert status["state"] == "unknown"
    assert persisted["status"] == "unknown"
    assert persisted["total_vectors"] is None
    assert persisted["matching_model_vectors"] is None
    assert persisted["matching_dimension_vectors"] is None


def test_unrelated_only_database_reports_no_persisted_vectors(tmp_path):
    path = tmp_path / "unrelated.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE unrelated (id TEXT PRIMARY KEY)")
    conn.commit()
    conn.close()

    status = _inspect(path)

    assert status["coverage"]["persisted"] == {
        "status": "no_vectors",
        "total_vectors": 0,
        "scanned_vectors": 0,
        "matching_model_vectors": 0,
        "matching_dimension_vectors": 0,
        "scan_limited": False,
    }


@pytest.mark.parametrize("denied_stage", ["catalog", "columns", "persisted"])
def test_sqlite_authorizer_errors_propagate_to_persisted_coverage(
    tmp_path, denied_stage
):
    path = _embedding_db(
        tmp_path,
        [("memory-1", "[0.1, 0.2, 0.3]", "BAAI/bge-small-en-v1.5")],
    )
    conn = open_readonly_doctor_db(path)

    def authorize(action, arg1, _arg2, _database, _source):
        denied = (
            (
                denied_stage == "catalog"
                and action == sqlite3.SQLITE_READ
                and arg1 == "sqlite_master"
            )
            or (
                denied_stage == "columns"
                and action == sqlite3.SQLITE_PRAGMA
                and arg1 == "table_xinfo"
            )
            or (
                denied_stage == "persisted"
                and action == sqlite3.SQLITE_READ
                and arg1 == "memory_embeddings"
            )
        )
        return sqlite3.SQLITE_DENY if denied else sqlite3.SQLITE_OK

    conn.set_authorizer(authorize)
    try:
        status = EmbeddingsStatusAdapter(conn, runtime=_runtime()).inspect().metrics
    finally:
        conn.close()

    assert status["state"] == "unknown"
    assert status["coverage"]["persisted"]["status"] == "unknown"
    assert status["coverage"]["persisted"]["error_class"] == "database_error"


def test_api_route_is_distinct_and_not_probed(tmp_path):
    status = _inspect(
        _embedding_db(tmp_path),
        runtime=_runtime(
            backend="openai_compatible_api",
            backend_available=None,
        ),
    )

    assert status["backend"] == "openai_compatible_api"
    assert status["backend_available"] is None
    assert status["state"] == "unknown"
    assert status["activity_evidence"] is None


def test_runtime_missing_dependency_and_import_error_are_non_throwing(
    tmp_path, monkeypatch
):
    from mnemosyne.core import embeddings

    monkeypatch.setattr(embeddings, "TextEmbedding", None)
    monkeypatch.setattr(embeddings, "_is_fastembed_available", lambda: False)
    missing = doctor._embedding_runtime_status()
    assert missing["backend"] == "fastembed_local"
    assert missing["backend_available"] is False
    assert missing["fastembed_installed"] is False

    real_import = builtins.__import__

    def reject_embeddings_import(name, *args, **kwargs):
        if name == "mnemosyne.core":
            raise ImportError("simulated import failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_embeddings_import)
    assert doctor._embedding_runtime_status() == {"error_class": "runtime_error"}

    status = _inspect(_embedding_db(tmp_path), runtime={"error_class": "runtime_error"})
    assert status["state"] == "unknown"
    assert status["configured"] is None


def test_payload_redacts_untrusted_model_metadata_from_both_renderers(tmp_path):
    raw_secret = "password=doctor-private-secret"  # nosec - privacy regression fixture
    report = DoctorReport(
        bank_name="work",
        embeddings={
            **EmbeddingsStatusAdapter(
                None,
                runtime=_runtime(
                    configured_model=raw_secret,
                    configured_model_raw=raw_secret,
                ),
            )
            .inspect()
            .metrics,
            "observed_model": raw_secret,
            "fastembed_version": raw_secret,
        },
    )

    payload = doctor_report_payload(report)
    json_text = render_doctor_json(payload)
    human_text = render_doctor_markdown(payload)

    assert payload["embeddings"]["configured_model"] is None
    assert payload["embeddings"]["observed_model"] is None
    assert payload["embeddings"]["fastembed_version"] is None
    assert raw_secret not in json_text
    assert raw_secret not in human_text


@pytest.mark.parametrize(
    "unsafe_version",
    [
        "sk-" + "a" * 20,
        "ghp_" + "a" * 20,
        "xoxb-" + "a" * 20,
        "../private/fastembed",
        "/private/fastembed",
        r"C:\\private\\fastembed",
    ],
)
def test_payload_redacts_secret_and_path_shaped_fastembed_versions(unsafe_version):
    report = DoctorReport(
        bank_name="work",
        embeddings={
            **EmbeddingsStatusAdapter(None, runtime=_runtime()).inspect().metrics,
            "fastembed_version": unsafe_version,
        },
    )

    payload = doctor_report_payload(report)
    json_text = render_doctor_json(payload)
    human_text = render_doctor_markdown(payload)

    assert payload["embeddings"]["fastembed_version"] is None
    assert unsafe_version not in json_text
    assert unsafe_version not in human_text


@pytest.mark.parametrize(
    "safe_version",
    ["0.7.3", "1.2.3rc1", "2.0.0.post1", "3.1.0+cpu", "2026.09.dev2"],
)
def test_payload_preserves_ordinary_fastembed_versions(safe_version):
    report = DoctorReport(
        bank_name="work",
        embeddings={
            **EmbeddingsStatusAdapter(None, runtime=_runtime()).inspect().metrics,
            "fastembed_version": safe_version,
        },
    )

    payload = doctor_report_payload(report)
    json_text = render_doctor_json(payload)
    human_text = render_doctor_markdown(payload)

    assert payload["embeddings"]["fastembed_version"] == safe_version
    assert safe_version in json_text
    assert safe_version in human_text


@pytest.mark.parametrize(
    "unsafe_model",
    [
        "sk-" + "a" * 20,
        "github_pat_" + "a" * 22,
        "AKIA" + "A" * 16,
        "pk-" + "a" * 20,
        "rk-" + "a" * 20,
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJkb2N0b3IifQ.signature",
        "../private/model",
        "org/../private-model",
        "relative/private/model",
        "relative/private/model.onnx",
        "/absolute/private/model",
        "C:/absolute/private/model",
    ],
)
def test_payload_redacts_secret_and_path_shaped_model_metadata(unsafe_model, tmp_path):
    report = DoctorReport(
        bank_name="work",
        embeddings={
            **EmbeddingsStatusAdapter(None, runtime=_runtime()).inspect().metrics,
            "configured_model": unsafe_model,
            "observed_model": unsafe_model,
        },
    )

    payload = doctor_report_payload(report)
    json_text = render_doctor_json(payload)
    human_text = render_doctor_markdown(payload)

    assert payload["embeddings"]["configured_model"] is None
    assert payload["embeddings"]["observed_model"] is None
    assert unsafe_model not in json_text
    assert unsafe_model not in human_text


@pytest.mark.parametrize(
    "safe_model",
    [
        "BAAI/bge-small-en-v1.5",
        "text-embedding-3-small",
        "nomic-ai/nomic-embed-text-v1.5",
        "pk-embedding-model",
        "rk-embedding-model",
    ],
)
def test_payload_preserves_ordinary_safe_model_identifiers(safe_model):
    report = DoctorReport(
        bank_name="work",
        embeddings={
            **EmbeddingsStatusAdapter(None, runtime=_runtime()).inspect().metrics,
            "configured_model": safe_model,
            "observed_model": safe_model,
        },
    )

    payload = doctor_report_payload(report)

    assert payload["embeddings"]["configured_model"] == safe_model
    assert payload["embeddings"]["observed_model"] == safe_model


def test_json_and_human_output_render_same_canonical_embeddings_payload(tmp_path):
    report = DoctorReport(
        bank_name="work",
        vector_coverage={
            "working": {"status": "complete", "active_source_rows": 1},
            "episodic": {"status": "no_vectors", "source_rows": 0},
        },
        embeddings=EmbeddingsStatusAdapter(None, runtime=_runtime(disabled=True))
        .inspect()
        .metrics,
    )
    payload = doctor_report_payload(report)

    json_payload = json.loads(render_doctor_json(payload))["embeddings"]
    human = render_doctor_markdown(payload)

    assert "## Embeddings" in human
    for key, value in json_payload.items():
        compact = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        assert f"- {key}: `{compact}`" in human


@pytest.mark.parametrize("state", ["unknown", "unavailable"])
@pytest.mark.parametrize("persisted_status", ["unknown", "scan_limited"])
def test_degradation_notes_include_embedding_state_and_persisted_coverage(
    state, persisted_status
):
    payload = {
        "embeddings": {
            "state": state,
            "coverage": {
                "working": {"status": "unknown"},
                "persisted": {"status": persisted_status},
            },
        }
    }

    assert doctor._degradation_notes(payload) == [
        f"embeddings.state: `{state}`",
        f"embeddings.coverage.persisted: `{persisted_status}`",
    ]


def test_embeddings_payload_is_additive_and_preserves_existing_report_keys():
    report = DoctorReport(bank_name="work")
    before = report.to_dict()
    payload = doctor_report_payload(report)

    assert set(payload) == set(before)
    assert payload["bank_name"] == "work"
    assert payload["execution"] == {
        "read_only": True,
        "query_only": True,
        "dry_run": True,
    }
    assert payload["embeddings"]["runtime_scope"] == "current_process"


def test_embeddings_field_preserves_original_doctor_report_positional_order():
    original_fields = [
        "bank_name",
        "database_identity",
        "findings",
        "repair_candidates",
        "schema_fingerprint",
        "runtime_diagnostics",
        "sqlite_health",
        "reference_contracts",
        "vector_coverage",
        "hygiene_summary",
        "execution",
    ]
    values = ["work", *({"position": index} for index in range(1, 11))]

    report = DoctorReport(*values)

    assert [item.name for item in fields(DoctorReport)] == [
        *original_fields,
        "embeddings",
    ]
    assert [getattr(report, name) for name in original_fields] == values
    assert report.embeddings == {}


@pytest.mark.parametrize(
    ("overrides", "expected_state"),
    [
        ({"backend_available": False}, "unavailable"),
        (
            {
                "backend": "fastembed_local",
                "backend_available": True,
                "fastembed_installed": False,
            },
            "unknown",
        ),
        (
            {
                "coverage": {
                    "persisted": {
                        "status": "no_vectors",
                        "total_vectors": 0,
                        "scanned_vectors": 0,
                        "matching_model_vectors": 0,
                        "matching_dimension_vectors": 0,
                        "scan_limited": False,
                    }
                }
            },
            "available",
        ),
        (
            {
                "coverage": {
                    "persisted": {
                        "status": "complete",
                        "total_vectors": 1,
                        "scanned_vectors": 1,
                        "matching_model_vectors": 1,
                        "matching_dimension_vectors": 1,
                        "scan_limited": False,
                        "error_class": "sqlite_error",
                    }
                }
            },
            "unknown",
        ),
        (
            {
                "coverage": {
                    "persisted": {
                        "status": "complete",
                        "total_vectors": 1,
                        "scanned_vectors": 1,
                        "matching_model_vectors": 1,
                        "matching_dimension_vectors": 1,
                        "scan_limited": False,
                        "columns_truncated": True,
                    }
                }
            },
            "unknown",
        ),
    ],
    ids=[
        "backend-unavailable",
        "fastembed-installation-availability-conflict",
        "no-vectors",
        "persisted-error",
        "persisted-columns-truncated",
    ],
)
def test_canonical_payload_downgrades_contradictory_active_claims(
    overrides, expected_state
):
    claimed_active = EmbeddingsStatusAdapter(None, runtime=_runtime()).inspect().metrics
    claimed_active.update(
        {
            "state": "active",
            "activity_evidence": "persisted_matching_vectors",
            "coverage": {
                "persisted": {
                    "status": "complete",
                    "total_vectors": 1,
                    "scanned_vectors": 1,
                    "matching_model_vectors": 1,
                    "matching_dimension_vectors": 1,
                    "scan_limited": False,
                }
            },
            **overrides,
        }
    )

    payload = doctor_report_payload(
        DoctorReport(bank_name="work", embeddings=claimed_active)
    )

    assert payload["embeddings"]["state"] == expected_state
    assert payload["embeddings"]["activity_evidence"] is None


@pytest.mark.parametrize(
    "persisted",
    [
        {
            "status": "no_vectors",
            "total_vectors": 0,
            "scanned_vectors": 0,
            "matching_model_vectors": 1,
            "matching_dimension_vectors": 1,
            "scan_limited": False,
        },
        {
            "status": "model_mismatch",
            "total_vectors": 1,
            "scanned_vectors": 1,
            "matching_model_vectors": 1,
            "matching_dimension_vectors": 1,
            "scan_limited": False,
        },
        {
            "status": "complete",
            "total_vectors": 0,
            "scanned_vectors": 1,
            "matching_model_vectors": 1,
            "matching_dimension_vectors": 1,
            "scan_limited": False,
        },
        {
            "status": "complete",
            "total_vectors": 1,
            "scanned_vectors": 0,
            "matching_model_vectors": 1,
            "matching_dimension_vectors": 1,
            "scan_limited": False,
        },
    ],
    ids=["no-vectors", "model-mismatch", "zero-total", "zero-scanned"],
)
def test_canonical_payload_rejects_impossible_positive_matching_coverage(persisted):
    claimed_active = EmbeddingsStatusAdapter(None, runtime=_runtime()).inspect().metrics
    claimed_active.update(
        {
            "state": "active",
            "activity_evidence": "persisted_matching_vectors",
            "coverage": {"persisted": persisted},
        }
    )

    payload = doctor_report_payload(
        DoctorReport(bank_name="work", embeddings=claimed_active)
    )

    assert payload["embeddings"]["state"] == "unknown"
    assert payload["embeddings"]["activity_evidence"] is None


def test_build_report_is_read_only_and_does_not_construct_or_call_embedding_routes(
    tmp_path, monkeypatch
):
    path = _embedding_db(
        tmp_path,
        [("memory-1", "[0.1, 0.2, 0.3]", "BAAI/bge-small-en-v1.5")],
    )
    from mnemosyne.core import embeddings

    class ForbiddenTextEmbedding:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("Doctor must not construct TextEmbedding")

    monkeypatch.setattr(embeddings, "TextEmbedding", ForbiddenTextEmbedding)
    monkeypatch.setattr(
        embeddings.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail(
            "Doctor must not call an embedding endpoint"
        ),
    )
    before_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    before_files = sorted(item.name for item in tmp_path.iterdir())

    report = build_doctor_report("work", path)

    assert report.execution == {"read_only": True, "query_only": True, "dry_run": True}
    assert set(report.embeddings["coverage"]) == {"persisted", "working", "episodic"}
    assert report.embeddings["coverage"]["working"] == report.vector_coverage["working"]
    assert (
        report.embeddings["coverage"]["episodic"] == report.vector_coverage["episodic"]
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before_digest
    assert sorted(item.name for item in tmp_path.iterdir()) == before_files
    assert not Path(f"{path}-wal").exists()
    assert not Path(f"{path}-shm").exists()
