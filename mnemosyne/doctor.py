"""Read-only, content-safe foundations for future Mnemosyne doctor reports.

This module deliberately contains no CLI registration, database repair, or
``Mnemosyne`` construction.  It only models report data and opens existing
SQLite databases with read-only safeguards for inspection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import tempfile
from typing import Any

from mnemosyne.core.filters import SECRET_LABELED_PATTERNS


STATUS_OK = "ok"
STATUS_WARNING = "warning"
STATUS_ERROR = "error"
STATUS_UNKNOWN = "unknown"
STATUS_PRESENT_BUT_UNLOADABLE = "present_but_unloadable"

SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_ERROR = "error"
SEVERITY_CRITICAL = "critical"

_VEC0_VIRTUAL_TABLE = re.compile(
    r"^\s*CREATE\s+VIRTUAL\s+TABLE\b.*\bUSING\s+vec0\b", re.IGNORECASE | re.DOTALL
)
_VEC0_UNAVAILABLE_ERROR = re.compile(r"\bno such module:\s*vec0\b", re.IGNORECASE)
_SAFE_DETAIL_OPERATIONS = frozenset({"schema_metadata", "table_introspection"})
_SAFE_DETAIL_ERROR_CLASSES = frozenset(
    {"sqlite_error", "operational_error", "database_error"}
)
_ERROR_CLASS_ALIASES = {
    "error": "sqlite_error",
    "sqlite_error": "sqlite_error",
    "operationalerror": "operational_error",
    "operational_error": "operational_error",
    "databaseerror": "database_error",
    "database_error": "database_error",
}
_PREVIEW_SECRET_ASSIGNMENT = re.compile(
    r"\b(password|passphrase|secret|token|api[_-]?key|authorization)\b"
    r"\s*([:=])\s*(?:bearer\s+)?[^\s,;]+",
    re.IGNORECASE,
)
_PREVIEW_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PREVIEW_PHONE = re.compile(r"\b(?:\+?\d[\d .()-]{7,}\d)\b")
_PREVIEW_STANDALONE_TOKEN = re.compile(
    r"\b(?:sk|ghp|github_pat|xox[bap]|AIza)[-_A-Za-z0-9]{16,}\b"
)
_PREVIEW_CANONICAL_SECRET_PATTERNS = tuple(
    (label, re.compile(pattern, re.IGNORECASE))
    for label, pattern in SECRET_LABELED_PATTERNS
)
_RUNTIME_ABSOLUTE_PATH = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:~[\\/]|(?:[A-Za-z]:)?[\\/])[^\s`<>\"']+"
)


class _SQLiteVecExtensionDisableError(RuntimeError):
    """The connection cannot safely continue after extension loading was enabled."""


# Adapter SQL is deliberately limited to these known contracts. Catalog
# metadata only proves that one of these names exists; it never turns an
# arbitrary application table into a doctor query target.
DEFAULT_SCAN_LIMIT = 200
_DOCTOR_TABLE_COLUMNS: dict[str, frozenset[str]] = {
    "working_memory": frozenset({"id", "valid_until", "superseded_by"}),
    "memories": frozenset({"id"}),
    "episodic_memory": frozenset({"id", "binary_vector"}),
    "memory_embeddings": frozenset({"memory_id"}),
    "vec_working": frozenset(),
    "vec_episodes": frozenset(),
    "graph_edges": frozenset({"source", "target"}),
    "canonical_facts": frozenset({"owner_id", "category", "name", "valid_until"}),
    "triples": frozenset({"valid_from", "valid_until"}),
}
_DOCTOR_TABLES = tuple(_DOCTOR_TABLE_COLUMNS)


@dataclass
class Finding:
    """A content-safe result from one diagnostic check.

    Details are deliberately restricted to safe diagnostic enums.  Free-form
    strings and arbitrary values are excluded so report producers cannot leak
    memory content, exception text, or non-JSON values through this field.
    """

    code: str
    status: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.details = _sanitize_finding_details(self.details)


@dataclass
class RepairCandidate:
    """A future, explicit repair proposal; this model never performs a repair."""

    id: str
    description: str
    bank_name: str | None = None
    finding_codes: list[str] = field(default_factory=list)
    requires_explicit_confirmation: bool = True


@dataclass
class ColumnFingerprint:
    """Schema-only metadata for one table column."""

    name: str
    declared_type: str = ""
    not_null: bool = False
    primary_key_position: int = 0


@dataclass
class TableFingerprint:
    """Schema-only metadata for one SQLite table or virtual table."""

    name: str
    object_type: str = "table"
    status: str = STATUS_OK
    columns: list[ColumnFingerprint] = field(default_factory=list)
    foreign_keys: list[dict[str, Any]] = field(default_factory=list)
    detail: str = ""
    error_class: str = ""
    columns_truncated: bool = False
    foreign_keys_truncated: bool = False


@dataclass
class TriggerFingerprint:
    """Content-free identity for one trigger that can change repair semantics."""

    name: str
    table_name: str
    sql_digest: str


@dataclass
class SchemaFingerprint:
    """Structured schema metadata used to gate future repair decisions."""

    tables: list[TableFingerprint] = field(default_factory=list)
    status: str = STATUS_OK
    detail: str = ""
    error_class: str = ""
    tables_truncated: bool = False
    triggers: list[TriggerFingerprint] = field(default_factory=list)
    triggers_truncated: bool = False


@dataclass
class DoctorReport:
    """Serializable, content-safe report envelope for a selected memory bank."""

    bank_name: str
    # st_dev/st_ino authorize this report for one on-disk database, not merely
    # another same-schema database selected with ``--db``.
    database_identity: dict[str, int] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    repair_candidates: list[RepairCandidate] = field(default_factory=list)
    schema_fingerprint: SchemaFingerprint = field(default_factory=SchemaFingerprint)
    runtime_diagnostics: dict[str, Any] = field(default_factory=dict)
    sqlite_health: dict[str, Any] = field(default_factory=dict)
    reference_contracts: dict[str, Any] = field(default_factory=dict)
    vector_coverage: dict[str, Any] = field(default_factory=dict)
    hygiene_summary: dict[str, Any] = field(default_factory=dict)
    execution: dict[str, bool] = field(
        default_factory=lambda: {"read_only": True, "query_only": True, "dry_run": True}
    )

    def to_dict(self) -> dict[str, Any]:
        """Return plain JSON-compatible containers without executing any action."""

        return asdict(self)


def _database_identity(db_path: str | Path) -> dict[str, int]:
    """Return only the stable filesystem identity used by repair authorization."""

    info = os.stat(Path(db_path), follow_symlinks=True)
    if not stat.S_ISREG(info.st_mode):
        raise OSError("database is not a regular file")
    return {"st_dev": int(info.st_dev), "st_ino": int(info.st_ino)}


def open_readonly_doctor_db(db_path: str | Path) -> sqlite3.Connection:
    """Open an existing SQLite database with defense-in-depth read-only settings.

    The URI ``mode=ro`` prevents filesystem writes and creation of a missing
    database.  ``query_only`` independently rejects write SQL, while foreign
    key enforcement makes schema-oriented reads reflect normal SQLite
    constraint behavior.  The caller owns and must close the connection.
    """

    uri = f"{Path(db_path).absolute().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _load_optional_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """Load the bundled sqlite-vec module into an already read-only connection.

    The Doctor must remain usable when the optional dependency is absent or
    incompatible with the host SQLite build.  In that case, its existing
    ``present_but_unloadable`` capability report remains the safe fallback.
    """

    extension_loading_enabled = False
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        extension_loading_enabled = True
        sqlite_vec.load(conn)
    except Exception:
        return False
    finally:
        if extension_loading_enabled:
            try:
                conn.enable_load_extension(False)
            except Exception as error:
                raise _SQLiteVecExtensionDisableError from error
    return True


def safe_preview(value: Any, max_length: int = 120) -> str:
    """Redact likely secrets and PII in bounded internal hygiene previews.

    Report artifacts apply the stricter ``_content_free_hygiene_candidates``
    boundary and never serialize memory text. Structured values and binary
    payloads are not rendered here either.
    """

    if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    if isinstance(value, (bytes, bytearray, memoryview)):
        text = "[binary data redacted]"
    elif isinstance(value, str):
        text = value
    else:
        text = "[structured value redacted]"

    # Keep every regex pass bounded even when callers hand Doctor a very large
    # string. This remains larger than the final preview so redaction happens
    # before truncation for values that cross the final preview boundary.
    text = text[: max_length * 4]
    text = _PREVIEW_SECRET_ASSIGNMENT.sub(r"\1\2<redacted>", text)
    text = _PREVIEW_STANDALONE_TOKEN.sub("<redacted>", text)
    for label, pattern in _PREVIEW_CANONICAL_SECRET_PATTERNS:
        if label in {"env_secret_assignment", "private_key_block"}:
            # The canonical env matcher intentionally identifies only the
            # assignment prefix, while a private-key match identifies only
            # its header.  Redact their values/bodies too, before truncation.
            for match in reversed(tuple(pattern.finditer(text))):
                end = len(text) if label == "private_key_block" else text.find("\n", match.end())
                if end < 0:
                    end = len(text)
                text = text[:match.start()] + "<redacted>" + text[end:]
        else:
            text = pattern.sub("<redacted>", text)
    text = _PREVIEW_EMAIL.sub("<redacted-email>", text)
    text = _PREVIEW_PHONE.sub("<redacted-phone>", text)
    if len(text) <= max_length:
        return text
    return text[: max_length - 1] + "…" if max_length > 1 else "…"


def inspect_schema_fingerprint(
    conn: sqlite3.Connection, scan_limit: int = DEFAULT_SCAN_LIMIT
) -> SchemaFingerprint:
    """Inspect bounded table metadata without reading memory rows or contents.

    A virtual table can be listed in SQLite metadata even when its extension is
    unavailable in the current Python runtime.  In that case, an
    ``OperationalError`` during table-specific introspection is represented as
    ``present_but_unloadable`` rather than being interpreted as corruption.
    """

    scan_limit = _validate_scan_limit(scan_limit)
    try:
        cursor = conn.execute(
            "SELECT name, type, sql FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name LIMIT ?",
            (scan_limit + 1,),
        )
        table_rows, tables_truncated = _bounded_metadata_rows(cursor, scan_limit)
        trigger_cursor = conn.execute(
            "SELECT name, tbl_name, sql FROM sqlite_master "
            "WHERE type = 'trigger' ORDER BY name LIMIT ?",
            (scan_limit + 1,),
        )
        trigger_rows, triggers_truncated = _bounded_metadata_rows(trigger_cursor, scan_limit)
    except sqlite3.Error as error:
        return SchemaFingerprint(
            status=STATUS_UNKNOWN,
            detail="SQLite schema metadata could not be inspected by this runtime.",
            error_class=_safe_sqlite_error_class(error),
        )

    tables: list[TableFingerprint] = []
    for row in table_rows:
        table_name = str(row[0])
        table = TableFingerprint(name=table_name, object_type=str(row[1]))
        table_sql = str(row[2] or "")
        quoted_name = _quote_identifier(table_name)
        try:
            column_rows, table.columns_truncated = _bounded_metadata_rows(
                conn.execute(f"PRAGMA table_xinfo({quoted_name})"), scan_limit
            )
            table.columns = [
                ColumnFingerprint(
                    name=str(column[1]),
                    declared_type=str(column[2] or ""),
                    not_null=bool(column[3]),
                    primary_key_position=int(column[5]),
                )
                for column in column_rows
            ]
            foreign_key_rows, table.foreign_keys_truncated = _bounded_metadata_rows(
                conn.execute(f"PRAGMA foreign_key_list({quoted_name})"), scan_limit
            )
            table.foreign_keys = [
                {
                    "id": int(foreign_key[0]),
                    "seq": int(foreign_key[1]),
                    "table": str(foreign_key[2]),
                    "from": str(foreign_key[3]),
                    "to": str(foreign_key[4]),
                    "on_update": str(foreign_key[5]),
                    "on_delete": str(foreign_key[6]),
                    "match": str(foreign_key[7]),
                }
                for foreign_key in foreign_key_rows
            ]
        except sqlite3.Error as error:
            table.status = (
                STATUS_PRESENT_BUT_UNLOADABLE
                if _is_confirmed_vec0_unloadable(table_sql, error)
                else STATUS_UNKNOWN
            )
            table.columns = []
            table.foreign_keys = []
            table.columns_truncated = False
            table.foreign_keys_truncated = False
            table.detail = (
                "SQLite vec0 virtual table is present but unavailable in this runtime."
                if table.status == STATUS_PRESENT_BUT_UNLOADABLE
                else "SQLite table metadata could not be inspected by this runtime."
            )
            table.error_class = _safe_sqlite_error_class(error)
        tables.append(table)

    triggers = [
        TriggerFingerprint(
            name=str(row[0]),
            table_name=str(row[1]),
            # Trigger bodies can contain literals. A stable digest detects a
            # semantic change without putting arbitrary SQL in a shareable report.
            sql_digest=hashlib.sha256(str(row[2] or "").encode("utf-8")).hexdigest(),
        )
        for row in trigger_rows
    ]
    return SchemaFingerprint(
        tables=tables,
        tables_truncated=tables_truncated,
        triggers=triggers,
        triggers_truncated=triggers_truncated,
    )


def _quote_identifier(identifier: str) -> str:
    """Return a SQLite double-quoted identifier from trusted schema metadata."""

    return '"' + identifier.replace('"', '""') + '"'


def _is_confirmed_vec0_unloadable(table_sql: str, error: sqlite3.Error) -> bool:
    """Return true only for vec0 metadata plus its specific unavailable signal."""

    return (
        isinstance(error, sqlite3.OperationalError)
        and bool(_VEC0_VIRTUAL_TABLE.search(table_sql))
        and bool(_VEC0_UNAVAILABLE_ERROR.search(str(error)))
    )


def _safe_sqlite_error_class(error: sqlite3.Error) -> str:
    """Map SQLite exceptions to bounded labels without retaining exception text."""

    if isinstance(error, sqlite3.OperationalError):
        return "operational_error"
    if isinstance(error, sqlite3.DatabaseError):
        return "database_error"
    return "sqlite_error"


def _sanitize_finding_details(details: Any) -> dict[str, Any]:
    """Keep only the explicit, JSON-safe finding detail schema.

    ``operation`` and ``error_class`` are bounded enums; every other supplied
    field is discarded and counted without preserving its name or value.
    """

    if not isinstance(details, dict):
        return {"redacted_field_count": 1}

    safe_details: dict[str, Any] = {}
    redacted_field_count = 0
    for key, value in details.items():
        if key == "operation" and isinstance(value, str):
            operation = value.strip().lower()
            if operation in _SAFE_DETAIL_OPERATIONS:
                safe_details[key] = operation
                continue
        elif key == "error_class" and isinstance(value, str):
            error_class = _ERROR_CLASS_ALIASES.get(value.strip().lower())
            if error_class in _SAFE_DETAIL_ERROR_CLASSES:
                safe_details[key] = error_class
                continue
        redacted_field_count += 1

    if redacted_field_count:
        safe_details["redacted_field_count"] = redacted_field_count
    return safe_details


@dataclass
class AdapterResult:
    """Bounded, content-safe output shared by read-only doctor adapters."""

    metrics: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    repair_candidates: list[RepairCandidate] = field(default_factory=list)


_RUNTIME_CHECK_NAMES = frozenset(
    {
        "python_version",
        "platform",
        "python_executable",
        "mnemosyne_version",
        "fastembed",
        "sqlite_vec",
        "numpy",
        "huggingface_hub",
        "ctransformers",
        "embeddings_available",
        "embeddings_model",
        "sqlite_vec_available",
        "sqlite_vec_warning",
    }
)
_RUNTIME_STATUSES = frozenset({"OK", "YES", "NO", "MISSING", "OPTIONAL", "ERROR"})


def _safe_runtime_detail(check: str, value: Any) -> str:
    """Keep runtime metadata useful without serializing host-specific paths."""

    if check == "python_executable" and isinstance(value, str):
        # Runtime data can originate on a different OS than the Doctor host.
        # Normalize Windows separators before extracting a portable basename.
        executable_name = value.replace("\\", "/").rsplit("/", 1)[-1]
        return safe_preview(executable_name, max_length=240)
    detail = safe_preview(value, max_length=240)
    return _RUNTIME_ABSOLUTE_PATH.sub("<redacted-path>", detail)


def _sanitize_runtime_diagnostics(runtime: Any) -> dict[str, Any]:
    """Apply the Doctor report boundary to runtime checks from any producer."""

    if not isinstance(runtime, dict):
        return {}
    if not runtime:
        return {}
    if not isinstance(runtime.get("checks"), list):
        return {"status": STATUS_UNKNOWN, "error_class": "runtime_error"}

    status = runtime.get("status")
    if status not in {STATUS_OK, STATUS_WARNING, "unavailable"}:
        status = STATUS_UNKNOWN
    checks = [
        {
            "check": entry["check"],
            "status": entry["status"],
            "detail": _safe_runtime_detail(entry["check"], entry.get("detail", "")),
        }
        for entry in runtime["checks"]
        if isinstance(entry, dict)
        and entry.get("check") in _RUNTIME_CHECK_NAMES
        and entry.get("status") in _RUNTIME_STATUSES
    ]
    return {"status": status, "checks": checks}


class RuntimeDiagnosticsAdapter:
    """Expose pure runtime/dependency checks without constructing Mnemosyne."""

    def inspect(self) -> AdapterResult:
        try:
            from mnemosyne.runtime_diagnostics import collect_runtime_diagnostics

            result = collect_runtime_diagnostics()
        except Exception:
            return AdapterResult(metrics={"status": STATUS_UNKNOWN, "error_class": "runtime_error"})
        if not isinstance(result, dict) or not isinstance(result.get("checks"), list):
            return AdapterResult(metrics={"status": STATUS_UNKNOWN, "error_class": "runtime_error"})
        return AdapterResult(metrics=_sanitize_runtime_diagnostics(result))


class HygieneSummaryAdapter:
    """Use only the bounded, content-safe hygiene doctor view on a supplied DB."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        db_path: str | Path,
        scan_limit: int = DEFAULT_SCAN_LIMIT,
        candidate_limit: int = 20,
    ):
        self.conn = conn
        self.db_path = Path(db_path)
        self.scan_limit = _validate_scan_limit(scan_limit)
        if (
            not isinstance(candidate_limit, int)
            or isinstance(candidate_limit, bool)
            or candidate_limit < 0
        ):
            raise ValueError("candidate_limit must be a non-negative integer")
        self.candidate_limit = candidate_limit

    def inspect(self) -> AdapterResult:
        try:
            from mnemosyne.core.hygiene import doctor_hygiene_summary

            result = doctor_hygiene_summary(
                self.db_path,
                limit=self.scan_limit,
                candidate_limit=self.candidate_limit,
                conn=self.conn,
            )
        except Exception:
            return AdapterResult(metrics={"status": STATUS_UNKNOWN, "error_class": "runtime_error", "candidates": []})
        if not isinstance(result, dict) or result.get("status") not in {STATUS_OK, STATUS_UNKNOWN, "unavailable"}:
            return AdapterResult(metrics={"status": STATUS_UNKNOWN, "error_class": "runtime_error", "candidates": []})
        return AdapterResult(metrics=self._sanitize_summary(result))

    def _sanitize_summary(self, result: dict[str, Any]) -> dict[str, Any]:
        """Keep the hygiene bridge bounded even if its implementation grows."""

        status = result["status"]
        if status != STATUS_OK:
            return {
                "status": status,
                "error_class": result.get("error_class")
                if result.get("error_class") in {"sqlite_error", "runtime_error"}
                else "runtime_error",
                "candidates": [],
            }

        def count(name: str) -> int:
            value = result.get(name, 0)
            return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0

        candidates: list[dict[str, Any]] = []
        raw_candidates = result.get("candidates", [])
        if isinstance(raw_candidates, list):
            for candidate in raw_candidates[: self.candidate_limit]:
                if not isinstance(candidate, dict):
                    continue
                score = candidate.get("noise_score")
                candidates.append(
                    {
                        "table": candidate.get("table")
                        if candidate.get("table") in {"working_memory", "memories", "episodic_memory"}
                        else "unknown",
                        "noise_score": score
                        if isinstance(score, (int, float)) and not isinstance(score, bool) and 0 <= score <= 1
                        else 0.0,
                        "reasons": [
                            safe_preview(reason, max_length=120)
                            for reason in candidate.get("reasons", [])
                            if isinstance(reason, str)
                        ][:20],
                        "secret_flags": [
                            safe_preview(flag, max_length=120)
                            for flag in candidate.get("secret_flags", [])
                            if isinstance(flag, str)
                        ][:20],
                        "suggested_action": candidate.get("suggested_action")
                        if candidate.get("suggested_action") in {"delete", "archive", "keep", "flag"}
                        else "keep",
                        "preview": safe_preview(candidate.get("preview"), max_length=120),
                    }
                )
        return {
            "status": STATUS_OK,
            "total_scanned": count("total_scanned"),
            "total_candidates": count("total_candidates"),
            "with_secrets": count("with_secrets"),
            "limit_per_table": count("limit_per_table"),
            "candidate_limit": self.candidate_limit,
            "candidates": candidates,
        }


def build_doctor_report(
    bank_name: str,
    db_path: str | Path,
    scan_limit: int = DEFAULT_SCAN_LIMIT,
    candidate_limit: int = 20,
) -> DoctorReport:
    """Build a complete report using only runtime checks and a read-only DB.

    The function intentionally does not invoke ``Mnemosyne``, provider setup,
    diagnostics repair, cleanup, reindexing, or embedding work.
    """

    scan_limit = _validate_scan_limit(scan_limit)
    report = DoctorReport(
        bank_name=bank_name,
        runtime_diagnostics=RuntimeDiagnosticsAdapter().inspect().metrics,
    )
    try:
        report.database_identity = _database_identity(db_path)
    except (OSError, ValueError):
        # A report without a regular-file identity remains useful read-only
        # diagnostics, but repair will deliberately reject it.
        report.database_identity = {}
    try:
        conn = open_readonly_doctor_db(db_path)
    except sqlite3.Error:
        unavailable = {"status": "unavailable", "error_class": "sqlite_error"}
        report.sqlite_health = dict(unavailable)
        report.reference_contracts = dict(unavailable)
        report.vector_coverage = dict(unavailable)
        report.hygiene_summary = dict(unavailable, candidates=[])
        return report
    try:
        try:
            _load_optional_sqlite_vec(conn)
        except _SQLiteVecExtensionDisableError:
            unavailable = {"status": "unavailable", "error_class": "sqlite_error"}
            report.sqlite_health = dict(unavailable)
            report.reference_contracts = dict(unavailable)
            report.vector_coverage = dict(unavailable)
            report.hygiene_summary = dict(unavailable, candidates=[])
            return report
        report.schema_fingerprint = inspect_schema_fingerprint(conn, scan_limit=scan_limit)
        sqlite_health = SQLiteHealthAdapter(conn, scan_limit=scan_limit).inspect()
        reference_contracts = ReferenceContractRegistry(conn, scan_limit=scan_limit).inspect()
        vector_coverage = VectorCoverageAdapter(conn, scan_limit=scan_limit).inspect()
        report.sqlite_health = sqlite_health.metrics
        report.reference_contracts = reference_contracts.metrics
        report.vector_coverage = vector_coverage.metrics
        report.findings.extend(sqlite_health.findings)
        report.findings.extend(reference_contracts.findings)
        report.findings.extend(vector_coverage.findings)
        report.repair_candidates.extend(sqlite_health.repair_candidates)
        report.repair_candidates.extend(reference_contracts.repair_candidates)
        report.repair_candidates.extend(vector_coverage.repair_candidates)
        report.hygiene_summary = HygieneSummaryAdapter(
            conn,
            db_path,
            scan_limit=scan_limit,
            candidate_limit=candidate_limit,
        ).inspect().metrics
    finally:
        conn.close()
    return report


def doctor_report_payload(report: DoctorReport, *, include_candidates: bool = False) -> dict[str, Any]:
    """Return the canonical JSON-safe doctor model used by every renderer."""

    payload = report.to_dict()
    payload["runtime_diagnostics"] = _sanitize_runtime_diagnostics(payload.get("runtime_diagnostics"))
    hygiene = payload.get("hygiene_summary")
    if isinstance(hygiene, dict):
        hygiene["candidates"] = _content_free_hygiene_candidates(hygiene.get("candidates"))
    payload["repair_candidates"] = _content_free_repair_candidates(
        payload.get("repair_candidates")
    )
    if not include_candidates:
        payload["repair_candidates"] = []
        if isinstance(hygiene, dict):
            hygiene["candidates"] = []
    return payload


def _content_free_hygiene_candidates(candidates: Any) -> list[dict[str, Any]]:
    """Expose only fixed-schema hygiene metadata in report artifacts.

    The hygiene adapter retains redacted previews for its internal audit
    contract, but report files have a stricter boundary: they must never carry
    memory text, even when that text is not secret.  Do not preserve IDs,
    reasons, flags, or previews here because those fields can be derived from
    arbitrary stored content.
    """

    if not isinstance(candidates, list):
        return []

    safe_candidates: list[dict[str, Any]] = []
    for candidate in candidates[:100]:
        if not isinstance(candidate, dict):
            continue
        score = candidate.get("noise_score")
        reasons = candidate.get("reasons")
        secret_flags = candidate.get("secret_flags")
        safe_candidates.append(
            {
                "candidate_class": "hygiene",
                "table": candidate.get("table")
                if candidate.get("table") in {"working_memory", "memories", "episodic_memory"}
                else "unknown",
                "noise_score": score
                if isinstance(score, (int, float)) and not isinstance(score, bool) and 0 <= score <= 1
                else 0.0,
                "reason_count": min(len(reasons), 20) if isinstance(reasons, list) else 0,
                "secret_flag_count": min(len(secret_flags), 20) if isinstance(secret_flags, list) else 0,
                "suggested_action": candidate.get("suggested_action")
                if candidate.get("suggested_action") in {"delete", "archive", "keep", "flag"}
                else "keep",
            }
        )
    return safe_candidates


def _content_free_repair_candidates(candidates: Any) -> list[dict[str, Any]]:
    """Expose repair proposals without serializing their raw identifying data.

    A future repair proposal may carry an ID, bank, description, or finding
    references.  Those values are useful only to an explicit repair workflow;
    they are neither needed nor safe in a shareable read-only report.  Retain
    only a fixed class and the confirmation requirement.
    """

    if not isinstance(candidates, list):
        return []

    safe_candidates: list[dict[str, Any]] = []
    for candidate in candidates[:100]:
        if not isinstance(candidate, dict):
            continue
        safe_candidates.append(
            {
                "candidate_class": "repair",
                "requires_explicit_confirmation": bool(
                    candidate.get("requires_explicit_confirmation", True)
                ),
            }
        )
    return safe_candidates


def render_doctor_json(payload: dict[str, Any]) -> str:
    """Render the canonical doctor model deterministically as JSON."""

    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def render_doctor_markdown(payload: dict[str, Any]) -> str:
    """Render the canonical, content-safe doctor model deterministically."""

    findings = payload.get("findings", [])
    severity_counts = {
        severity: sum(
            1
            for finding in findings
            if isinstance(finding, dict) and finding.get("severity") == severity
        )
        for severity in (SEVERITY_CRITICAL, SEVERITY_ERROR, SEVERITY_WARNING, SEVERITY_INFO)
    }
    lines = [
        "# Mnemosyne Doctor Report",
        "",
        "**Read-only / query_only / dry-run — no repair, migration, or database write was performed.**",
        "",
        "## Summary",
        "",
        f"- Bank: `{_markdown_scalar(payload.get('bank_name', 'default'))}`",
        "- Findings: " + ", ".join(f"{name}={severity_counts[name]}" for name in severity_counts),
        f"- Repair candidates requiring review: {len(payload.get('repair_candidates', []))}",
        "",
        "## SQLite",
        "",
    ]
    sqlite_health = payload.get("sqlite_health", {})
    if isinstance(sqlite_health, dict):
        for name in ("quick_check", "foreign_key_check", "foreign_key_inventory", "vec0"):
            if name in sqlite_health:
                lines.append(f"- {name}: `{_compact_json(sqlite_health[name])}`")
    else:
        lines.append(f"- status: `{_compact_json(sqlite_health)}`")

    lines.extend(["", "## Runtime", ""])
    runtime_diagnostics = payload.get("runtime_diagnostics", {})
    if isinstance(runtime_diagnostics, dict):
        if "status" in runtime_diagnostics:
            lines.append(f"- status: `{_compact_json(runtime_diagnostics['status'])}`")
        checks = runtime_diagnostics.get("checks", [])
        if isinstance(checks, list):
            for check in checks:
                if not isinstance(check, dict) or not isinstance(check.get("check"), str):
                    continue
                detail = safe_preview(check.get("detail", ""), max_length=240)
                lines.append(
                    f"- {_markdown_scalar(check['check'])}: "
                    f"`{_compact_json({'status': check.get('status'), 'detail': detail})}`"
                )
    else:
        lines.append(f"- status: `{_compact_json(runtime_diagnostics)}`")

    lines.extend(["", "## References", ""])
    _append_metric_lines(lines, payload.get("reference_contracts"))
    lines.extend(["", "## Vector tiers", ""])
    _append_metric_lines(lines, payload.get("vector_coverage"))

    lines.extend(["", "## Hygiene", ""])
    hygiene = payload.get("hygiene_summary", {})
    if isinstance(hygiene, dict):
        for name in ("status", "total_scanned", "total_candidates", "with_secrets", "limit_per_table"):
            if name in hygiene:
                lines.append(f"- {name}: `{_compact_json(hygiene[name])}`")
        candidates = hygiene.get("candidates", [])
        if isinstance(candidates, list) and candidates:
            lines.append("- Redacted candidate samples:")
            for candidate in candidates[:20]:
                lines.append(f"  - `{_compact_json(candidate)}`")
    else:
        lines.append(f"- status: `{_compact_json(hygiene)}`")

    notes = _degradation_notes(payload)
    lines.extend(["", "## Degradations and review", ""])
    if notes:
        lines.extend(f"- {note}" for note in notes[:10])
    else:
        lines.append("- No bounded degradation signal was reported.")
    if _contains_status(payload, STATUS_PRESENT_BUT_UNLOADABLE):
        lines.append(
            "- sqlite-vec tables are present but this runtime cannot load vec0; "
            "review the sqlite-vec runtime installation before treating vector checks as unavailable."
        )
    lines.append("- Review the findings and explicit candidates before any future repair.")
    lines.append("  Repairs must remain individually selected and explicitly confirmed; this report performs none.")

    candidates = payload.get("repair_candidates", [])
    if isinstance(candidates, list) and candidates:
        lines.extend(["", "## Explicit repair candidates", ""])
        for candidate in candidates[:20]:
            lines.append(f"- `{_compact_json(candidate)}`")
    return "\n".join(lines) + "\n"


def _rollback_doctor_artifacts(
    committed: list[Path], originals: dict[Path, bytes | None], temporary_paths: set[Path]
) -> list[BaseException]:
    """Best-effort restore of committed Doctor artifacts in reverse order."""

    rollback_errors: list[BaseException] = []
    for target in reversed(committed):
        try:
            original = originals[target]
            if original is None:
                target.unlink(missing_ok=True)
                continue
            fd, temporary = tempfile.mkstemp(
                prefix=f".{target.name}-restore-", suffix=".tmp", dir=target.parent
            )
            temporary_path = Path(temporary)
            temporary_paths.add(temporary_path)
            with os.fdopen(fd, "wb") as handle:
                handle.write(original)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, target)
        except BaseException as rollback_error:
            # Continue trying every committed target.  The original write
            # error remains the public failure, with rollback trouble
            # attached as diagnostic context rather than masking it.
            rollback_errors.append(rollback_error)
    return rollback_errors


def _write_doctor_artifacts_atomically(targets: list[tuple[Path, str]]) -> None:
    """Stage and replace one or more rendered Doctor artifacts safely."""

    if not targets:
        raise ValueError("At least one Doctor output target is required")
    if len({target.resolve() for target, _ in targets}) != len(targets):
        raise ValueError("JSON and Markdown output paths must be different")
    for target, _ in targets:
        if not target.parent.is_dir():
            raise ValueError(f"Output directory does not exist: {target.parent}")

    temporary_paths: set[Path] = set()
    staged: dict[Path, Path] = {}
    originals: dict[Path, bytes | None] = {}
    committed: list[Path] = []
    try:
        for target, text in targets:
            originals[target] = target.read_bytes() if target.exists() else None
            fd, temporary = tempfile.mkstemp(prefix=f".{target.name}-", suffix=".tmp", dir=target.parent)
            temporary_path = Path(temporary)
            # Register immediately: failures in fdopen/write/flush/fsync must
            # not leak a temp created before the old staged assignment point.
            temporary_paths.add(temporary_path)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            staged[target] = temporary_path
        for target, _ in targets:
            os.replace(staged[target], target)
            committed.append(target)
        for target, _ in targets:
            _fsync_directory(target.parent)
    except BaseException as error:
        rollback_errors = _rollback_doctor_artifacts(committed, originals, temporary_paths)
        if rollback_errors and hasattr(error, "add_note"):
            error.add_note(
                f"Doctor output rollback encountered {len(rollback_errors)} error(s)."
            )
        raise
    finally:
        for temporary in temporary_paths:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                # Cleanup is best effort under OS failures, but attempt every
                # tracked path and never mask the primary write/rollback error.
                pass


def write_doctor_artifact_atomically(*, path: str | Path, text: str) -> None:
    """Atomically replace one rendered Doctor artifact."""

    _write_doctor_artifacts_atomically([(Path(path), text)])


def write_doctor_artifacts_atomically(
    *, json_path: str | Path, json_text: str, markdown_path: str | Path, markdown_text: str
) -> None:
    """Write both artifacts together and roll back a partial second replacement."""

    _write_doctor_artifacts_atomically(
        [(Path(json_path), json_text), (Path(markdown_path), markdown_text)]
    )


def _append_metric_lines(lines: list[str], metrics: Any) -> None:
    if isinstance(metrics, dict):
        for name in sorted(metrics):
            lines.append(f"- {name}: `{_compact_json(metrics[name])}`")
    else:
        lines.append(f"- status: `{_compact_json(metrics)}`")


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _markdown_scalar(value: Any) -> str:
    return safe_preview(value, max_length=120).replace("`", "'")


def _contains_status(value: Any, status: str) -> bool:
    if isinstance(value, dict):
        return value.get("status") == status or any(_contains_status(item, status) for item in value.values())
    if isinstance(value, list):
        return any(_contains_status(item, status) for item in value)
    return False


def _degradation_notes(payload: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    for section in ("sqlite_health", "reference_contracts", "vector_coverage", "hygiene_summary"):
        value = payload.get(section)
        if not isinstance(value, dict):
            continue
        for name in sorted(value):
            metric = value[name]
            status = metric.get("status") if isinstance(metric, dict) else None
            if status in {STATUS_UNKNOWN, "unavailable", STATUS_PRESENT_BUT_UNLOADABLE, "scan_limited"}:
                notes.append(f"{section}.{name}: `{status}`")
    return notes


def _fsync_directory(directory: Path) -> None:
    """Durably sync a completed output replacement's parent directory."""

    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class _CatalogResult:
    tables: dict[str, str] = field(default_factory=dict)
    error_class: str | None = None


@dataclass(frozen=True)
class _ColumnsResult:
    columns: frozenset[str] = frozenset()
    error_class: str | None = None
    truncated: bool = False


@dataclass(frozen=True)
class _BoundedCount:
    value: int | None = None
    truncated: bool = False
    error_class: str | None = None


def _validate_scan_limit(scan_limit: int) -> int:
    if not isinstance(scan_limit, int) or isinstance(scan_limit, bool) or scan_limit < 1:
        raise ValueError("scan_limit must be a positive integer")
    return scan_limit


def _bounded_metadata_rows(cursor: sqlite3.Cursor, scan_limit: int) -> tuple[list[Any], bool]:
    """Read at most a metadata limit plus one sentinel without report growth."""

    rows: list[Any] = []
    for row in cursor:
        if len(rows) == scan_limit:
            return rows, True
        rows.append(row)
    return rows, False


def _catalog(conn: sqlite3.Connection) -> _CatalogResult:
    """Return only registry-approved adapter tables, never arbitrary catalog names."""

    placeholders = ", ".join("?" for _ in _DOCTOR_TABLES)
    try:
        cursor = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table' "
            f"AND name IN ({placeholders})",
            _DOCTOR_TABLES,
        )
        tables: dict[str, str] = {}
        for row in cursor:
            name = str(row[0])
            if name in _DOCTOR_TABLE_COLUMNS:
                tables[name] = str(row[1] or "")
        return _CatalogResult(tables=tables)
    except sqlite3.Error as error:
        return _CatalogResult(error_class=_safe_sqlite_error_class(error))


def _table_columns(
    conn: sqlite3.Connection, table: str, scan_limit: int = DEFAULT_SCAN_LIMIT
) -> _ColumnsResult:
    """Return bounded, registry-approved columns for a fixed doctor table."""

    if table not in _DOCTOR_TABLE_COLUMNS:
        raise ValueError("table is not in the doctor contract registry")
    scan_limit = _validate_scan_limit(scan_limit)
    try:
        rows, truncated = _bounded_metadata_rows(
            conn.execute(f"PRAGMA table_xinfo({_quote_identifier(table)})"), scan_limit
        )
        columns = frozenset(
            str(row[1])
            for row in rows
            if str(row[1]) in _DOCTOR_TABLE_COLUMNS[table]
        )
        return _ColumnsResult(columns=columns, truncated=truncated)
    except sqlite3.Error as error:
        return _ColumnsResult(error_class=_safe_sqlite_error_class(error))


def _bounded_count(conn: sqlite3.Connection, candidate_sql: str, scan_limit: int) -> _BoundedCount:
    """Count at most ``scan_limit`` candidate rows without an unbounded result set."""

    try:
        cursor = conn.execute(f"SELECT 1 FROM ({candidate_sql}) LIMIT ?", (scan_limit + 1,))
        count = 0
        for _ in cursor:
            count += 1
        return _BoundedCount(value=min(count, scan_limit), truncated=count > scan_limit)
    except sqlite3.Error as error:
        return _BoundedCount(error_class=_safe_sqlite_error_class(error))


def _with_counts(status: str, counts: dict[str, _BoundedCount]) -> dict[str, Any]:
    error_class = next((count.error_class for count in counts.values() if count.error_class), None)
    if error_class:
        return {"status": STATUS_UNKNOWN, "error_class": error_class}
    metric: dict[str, Any] = {
        "status": "scan_limited" if any(count.truncated for count in counts.values()) else status
    }
    for name, count in counts.items():
        metric[name] = count.value
        if count.truncated:
            metric[f"{name}_truncated"] = True
    return metric


def _vec_table_status(conn: sqlite3.Connection, table: str, ddl: str | None) -> tuple[str, str | None]:
    """Classify a queryable, confirmed vec0 table; plain names are not vec tables."""

    if ddl is None or not _VEC0_VIRTUAL_TABLE.search(ddl):
        return "not_configured", None
    try:
        conn.execute(f"SELECT 1 FROM {_quote_identifier(table)} LIMIT 0").fetchone()
        return "available", None
    except sqlite3.Error as error:
        if _is_confirmed_vec0_unloadable(ddl, error):
            return STATUS_PRESENT_BUT_UNLOADABLE, _safe_sqlite_error_class(error)
        return STATUS_UNKNOWN, _safe_sqlite_error_class(error)


def _live_memory_tables(
    conn: sqlite3.Connection, catalog: _CatalogResult, scan_limit: int
) -> tuple[list[str], str | None, bool]:
    """Return source tables with the fixed ``id`` contract or uncertainty state."""

    live_tables: list[str] = []
    for table in ("working_memory", "memories", "episodic_memory"):
        if table not in catalog.tables:
            continue
        columns = _table_columns(conn, table, scan_limit)
        if columns.error_class:
            return [], columns.error_class, False
        if "id" in columns.columns:
            live_tables.append(table)
        elif columns.truncated:
            return [], None, True
    return live_tables, None, False


def _live_id_union(live_tables: list[str]) -> str:
    return " UNION ".join(
        f"SELECT id FROM {_quote_identifier(table)} WHERE id IS NOT NULL"
        for table in live_tables
    )


class SQLiteHealthAdapter:
    """Read-only health checks with a bounded diagnostic scan budget."""

    def __init__(self, conn: sqlite3.Connection, scan_limit: int = DEFAULT_SCAN_LIMIT):
        self.conn = conn
        self.scan_limit = _validate_scan_limit(scan_limit)

    def _pragma_rows(self, pragma: str) -> tuple[list[sqlite3.Row], bool, str | None]:
        try:
            rows: list[sqlite3.Row] = []
            cursor = self.conn.execute(pragma)
            for row in cursor:
                if len(rows) == self.scan_limit:
                    return rows, True, None
                rows.append(row)
            return rows, False, None
        except sqlite3.Error as error:
            return [], False, _safe_sqlite_error_class(error)

    def inspect(self) -> AdapterResult:
        catalog = _catalog(self.conn)
        findings: list[Finding] = []
        quick_rows, quick_truncated, quick_error = self._pragma_rows("PRAGMA quick_check")
        if quick_error:
            quick_check: dict[str, Any] = {"status": STATUS_UNKNOWN, "error_class": quick_error}
        else:
            quick_ok = bool(quick_rows) and all(str(row[0]).lower() == "ok" for row in quick_rows)
            quick_check = {
                "status": "truncated" if quick_truncated and quick_ok else "ok" if quick_ok else "failed",
                "capability": "sqlite_integrity_primitive",
                "runtime_cost": "potentially_global",
            }
            if quick_truncated:
                quick_check["truncated"] = True

        fk_rows, fk_truncated, fk_error = self._pragma_rows("PRAGMA foreign_key_check")
        if fk_error:
            foreign_key_check: dict[str, Any] = {"status": STATUS_UNKNOWN, "error_class": fk_error}
        else:
            foreign_key_check = {
                "status": "violations" if fk_rows else "ok",
                "count": len(fk_rows),
            }
            if fk_truncated:
                foreign_key_check["truncated"] = True

        if catalog.error_class:
            inventory: dict[str, Any] | list[dict[str, Any]] = {
                "status": STATUS_UNKNOWN,
                "error_class": catalog.error_class,
            }
            vec0: dict[str, Any] = {"status": STATUS_UNKNOWN, "error_class": catalog.error_class}
        else:
            entries: list[dict[str, Any]] = []
            inventory_error: str | None = None
            for table in sorted(catalog.tables):
                rows, truncated, error_class = self._pragma_rows(
                    f"PRAGMA foreign_key_list({_quote_identifier(table)})"
                )
                if error_class:
                    inventory_error = error_class
                    break
                if rows:
                    entry: dict[str, Any] = {"table": table, "count": len(rows)}
                    if truncated:
                        entry["truncated"] = True
                    entries.append(entry)
            inventory = (
                {"status": STATUS_UNKNOWN, "error_class": inventory_error}
                if inventory_error
                else entries
            )
            vec_tables = [
                (name, ddl)
                for name, ddl in catalog.tables.items()
                if _VEC0_VIRTUAL_TABLE.search(ddl)
            ]
            if not vec_tables:
                vec0 = {"status": "not_configured", "table_count": 0}
            else:
                statuses = [_vec_table_status(self.conn, name, ddl) for name, ddl in vec_tables]
                status_values = [status for status, _ in statuses]
                error_class = next((error for _, error in statuses if error), None)
                vec_status = (
                    STATUS_PRESENT_BUT_UNLOADABLE
                    if STATUS_PRESENT_BUT_UNLOADABLE in status_values
                    else STATUS_UNKNOWN if STATUS_UNKNOWN in status_values else "available"
                )
                vec0 = {"status": vec_status, "table_count": len(vec_tables)}
                if error_class:
                    vec0["error_class"] = error_class
                if vec_status == STATUS_PRESENT_BUT_UNLOADABLE:
                    findings.append(
                        Finding(
                            code="sqlite.vec0_capability",
                            status=STATUS_PRESENT_BUT_UNLOADABLE,
                            severity=SEVERITY_WARNING,
                            message="SQLite vec0 tables are present but unavailable in this runtime.",
                        )
                    )
        return AdapterResult(
            metrics={
                "quick_check": quick_check,
                "foreign_key_check": foreign_key_check,
                "foreign_key_inventory": inventory,
                "vec0": vec0,
            },
            findings=findings,
        )


class ReferenceContractRegistry:
    """Known reference contracts with bounded candidate scans only."""

    def __init__(self, conn: sqlite3.Connection, scan_limit: int = DEFAULT_SCAN_LIMIT):
        self.conn = conn
        self.scan_limit = _validate_scan_limit(scan_limit)

    @staticmethod
    def _unknown_metrics(
        error_class: str | None = None, *, columns_truncated: bool = False
    ) -> dict[str, dict[str, Any]]:
        metric: dict[str, Any] = {"status": STATUS_UNKNOWN}
        if error_class:
            metric["error_class"] = error_class
        if columns_truncated:
            metric["columns_truncated"] = True
        return {
            name: dict(metric)
            for name in (
                "memory_embeddings",
                "vec_working",
                "graph_edges",
                "episodic_memory",
                "canonical_facts",
                "triples",
            )
        }

    def inspect(self) -> AdapterResult:
        catalog = _catalog(self.conn)
        if catalog.error_class:
            return AdapterResult(metrics=self._unknown_metrics(catalog.error_class))
        live_tables, live_error, live_columns_truncated = _live_memory_tables(
            self.conn, catalog, self.scan_limit
        )
        if live_error:
            return AdapterResult(metrics=self._unknown_metrics(live_error))
        if live_columns_truncated:
            return AdapterResult(metrics=self._unknown_metrics(columns_truncated=True))
        live_union = _live_id_union(live_tables)
        metrics: dict[str, Any] = {}
        findings: list[Finding] = []
        candidates: list[RepairCandidate] = []
        embedding_known_ids_sql = ""
        embedding_error: str | None = None

        embedding_columns = (
            _table_columns(self.conn, "memory_embeddings", self.scan_limit)
            if "memory_embeddings" in catalog.tables
            else None
        )
        embedding_columns_truncated = bool(
            embedding_columns
            and embedding_columns.truncated
            and "memory_id" not in embedding_columns.columns
        )
        if embedding_columns and embedding_columns.error_class:
            metrics["memory_embeddings"] = {"status": STATUS_UNKNOWN, "error_class": embedding_columns.error_class}
            embedding_error = embedding_columns.error_class
        elif embedding_columns_truncated:
            metrics["memory_embeddings"] = {"status": STATUS_UNKNOWN, "columns_truncated": True}
        elif not embedding_columns or "memory_id" not in embedding_columns.columns:
            metrics["memory_embeddings"] = {"status": "not_configured", "rows": 0, "orphan_rows": 0}
        elif not live_union:
            metrics["memory_embeddings"] = {"status": "unverifiable_contract", "rows": 0, "orphan_rows": 0}
        else:
            rows = _bounded_count(self.conn, "SELECT 1 FROM memory_embeddings", self.scan_limit)
            orphans = _bounded_count(
                self.conn,
                "SELECT 1 FROM memory_embeddings me WHERE me.memory_id IS NOT NULL "
                f"AND me.memory_id NOT IN ({live_union})",
                self.scan_limit,
            )
            metrics["memory_embeddings"] = _with_counts("checked", {"rows": rows, "orphan_rows": orphans})
            embedding_error = next((count.error_class for count in (rows, orphans) if count.error_class), None)
            if not rows.error_class and not orphans.error_class:
                embedding_known_ids_sql = "SELECT memory_id FROM memory_embeddings WHERE memory_id IS NOT NULL"
                if orphans.value:
                    findings.append(
                        Finding(
                            code="references.memory_embeddings_orphan",
                            status=STATUS_WARNING,
                            severity=SEVERITY_WARNING,
                            message="Fallback embedding references missing from all known memory tiers.",
                        )
                    )

        vec_status, vec_error = _vec_table_status(self.conn, "vec_working", catalog.tables.get("vec_working"))
        if embedding_error and vec_status == "available":
            metrics["vec_working"] = {"status": STATUS_UNKNOWN, "error_class": embedding_error}
        elif embedding_columns_truncated and vec_status == "available":
            metrics["vec_working"] = {"status": STATUS_UNKNOWN, "columns_truncated": True}
        elif "working_memory" not in live_tables or vec_status == "not_configured":
            metrics["vec_working"] = {"status": "not_configured", "orphan_rows": 0, "missing_backfill_rows": 0}
        elif vec_status != "available":
            metrics["vec_working"] = {"status": vec_status, "error_class": vec_error} if vec_error else {"status": vec_status}
        elif not embedding_columns or embedding_columns.error_class or "memory_id" not in embedding_columns.columns:
            metrics["vec_working"] = {"status": "unverifiable_contract"}
        else:
            orphan_rows = _bounded_count(
                self.conn,
                "SELECT 1 FROM vec_working vw LEFT JOIN working_memory wm ON wm.rowid = vw.rowid WHERE wm.rowid IS NULL",
                self.scan_limit,
            )
            missing_rows = _bounded_count(
                self.conn,
                "SELECT 1 FROM working_memory wm JOIN memory_embeddings me ON me.memory_id = wm.id "
                "LEFT JOIN vec_working vw ON vw.rowid = wm.rowid WHERE vw.rowid IS NULL",
                self.scan_limit,
            )
            metrics["vec_working"] = _with_counts("checked", {"orphan_rows": orphan_rows, "missing_backfill_rows": missing_rows})
            if not orphan_rows.error_class and not missing_rows.error_class and missing_rows.value:
                candidates.append(
                    RepairCandidate(
                        id="backfill-vec-working",
                        description="Backfill missing vec_working rows from existing fallback embeddings.",
                        finding_codes=["references.vec_working_missing_backfill"],
                    )
                )

        self._inspect_graph(
            catalog,
            live_union,
            embedding_known_ids_sql,
            embedding_error,
            embedding_columns_truncated,
            metrics,
            findings,
        )
        self._inspect_unverifiable_stores(catalog, metrics)
        return AdapterResult(metrics=metrics, findings=findings, repair_candidates=candidates)

    def _inspect_graph(
        self,
        catalog: _CatalogResult,
        live_union: str,
        embedding_known_ids_sql: str,
        embedding_error: str | None,
        embedding_columns_truncated: bool,
        metrics: dict[str, Any],
        findings: list[Finding],
    ) -> None:
        if "graph_edges" not in catalog.tables:
            metrics["graph_edges"] = {"status": "not_configured", "dangling_memory_endpoints": 0, "opaque_graph_node": 0}
            return
        columns = _table_columns(self.conn, "graph_edges", self.scan_limit)
        if columns.error_class:
            metrics["graph_edges"] = {"status": STATUS_UNKNOWN, "error_class": columns.error_class}
        elif columns.truncated and not {"source", "target"}.issubset(columns.columns):
            metrics["graph_edges"] = {"status": STATUS_UNKNOWN, "columns_truncated": True}
        elif not {"source", "target"}.issubset(columns.columns):
            metrics["graph_edges"] = {"status": "not_configured", "dangling_memory_endpoints": 0, "opaque_graph_node": 0}
        elif embedding_error:
            metrics["graph_edges"] = {"status": STATUS_UNKNOWN, "error_class": embedding_error}
        elif embedding_columns_truncated:
            metrics["graph_edges"] = {"status": STATUS_UNKNOWN, "columns_truncated": True}
        elif not live_union or not embedding_known_ids_sql:
            metrics["graph_edges"] = {"status": "unverifiable_contract", "dangling_memory_endpoints": 0, "opaque_graph_node": 0}
        else:
            known_ids = f"({live_union} UNION {embedding_known_ids_sql})"
            nodes = "SELECT source AS node FROM graph_edges UNION ALL SELECT target AS node FROM graph_edges"
            dangling = _bounded_count(self.conn, f"SELECT 1 FROM ({nodes}) nodes WHERE node IN {known_ids} AND node NOT IN ({live_union})", self.scan_limit)
            opaque = _bounded_count(self.conn, f"SELECT 1 FROM ({nodes}) nodes WHERE node NOT IN {known_ids}", self.scan_limit)
            metrics["graph_edges"] = _with_counts("checked", {"dangling_memory_endpoints": dangling, "opaque_graph_node": opaque})
            if not dangling.error_class and dangling.value:
                findings.append(Finding(code="references.graph_dangling_memory_endpoint", status=STATUS_WARNING, severity=SEVERITY_WARNING, message="Known graph memory endpoints are dangling; opaque graph nodes remain unverifiable."))

    def _inspect_unverifiable_stores(self, catalog: _CatalogResult, metrics: dict[str, Any]) -> None:
        metrics["episodic_memory"] = {"status": "unverifiable_parent" if "episodic_memory" in catalog.tables else "not_configured"}
        for table, required, metric_name, sql, status in (
            ("canonical_facts", {"owner_id", "category", "name", "valid_until"}, "duplicate_current_slots", "SELECT 1 FROM (SELECT 1 FROM canonical_facts WHERE valid_until IS NULL GROUP BY owner_id, category, name HAVING COUNT(*) > 1)", "unverifiable_contract"),
            ("triples", {"valid_from", "valid_until"}, "invalid_temporal_ranges", "SELECT 1 FROM triples WHERE valid_until IS NOT NULL AND valid_from IS NOT NULL AND julianday(valid_until) < julianday(valid_from)", "not_a_memory_reference"),
        ):
            if table not in catalog.tables:
                metrics[table] = {"status": "not_configured", metric_name: 0}
                continue
            columns = _table_columns(self.conn, table, self.scan_limit)
            if columns.error_class:
                metrics[table] = {"status": STATUS_UNKNOWN, "error_class": columns.error_class}
            elif columns.truncated and not required.issubset(columns.columns):
                metrics[table] = {"status": STATUS_UNKNOWN, "columns_truncated": True}
            elif not required.issubset(columns.columns):
                metrics[table] = {"status": "not_configured", metric_name: 0}
            else:
                metrics[table] = _with_counts(status, {metric_name: _bounded_count(self.conn, sql, self.scan_limit)})


class VectorCoverageAdapter:
    """Bounded coverage checks for working, episodic, and compatibility tiers."""

    def __init__(self, conn: sqlite3.Connection, scan_limit: int = DEFAULT_SCAN_LIMIT):
        self.conn = conn
        self.scan_limit = _validate_scan_limit(scan_limit)

    def inspect(self) -> AdapterResult:
        catalog = _catalog(self.conn)
        if catalog.error_class:
            unknown = {"status": STATUS_UNKNOWN, "error_class": catalog.error_class}
            return AdapterResult(metrics={name: dict(unknown) for name in ("working", "episodic", "legacy", "canonical", "triples")})
        return AdapterResult(metrics={
            "working": self._working(catalog),
            "episodic": self._episodic(catalog),
            "legacy": {"status": "compatibility_store" if "memories" in catalog.tables else "not_configured"},
            "canonical": {"status": "not_applicable" if "canonical_facts" in catalog.tables else "not_configured"},
            "triples": {"status": "not_applicable" if "triples" in catalog.tables else "not_configured"},
        })

    def _working(self, catalog: _CatalogResult) -> dict[str, Any]:
        empty = {"status": "no_vectors", "active_source_rows": 0, "fallback_embedding_rows": 0, "vec_working_rows": 0, "missing_vec_working_rows": 0, "orphan_vec_working_rows": 0}
        if "working_memory" not in catalog.tables:
            return empty
        columns = _table_columns(self.conn, "working_memory", self.scan_limit)
        if columns.error_class:
            return {"status": STATUS_UNKNOWN, "error_class": columns.error_class}
        if "id" not in columns.columns:
            if columns.truncated:
                return {"status": STATUS_UNKNOWN, "columns_truncated": True}
            return empty
        if columns.truncated and not _DOCTOR_TABLE_COLUMNS["working_memory"].issubset(columns.columns):
            return {"status": STATUS_UNKNOWN, "columns_truncated": True}
        predicates = []
        if "valid_until" in columns.columns:
            predicates.append("(wm.valid_until IS NULL OR julianday(wm.valid_until) > julianday('now'))")
        if "superseded_by" in columns.columns:
            predicates.append("wm.superseded_by IS NULL")
        active_where = " WHERE " + " AND ".join(predicates) if predicates else ""
        source = _bounded_count(self.conn, "SELECT 1 FROM working_memory wm" + active_where, self.scan_limit)
        if source.error_class:
            return {"status": STATUS_UNKNOWN, "error_class": source.error_class}
        result = dict(empty, active_source_rows=source.value)
        if source.truncated:
            result["active_source_rows_truncated"] = True
        embedding_columns = (
            _table_columns(self.conn, "memory_embeddings", self.scan_limit)
            if "memory_embeddings" in catalog.tables
            else None
        )
        if embedding_columns and embedding_columns.error_class:
            return {"status": STATUS_UNKNOWN, "error_class": embedding_columns.error_class}
        if (
            embedding_columns
            and embedding_columns.truncated
            and "memory_id" not in embedding_columns.columns
        ):
            return {"status": STATUS_UNKNOWN, "columns_truncated": True}
        if embedding_columns and "memory_id" in embedding_columns.columns:
            fallback = _bounded_count(self.conn, "SELECT 1 FROM working_memory wm JOIN memory_embeddings me ON me.memory_id = wm.id" + active_where, self.scan_limit)
            if fallback.error_class:
                return {"status": STATUS_UNKNOWN, "error_class": fallback.error_class}
            result["fallback_embedding_rows"] = fallback.value
            if fallback.truncated:
                result["fallback_embedding_rows_truncated"] = True
        vec_status, vec_error = _vec_table_status(self.conn, "vec_working", catalog.tables.get("vec_working"))
        if vec_status not in {"available", "not_configured"}:
            if vec_status == STATUS_PRESENT_BUT_UNLOADABLE:
                return {
                    "status": "unavailable",
                    "vec0_status": vec_status,
                    "error_class": vec_error or "sqlite_error",
                }
            return {"status": STATUS_UNKNOWN, "error_class": vec_error or "sqlite_error"}
        if vec_status == "available":
            counts = {
                "vec_working_rows": _bounded_count(self.conn, "SELECT 1 FROM vec_working", self.scan_limit),
                "missing_vec_working_rows": _bounded_count(self.conn, "SELECT 1 FROM working_memory wm LEFT JOIN vec_working vw ON vw.rowid = wm.rowid" + active_where + (" AND " if active_where else " WHERE ") + "vw.rowid IS NULL", self.scan_limit),
                "orphan_vec_working_rows": _bounded_count(self.conn, "SELECT 1 FROM vec_working vw LEFT JOIN working_memory wm ON wm.rowid = vw.rowid WHERE wm.rowid IS NULL", self.scan_limit),
            }
            if any(count.error_class for count in counts.values()):
                return {"status": STATUS_UNKNOWN, "error_class": next(count.error_class for count in counts.values() if count.error_class)}
            for name, count in counts.items():
                result[name] = count.value
                if count.truncated:
                    result[f"{name}_truncated"] = True
            if source.truncated or any(count.truncated for count in counts.values()):
                result["status"] = "scan_limited"
            elif source.value == 0:
                result["status"] = "no_vectors"
            elif result["fallback_embedding_rows"] and result["missing_vec_working_rows"]:
                result["status"] = "backfill_available"
            elif result["missing_vec_working_rows"] == 0:
                result["status"] = "complete"
        else:
            if vec_status != "not_configured":
                result["vec0_status"] = vec_status
                if vec_error:
                    result["vec0_error_class"] = vec_error
            if result["fallback_embedding_rows"]:
                result["status"] = "fallback_only"
            elif source.value:
                result["status"] = "unavailable"
        return result

    def _episodic(self, catalog: _CatalogResult) -> dict[str, Any]:
        result = {"status": "not_configured", "source_rows": 0, "binary_vector_rows": 0, "vec_episode_rows": 0}
        if "episodic_memory" not in catalog.tables:
            return result
        columns = _table_columns(self.conn, "episodic_memory", self.scan_limit)
        if columns.error_class:
            return {"status": STATUS_UNKNOWN, "error_class": columns.error_class}
        if columns.truncated and "binary_vector" not in columns.columns:
            return {"status": STATUS_UNKNOWN, "columns_truncated": True}
        source = _bounded_count(self.conn, "SELECT 1 FROM episodic_memory", self.scan_limit)
        if source.error_class:
            return {"status": STATUS_UNKNOWN, "error_class": source.error_class}
        result["source_rows"] = source.value
        if source.truncated:
            result["source_rows_truncated"] = True
        if "binary_vector" in columns.columns:
            binary = _bounded_count(self.conn, "SELECT 1 FROM episodic_memory WHERE binary_vector IS NOT NULL", self.scan_limit)
            if binary.error_class:
                return {"status": STATUS_UNKNOWN, "error_class": binary.error_class}
            result["binary_vector_rows"] = binary.value
            if binary.truncated:
                result["binary_vector_rows_truncated"] = True
        vec_status, vec_error = _vec_table_status(self.conn, "vec_episodes", catalog.tables.get("vec_episodes"))
        if vec_status not in {"available", "not_configured"}:
            if vec_status == STATUS_PRESENT_BUT_UNLOADABLE:
                return {
                    "status": "unavailable",
                    "vec0_status": vec_status,
                    "error_class": vec_error or "sqlite_error",
                }
            return {"status": STATUS_UNKNOWN, "error_class": vec_error or "sqlite_error"}
        if vec_status == "available":
            vec_rows = _bounded_count(self.conn, "SELECT 1 FROM vec_episodes", self.scan_limit)
            if vec_rows.error_class:
                return {"status": STATUS_UNKNOWN, "error_class": vec_rows.error_class}
            result["vec_episode_rows"] = vec_rows.value
            if vec_rows.truncated:
                result["vec_episode_rows_truncated"] = True
        if source.truncated:
            result["status"] = "scan_limited"
        elif source.value == 0:
            result["status"] = "not_configured" if vec_status == "not_configured" else "no_vectors"
        elif result["binary_vector_rows"] == source.value:
            result["status"] = "complete"
        elif result["binary_vector_rows"] or result["vec_episode_rows"]:
            result["status"] = "partial"
        elif vec_status == "not_configured":
            result["status"] = "not_configured"
        elif vec_status == STATUS_PRESENT_BUT_UNLOADABLE:
            result["status"] = "unavailable"
        else:
            result["status"] = "fallback_only"
        if vec_error and vec_status != "available":
            result["vec0_error_class"] = vec_error
        return result
