"""Regression tests for pending-record claim handling (Codex finding, 2026-09-27).

`_claim_pending_record` caught ONLY FileNotFoundError, and its call site sits
OUTSIDE the per-record `try`. So a single record that could not be renamed for
any other reason (permission, ENOSPC, EXDEV) raised out of the replay loop and
aborted every REMAINING pending record.

The fix raises a dedicated `PendingClaimError` (an OSError subclass) so the
caller can tell "could not claim" from "already claimed" and CONTINUE the batch.

These tests pin:
  1. ENOENT still returns None — the benign "already claimed" signal is unchanged.
  2. Any other OSError raises PendingClaimError, carrying the original cause.
  3. A failed claim leaves no stray .*.claim file and the record still pending.
  4. The happy path is unchanged: atomic move to a private claim name.
  5. Batch continuation: the caller's loop keeps going past an unclaimable
     record, and reports the TRUE reason rather than "already claimed".
"""

from __future__ import annotations

import ast
import inspect
import os
import pathlib
import stat
from pathlib import Path

import pytest

import hermes_memory_provider as hmp
from hermes_memory_provider import PendingClaimError, _claim_pending_record


def _caller_source() -> str:
    return pathlib.Path(hmp.__file__).read_text(encoding="utf-8")


def test_missing_file_returns_none() -> None:
    """The documented 'nothing to claim' signal is preserved."""
    assert _claim_pending_record(Path("/nonexistent/pending/abc.json")) is None


@pytest.mark.parametrize(
    "exc",
    [
        PermissionError(13, "Permission denied"),
        OSError(28, "No space left on device"),
        OSError(18, "Invalid cross-device link"),
        IsADirectoryError(21, "Is a directory"),
        OSError(16, "Device or resource busy"),
    ],
)
def test_non_enoent_oserror_raises_pending_claim_error(tmp_path: Path, exc: OSError) -> None:
    """A rename failure that is NOT FileNotFoundError must not silently degrade.

    It must raise PendingClaimError specifically, so the caller can distinguish
    an actionable OS failure from a benign claim race.
    """
    record = tmp_path / "rec.json"
    record.write_text("{}", encoding="utf-8")

    def boom(self, target, *a, **k):  # noqa: ANN001, ANN002, ANN003
        raise exc

    original = Path.rename
    Path.rename = boom  # type: ignore[method-assign]
    try:
        # Assert on the exception class bound to the FUNCTION, not the one this
        # module imported. Other tests in this suite purge `hermes_memory_provider`
        # from sys.modules and re-import it, so the package can be loaded twice
        # in one session; the two PendingClaimError classes are then distinct
        # objects and pytest.raises(PendingClaimError) — bound at OUR import
        # time — does not catch the one raised by the function under test.
        # Reading the class off the function's own globals is immune to that.
        expected = _claim_pending_record.__globals__['PendingClaimError']
        with pytest.raises(expected) as ei:
            _claim_pending_record(record)
    finally:
        Path.rename = original  # type: ignore[method-assign]

    assert ei.value.__cause__ is exc, "the OS cause must be preserved for the operator"


def test_failed_claim_creates_no_claim_file(tmp_path: Path) -> None:
    """No stray .*.claim file may be left behind, and the record must survive."""
    record = tmp_path / "rec.json"
    record.write_text("{}", encoding="utf-8")

    def boom(self, target, *a, **k):  # noqa: ANN001, ANN002, ANN003
        raise OSError(28, "No space left on device")

    original = Path.rename
    Path.rename = boom  # type: ignore[method-assign]
    try:
        expected = _claim_pending_record.__globals__['PendingClaimError']
        with pytest.raises(expected):
            _claim_pending_record(record)
    finally:
        Path.rename = original  # type: ignore[method-assign]

    assert list(tmp_path.glob(".*.claim")) == []
    assert record.exists(), "the original pending record must be left intact"


def test_successful_claim_is_atomic_and_private(tmp_path: Path) -> None:
    """The happy path still works: the record moves to a private claim name."""
    record = tmp_path / "rec.json"
    record.write_text('{"id": "rec"}', encoding="utf-8")

    claim = _claim_pending_record(record)
    assert claim is not None
    assert claim.exists()
    assert not record.exists()
    assert claim.name.startswith(".rec.json.") and claim.name.endswith(".claim")


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses directory permissions")
def test_unreadable_directory_raises_pending_claim_error(tmp_path: Path) -> None:
    """A real permission failure, not a monkeypatch, behaves the same way."""
    d = tmp_path / "pending"
    d.mkdir()
    record = d / "rec.json"
    record.write_text("{}", encoding="utf-8")
    d.chmod(stat.S_IRUSR | stat.S_IXUSR)  # r-x: no write -> rename must fail
    try:
        expected = _claim_pending_record.__globals__['PendingClaimError']
        with pytest.raises(expected):
            _claim_pending_record(record)
    finally:
        d.chmod(stat.S_IRWXU)


# --- the central regression Codex asked for: batch continuation -------------

def test_caller_catches_pending_claim_error_and_continues() -> None:
    """The call site must catch PendingClaimError, report the true cause, and continue.

    Verified structurally against the real source: a `try` around the claim that
    catches PendingClaimError, appends a failure, and `continue`s — and no longer
    reports an OS-level failure as "already claimed".
    """
    src = _caller_source()
    tree = ast.parse(src)

    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_claim_pending_record":
            target = node
            break
    assert target is not None, "call to _claim_pending_record not found"

    # the call must be wrapped in a Try whose handlers catch PendingClaimError
    wrapper = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and any(
            isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_claim_pending_record"
            for n in ast.walk(node)
        ):
            wrapper = node
            break
    assert wrapper is not None, (
        "the claim call is not wrapped in try/except — a claim failure would abort "
        "the whole replay loop, which is the original bug"
    )

    caught = set()
    for handler in wrapper.handlers:
        exc = handler.type
        if exc is None:
            caught.add("bare")
        elif isinstance(exc, ast.Name):
            caught.add(exc.id)
        elif isinstance(exc, ast.Tuple):
            caught.update(e.id for e in exc.elts if isinstance(e, ast.Name))
    assert "PendingClaimError" in caught, f"PendingClaimError not caught, only {caught}"

    # and the handler must continue the loop rather than re-raise
    handler = next(
        h
        for h in wrapper.handlers
        if isinstance(h.type, ast.Name) and h.type.id == "PendingClaimError"
    )
    assert any(isinstance(n, ast.Continue) for n in ast.walk(handler)), (
        "the PendingClaimError handler must `continue` so later records still replay"
    )
    assert not any(isinstance(n, ast.Raise) for n in ast.walk(handler)), (
        "the handler must not re-raise; that is the original batch-abort bug"
    )


def test_caller_still_reports_claim_race_as_already_claimed() -> None:
    """The benign None path keeps its own distinct message."""
    src = _caller_source()
    assert "pending record already claimed" in src
    assert "pending record not claimable" in src, (
        "an OS-level claim failure must get its own error string, not be "
        "misreported as 'already claimed'"
    )


def test_pending_claim_error_is_an_oserror() -> None:
    """It must stay an OSError subclass so existing broad handlers still work."""
    assert issubclass(PendingClaimError, OSError)
    assert "record existed but could not be claimed" in (PendingClaimError.__doc__ or "")


def test_helper_signature_unchanged() -> None:
    sig = inspect.signature(_claim_pending_record)
    assert list(sig.parameters) == ["record_path"]
