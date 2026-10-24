#!/usr/bin/env python3
"""Regression test for the per-home binding defect (#1050 follow-up).

FAILS before the patch: constructing MnemosyneMemoryProvider while a profile's
turn-scoped home is set raises RuntimeError("no binding for the current home").
PASSES after the patch: the provider constructs, initialize() binds the home,
and a memory write lands in THAT profile's store.

Run with the venv python of the environment under test:
  <env>/venv/bin/python3 test_binding_regression.py <profile-home>
"""
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

HOME = sys.argv[1] if len(sys.argv) > 1 else "/home/beast/.hermes/profiles/markets"
os.environ["HERMES_HOME"] = HOME

sys.path.insert(0, str(Path(__file__).parent / "src"))

FAILURES = []


def check(label, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(label)


print(f"home under test: {HOME}")

# --- 1. construction must not raise -----------------------------------------
from mnemosyne_hermes import MnemosyneMemoryProvider  # noqa: E402

try:
    provider = MnemosyneMemoryProvider()
    check("provider constructs under a turn-scoped home", True)
except RuntimeError as exc:
    check("provider constructs under a turn-scoped home", False, str(exc)[:90])
    print("\nRESULT: FAIL — defect present (this is the pre-patch behaviour).")
    raise SystemExit(1)

# --- 2. _beam reads None before initialize (no AttributeError) ---------------
check("self._beam reads None before initialize", provider._beam is None)

# --- 3. initialize() must bind the home, not raise ---------------------------
# Every real caller passes hermes_home (see _ensure_initialized at ~1529 and
# PluginManager). Without it _initialize_locked sets _init_home=None and the
# turn-scoped key is deliberately NOT used to bind — that is the #1050 guard,
# so the test must mirror production and pass the home explicitly.
try:
    provider.initialize("binding-regression-test", hermes_home=HOME)
    check("initialize() binds the home", True)
except Exception as exc:  # noqa: BLE001
    check("initialize() binds the home", False, f"{type(exc).__name__}: {exc}"[:90])
    raise SystemExit(1)

# --- 4. a real write lands in THIS profile's store ---------------------------
# Use the beam initialize() already bound to this profile's own store. No
# scratch DB and no re-pointing: the whole point is to prove the row lands in
# THE PROFILE'S REAL STORE, so the count is measured against the file on disk.
store = Path(HOME) / "mnemosyne" / "data" / "mnemosyne.db"
marker = "binding-regression-probe-5oct2026"


def row_count():
    if not store.exists():
        return 0
    with sqlite3.connect(store) as con:
        try:
            return con.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0]
        except sqlite3.Error:
            return 0


before = row_count()
# Go through the real tool entry point the agent calls, so the test exercises
# the exact path that silently lost writes in production.
provider.handle_tool_call("mnemosyne_remember",
                          {"content": marker, "importance": 0.1, "scope": "session"})
after = row_count()
check("remember() persists a row to the profile's own store", after > before,
      f"{before} -> {after} rows in {store}")

with sqlite3.connect(store) as con:
    hit = con.execute(
        "SELECT COUNT(*) FROM working_memory WHERE content LIKE ?", (f"%{marker}%",)
    ).fetchone()[0]
check("the exact fact is readable back from the store", hit > 0, f"{hit} match(es)")

# Clean up the probe row so the store is left as we found it.
if hit:
    with sqlite3.connect(store) as con:
        con.execute("DELETE FROM working_memory WHERE content LIKE ?", (f"%{marker}%",))
        con.commit()
    check("probe row removed, store restored", row_count() == before, f"{row_count()} rows")

# --- 5. the fail-closed guard must STILL protect real cross-home writes -----
try:
    from hermes_constants import set_hermes_home_override
    import hermes_constants

    provider.__dict__["_init_home"] = None
    tok = set_hermes_home_override("/home/beast/.hermes/profiles/__no_such_profile__")
    try:
        provider._beam = object()
        check("fail-closed guard still blocks an uninitialised home", False,
              "write was ALLOWED — guard was weakened")
    except RuntimeError:
        check("fail-closed guard still blocks an uninitialised home", True)
    finally:
        hermes_constants.reset_hermes_home_override(tok)
except ImportError:
    print("  [SKIP] hermes_constants unavailable; cannot assert guard")

print()
if FAILURES:
    print(f"RESULT: FAIL — {len(FAILURES)} check(s) failed: {FAILURES}")
    raise SystemExit(1)
print("RESULT: PASS — provider binds, writes land, fail-closed guard intact.")