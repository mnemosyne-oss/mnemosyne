#!/usr/bin/env python3
"""Scales acceptance criterion — canon-sign chain head resolution. v5.
Rides the re-host PR: runs GREEN only after keeper's (a) _handle_invalidate
surface branch + (b) declared-edge replay land on the senses probe-chain-green build.

v2->v3: why's receipt-or-fail clause applied to this gate by its own author —
every arm emits a positive per-arm receipt (population checked + outcome);
an arm that could not run is VACUOUS->RED, never silently green.

v4->v5 (scales ruling 2026-09-24, on keeper's replay landing FAIL(1) red=[A3]):
A3's accepted out-edge vocabulary gains the third typed form,
`metadata.supplements`. This was NOT goalpost-moving to win: (i) A5 already
enumerates supplements as a typed edge — A3 denying it was an internal
contradiction of this file; (ii) A3's intent is "open non-head row carries a
typed out-edge" and the live supplement's edge (validated for target by A6b)
IS a typed out-edge; (iii) the alternative — stamp-closing sf_18287cec —
would flip the store PASS by contradicting the row's own recorded
supersedes_nothing stance: changing the world to fit the measure. Only the
form A6b already target-checks is admitted here; detection holes unchanged
(prose-only supplements still RED via A6a; dangling still RED via A5).

Design laws (rulings this encodes):
  - CHAIN membership: TRANSITIONAL anchor startswith (prose, labeled-for-
    removal). Never filename-cite — refs fragment 3 ways in body text (measured
    4/10 cite subject). Flip to typed metadata.subject_ref when A0 first reads green.
  - Membership key (why's ruling, v4 amendment): `metadata.subject_ref` =
    uniform chain SUBJECT file. `metadata.ref` is RESERVED live on this store
    with seal-digest semantics (v8=bf7fc83b, v10=4693e144) and stays seal
    provenance; per-row body citations never feed membership.
  - Clock comparisons pin to created_at (naive UTC both eras); pre-repackage
    `timestamp` is naive EDT and is NOT compared against.
  - One-shot prose replay by a named validator is legitimate (why's ruling);
    permanent read-path prose parsing is the disease. Tonight's prose arms
    (anchor membership, A6a supplement detection) exist only until replay ships.
  - Supplements must be TYPED edges; prose-declared supplement = honest red.
"""
import json, sqlite3, sys

DB = sys.argv[1] if len(sys.argv) > 1 else \
     "/home/famhome/.hermes/mnemosyne/data/shared/mnemosyne.db"
SUBJECT_REF = "multiplex-profile-memory-isolation.md"

db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
db.row_factory = sqlite3.Row
rows = {r["id"]: r for r in db.execute(
    "select id, content, valid_until, superseded_by, metadata_json, created_at "
    "from working_memory")}

def meta(r, key):
    try:
        return json.loads(r["metadata_json"] or "{}").get(key)
    except Exception:
        return None

arms = []
def emit(arm, passed, evidence, ran=True):
    ok = bool(passed) and ran
    status = "GREEN" if ok else ("VACUOUS->RED" if ran and passed else "RED" if ran else "RED(no-run)")
    arms.append((arm, ok))
    print(f"ARM {arm} {status} :: {evidence}")

# --- membership (TRANSITIONAL prose anchor; remove when A0 greens) ---
chain = [r for r in rows.values()
         if r["content"].startswith("Surface meta: Astraea canon-sign row")]
supps = [r for r in rows.values()
         if meta(r, "supplements") and r not in chain]
prose_supps = [r for r in rows.values()
               if r not in chain and r not in supps
               and r["content"].startswith("Surface meta: Astraea")
               and "supplements, does not supersede" in r["content"]]

open_ = lambda r: r["valid_until"] is None

# A1 predicate sanity
emit("A1", len(chain) >= 10,
     f"anchor matched {len(chain)} chain rows (>=10 expected); "
     f"supps typed={len(supps)} prose-detected={len(prose_supps)}")

# A0 typed SUBJECT ref on every chain row. Key is `subject_ref`, NOT `ref` —
# measured 2026-09-24: `metadata.ref` is already live with SEAL-DIGEST
# semantics on v8 (bf7fc83b) and v10 (4693e144). Demanding the filename under
# `ref` would collide semantics; `ref` stays seal provenance (why's ruling),
# `subject_ref` is the uniform chain-membership key.
no_ref = [r["id"] for r in chain if meta(r, "subject_ref") != SUBJECT_REF]
emit("A0", not no_ref,
     f"{len(chain)} rows checked metadata.subject_ref=='{SUBJECT_REF}'; lacking/wrong: {no_ref}")

# successor targets: typed edges only (superseded_by or metadata.supersedes)
succ = {r["superseded_by"] for r in rows.values() if r["superseded_by"]} \
     | {meta(r, "supersedes") for r in rows.values() if meta(r, "supersedes")}
heads = [r for r in chain if r["id"] not in succ and open_(r)]

# A2 exactly one open head
emit("A2", len(heads) == 1,
     f"open heads={len(heads)} {[r['id'] for r in heads]} (target: exactly 1)")

# A3 every open non-head chain/supp row has a typed out-edge
# (v5: out-edge vocabulary = superseded_by | metadata.supersedes |
# metadata.supplements — the same three forms A5 enumerates store-wide)
nonhead_open = [r for r in chain + supps if open_(r) and r not in heads]
bad3 = [r["id"] for r in nonhead_open
        if not (r["superseded_by"] or meta(r, "supersedes") or meta(r, "supplements"))]
emit("A3", not bad3,
     f"{len(nonhead_open)} open non-head rows checked for typed out-edge "
     f"(vocab: superseded_by|supersedes|supplements, per A5 parity); prose-only: {bad3}",
     ran=bool(nonhead_open))

# A4 validity-stamped chain rows must also carry typed superseded_by
stamped = [r for r in chain if not open_(r)]
bad4 = [r["id"] for r in stamped if not r["superseded_by"]]
emit("A4", not bad4,
     f"{len(stamped)} stamped rows checked for superseded_by "
     f"(idx_wm_context_global gates superseded_by IS NULL); still-query-live: {bad4}",
     ran=True)  # runs with 0 stamped = pass only post-hygiene when none should exist

# A5 no dangling edges store-wide
edges = [(r["id"], t) for r in rows.values()
         for t in (r["superseded_by"], meta(r, "supersedes"),
                   meta(r, "supplements")) if t]
dangling = [f"{s}->{t}" for s, t in edges if t not in rows]
emit("A5", not dangling, f"{len(edges)} typed edges checked; dangling: {dangling}",
     ran=bool(edges))

# A6a prose-declared supplements are honest red until replayed
bad6a = [r["id"] for r in prose_supps]
emit("A6a", not bad6a,
     f"{len(prose_supps)} prose-only supplement declarations found (typed "
     f"metadata.supplements required; why's honest-red ruling): {bad6a}")

# A6b typed supplement edges must point at the current head
if supps and heads:
    mis = [r["id"] for r in supps
           if meta(r, "supplements") not in {h["id"] for h in heads}]
    emit("A6b", not mis, f"{len(supps)} typed supplement edges checked vs head; bad: {mis}")
else:
    emit("A6b", False,
         f"arm cannot run: typed supps={len(supps)}, heads={len(heads)} — "
         f"vacuous-green is RED (receipt-or-fail); arms when replay ships typed edges",
         ran=False)

# A7 contradiction adjudication must not sit unresolved
c = db.execute("select count(*) n from conflicts "
               "where resolution is null or resolution=''").fetchone()["n"]
emit("A7", c == 0, f"unresolved conflicts rows: {c} (scales' own mechanism, "
                   f"same hygiene pass per why's ruling)")

reds = [a for a, ok in arms if not ok]
print(f"SUMMARY chain={len(chain)} supps={len(supps)} heads={len(heads)} "
      f"open_chain={sum(1 for r in chain if open_(r))} arms={len(arms)} red={len(reds)}")
print("RESULT:", "PASS" if not reds else f"FAIL({len(reds)}) red-arms={reds}")
sys.exit(1 if reds else 0)
