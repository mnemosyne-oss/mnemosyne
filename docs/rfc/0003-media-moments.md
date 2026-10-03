# RFC 0003: Media Assets and the Moment Index

**Status:** Draft
**Author:** abdiisan
**Related issue:** TBD
**Target version:** next MAJOR
**Depends on:** RFC 0002 (modality providers)
**Related:** RFC 0004 (archive boundary)

---

## 0. Summary

Mnemosyne should be able to answer "at what point in that video did someone talk about X" and "which of my screenshots shows the dark-mode palette", without ever becoming an ambient screen recorder.

This RFC specifies how: a **media asset registry** keyed by reference hash, and a **moment index** of semantically tagged spans within those assets. Mnemosyne stores the reference and the text. It does not store the bytes.

Two decisions carry the design:

1. **A moment is an ordinary `working_memory` row.** No new vec0 table. It therefore inherits recall, decay, consolidation, sync, and reindex with no new code.
2. **Media-internal time is a different axis from wall-clock time.** `t_start_ms` is meaningless across assets and must never enter `event_date`.

Image and video come first. Audio and voice profiles reuse the same primitives with a `speaker` dimension, and are phased later.

---

## 1. R&D Summary: what the codebase provides and what it lacks

### 1.1 There is no span model anywhere in the repo

The only structured non-text metadata with real columns is wall-clock time: `event_date` and `event_date_precision`, added to both memory tables at `beam.py:1228` and `:1231` and indexed immediately after, populated by `core/temporal_parser.py`, consumed by `_temporal_voice` (`core/polyphonic_recall.py:617`).

There is no offset, no duration-within-content, no page, no bounding box, no chunk table, and no parent-document linkage. Note that `chunk_memories_by_budget()` in `core/local_llm.py` is *not* content chunking: it groups whole memories into batches that fit an LLM context window. The nearest thing to provenance linking is `episodic_memory.summary_of` and `annotations.kind='has_source'`.

So the moment primitive is genuinely new. Nothing needs to be refactored to make room for it.

### 1.2 Content-addressed storage already exists, and is write-only

`core/content_sanitizer.py` is a working content-addressed blob store: `_store_blob` at `:91` writes bytes to `~/.hermes/mnemosyne/blobs/<h[0:2]>/<h[0:4]>/<sha256>` under `_blob_root()` (`:32`, overridable via `MNEMOSYNE_BLOB_DIR`), and `sanitize_content` at `:103` replaces the row content with a text stub while recording `{"blob_ref": "blob://sha256/<hash>", ...}` into metadata.

It is invoked from three places: `core/memory.py:358`, `core/beam.py:3251`, and `core/beam.py:3523`. **Nothing in the tree ever reads a blob back.** A grep for `blob_ref` or `blob://` outside `content_sanitizer.py` and `tests/` returns nothing. RFC 0004 gives it a reader.

Two details that matter here:

- `mime` is populated **only** on the `data:` URI branch (`content_sanitizer.py:128`). The size-cap branch (`:141-145`) and the high-entropy branch (`:157-162`) omit it.
- `sanitize_content` fires on any content over 1 MB (`SIZE_HARD_CAP`, `:22`) or over 100 KB with Shannon entropy above 5.0 (`:24`, `:153`). **A `data:` URI passed as `content` will be rewritten into a stub.** This directly constrains the ingest API in §3.3.

### 1.3 A new vec0 table is expensive and permanent

Adding a vec0 virtual table costs edits in four hardcoded whitelists, each of which must agree:

| Location | Function | Purpose |
|---|---|---|
| `beam.py:552` | `_existing_vec_dim` | which tables to parse for the stored dimension |
| `beam.py:1546` | `_vec_table_available` | availability probe |
| `beam.py:2152` | `_vec_table_insert` | write dispatch |
| `beam.py:2450` | `reindex_vectors` | recreate loop |

It also permanently enlarges the dimension-mismatch blast radius described in RFC 0002 §1.5, and it requires maintaining rowid alignment between the vec table and its parent (see `_store_working_embedding` at `beam.py:2216` and `_wm_vec_delete` at `:2204`).

`vec_facts` (declared `beam.py:1220`, never written) is the proof of cost: `reindex_vectors` recreates it empty forever, with a comment at `beam.py:2448` explaining that this is so its declared dimension cannot mismatch a query.

### 1.4 First-class memory rows come with the whole engine attached

`_store_working_embedding` (`beam.py:2216`) already writes both `memory_embeddings` and `vec_working` for any working-memory row. So a row placed in `working_memory` automatically receives:

- the dense vector voice and the lexical FTS voice
- recency decay and tier degradation
- sleep and consolidation into `episodic_memory`
- the veracity multiplier
- a `memory_events` row (`beam.py:745`) so deletion and mutation propagate through sync
- `mnemosyne reindex` coverage (`reindex_vectors`, `beam.py:2394`)

This is the argument for §3.2.

### 1.5 The recall filter pattern is established, and has six touch points

`memory_type` is the template to copy. It was added as an additive column via plain `ALTER TABLE` wrapped in try/except at `beam.py:658` and `:662`, and its filter clauses appear at `beam.py:5695` (working memory) and `:6195` (episodic). `recall()` is at `beam.py:5451`; `_recall_polyphonic` at `:7233`; the enhanced-recall cache key at `:6715`.

### 1.6 Schema conventions

- **All DDL lives in `init_beam`** (`beam.py:592`), or in a subsystem module with its own idempotent `init_*` function. `core/annotations.py:124` (`_init_annotations_with_conn`) and `core/canonical.py` are the templates for the latter, including the pattern of accepting an optional shared connection in the store's `__init__` (`annotations.py:179`).
- **Constraints are added as indexes, not table constraints**, so existing databases acquire the guarantee on next init with no migration. See `annotations.py:159` and the *partial* unique index at `canonical.py:131-132`.
- **No schema-level foreign keys.** Issue #503 was closed because `PRAGMA foreign_keys=ON` broke 22 tests that intentionally create orphan rows, and an FK on `memory_embeddings` previously made every embedding insert silently fail. Referential integrity is checked at the application layer, not enforced by SQLite.

---

## 2. Data model

New module `mnemosyne/core/media.py`, following the `core/annotations.py` template: an idempotent `_init_media_with_conn(conn)` called from a `MediaStore(conn=...)` that reuses BeamMemory's thread-local connection.

> **Correction (2026-07-30).** An earlier draft of this section said `_init_media_with_conn` is called "both from `init_beam` and from a `MediaStore(conn=...)`", and §5 budgeted "3 lines in `init_beam`". Both are wrong, and they misdescribe the precedent this RFC claims to follow. `init_annotations` is **not** called from `init_beam` — it is called from `BeamMemory._ensure_e6_schema_with_migration` (`beam.py:3091`). The closer precedent is `CanonicalStore`, constructed in `BeamMemory.__init__` at `beam.py:3043`, with its DDL owned entirely by `canonical.py:89-137`. `init_beam` (`beam.py:592`) contains only first-party DDL.
>
> **The correct wiring is therefore: construct `MediaStore(db_path=..., conn=self.conn)` in `BeamMemory.__init__` beside `self.canonical`, and let the store's own init run the idempotent DDL. Zero lines in `init_beam`.**
>
> This does not change the migration story. `BeamMemory.__init__` runs on every open and the DDL is all `CREATE TABLE/INDEX IF NOT EXISTS`, so existing banks acquire the tables on next open with no migration module — which is exactly how `canonical_facts` reached existing databases.

### 2.1 `media_assets`

One row per referenced piece of media. **No BLOB column, by design.**

```sql
CREATE TABLE IF NOT EXISTS media_assets (
    asset_id                TEXT PRIMARY KEY,
    content_hash            TEXT,
    ref_kind                TEXT NOT NULL,   -- sha256|url|youtube|file|blob|archive
    ref_value               TEXT NOT NULL,
    modality                TEXT NOT NULL,   -- image|video|audio|document
    mime                    TEXT,
    byte_size               INTEGER,
    duration_ms             INTEGER,
    width                   INTEGER,
    height                  INTEGER,
    page_count              INTEGER,
    title                   TEXT,
    source                  TEXT,
    session_id              TEXT,
    scope                   TEXT,
    captured_at             TEXT,
    captured_at_precision   TEXT DEFAULT 'unknown',
    understanding_status    TEXT DEFAULT 'pending',
    provider                TEXT,
    provider_model          TEXT,
    archive_locator         TEXT,
    metadata                TEXT DEFAULT '{}',
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

`asset_id` is deterministic, never random: `sha256:<hash>` when the bytes were seen, otherwise `ref:<sha256(ref_kind + "\x1f" + normalized_ref_value)>`. Determinism is what makes re-ingest and archive re-push idempotent without a lookup round trip.

> **Correction (2026-07-30), two rules an earlier draft omitted.**
>
> **1. `ref_kind` must be inside the digest.** The draft specified `ref:<sha256(normalized_ref_value)>`. But uniqueness is `(ref_kind, ref_value)` — the same `ref_value` under two different kinds is two legal rows, and both would hash to the same `asset_id`, colliding on the PRIMARY KEY.
>
> **2. `asset_id` is assigned at first registration and is never rewritten.** Read literally, the rule above gives an asset first seen by reference (`ref:…`) and later seen with its bytes (`sha256:…`) a *new* id, orphaning every moment already written against the old one. The `sha256:` form is used only when the hash is known at first registration; a hash that arrives later fills in `content_hash` via `UPDATE` and leaves `asset_id` untouched.
>
> **Consequence for the two unique indexes below.** `idx_media_ref` says identity is the reference; `idx_media_hash` says identity is the bytes. One file reachable by both a URL and a local path, both resolving to the same known hash, cannot be two rows. Registering the second reference returns the **existing** `asset_id` and records the new reference under `metadata.alt_refs`, rather than raising `IntegrityError`.

`understanding_status` is one of `pending | ok | partial | unavailable | refused`. `unavailable` is rung 4 of the RFC 0002 §3.3 degradation ladder and is a success state.

`archive_locator` is the opaque handle described in RFC 0004 §2. Mnemosyne never parses it.

Indexes, per §1.6:

```sql
CREATE UNIQUE INDEX IF NOT EXISTS idx_media_ref
    ON media_assets(ref_kind, ref_value);
CREATE UNIQUE INDEX IF NOT EXISTS idx_media_hash
    ON media_assets(content_hash) WHERE content_hash IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_media_modality ON media_assets(modality);
CREATE INDEX IF NOT EXISTS idx_media_session ON media_assets(session_id, created_at);
```

The second is a partial unique index, the trick already used at `canonical.py:131-132`, so that many assets may have a NULL `content_hash` (reference-only, bytes never seen) while any known hash is unique.

### 2.2 `media_moments`

One row per semantically tagged span.

```sql
CREATE TABLE IF NOT EXISTS media_moments (
    moment_id       TEXT PRIMARY KEY,
    asset_id        TEXT NOT NULL,
    memory_id       TEXT,
    kind            TEXT NOT NULL,   -- caption|shot|transcript|style|ocr|page|summary
    text            TEXT NOT NULL,
    span_kind       TEXT NOT NULL,   -- whole|time|page|char|box
    t_start_ms      INTEGER,
    t_end_ms        INTEGER,
    page_start      INTEGER,
    page_end        INTEGER,
    char_start      INTEGER,
    char_end        INTEGER,
    bbox            TEXT,
    speaker         TEXT,
    confidence      REAL DEFAULT 1.0,
    ordinal         INTEGER,
    span_key        TEXT NOT NULL,
    provider        TEXT,
    provider_model  TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX  IF NOT EXISTS idx_moment_asset  ON media_moments(asset_id, ordinal);
CREATE INDEX  IF NOT EXISTS idx_moment_time   ON media_moments(asset_id, t_start_ms);
CREATE INDEX  IF NOT EXISTS idx_moment_memory ON media_moments(memory_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_moment_span
    ON media_moments(asset_id, kind, span_key);
```

`span_key` is a canonical string form of whichever span columns are populated, **plus the speaker**. Paired with `INSERT OR IGNORE`, the unique index makes re-ingest and archive re-push idempotent. This is deliberately the same contract `AnnotationStore.import_all` received in #538, where UNIQUE collisions on `(memory_id, kind, value)` are skipped rather than aborting the import.

> **Correction (2026-07-30): `speaker` must be part of `span_key`, or multi-speaker audio is silently lossy.**
>
> §2.3 below says an audio moment is `span_kind='time'` with `t_start_ms`, `t_end_ms`, **and `speaker`**. But an earlier draft defined `span_key` from the *geometric* span alone. Two speakers talking over the same window — a crosstalk segment, an interviewer and interviewee, any diarized overlap — then produce **identical** `span_key`s under the same `kind='transcript'`, and `INSERT OR IGNORE` drops the second one. No error, no warning, no count. The lossiness is invisible precisely in the case diarization exists to handle.
>
> This redefines `span_key` from "canonical form of the span" to **"canonical form of the span identity"**, which is a real semantic change and is why it is called out rather than quietly patched.

**`span_key` format, normative.**

```
span_key := "v1:" <span_kind> [ ":" <payload> ] [ ":spk=" <speaker-slug> ]
```

| `span_kind` | payload | example |
|---|---|---|
| `whole` | omitted | `v1:whole` |
| `time` | `{t_start_ms}-{t_end_ms or "na"}` | `v1:time:90000-96000` |
| `page` | `{page_start}-{page_end or page_start}` | `v1:page:3-3` |
| `char` | `{char_start}-{char_end}` | `v1:char:0-512` |
| `box` | four normalized floats, `f"{v:.6f}"`, comma-joined | `v1:box:0.100000,0.200000,0.300000,0.400000` |

Rules: the `v1:` prefix is **mandatory** — without a version tag, a future format change silently produces both old- and new-form rows for the same span and the unique index cannot tell them apart; with it, the duplication is greppable and a rewrite migration can scope itself. Integers are coerced with `int()` and `bool` is rejected explicitly (`True` would otherwise become `1`). Floats are never rendered with `str()`, which is repr- and locale-sensitive. An unparseable `bbox` falls back to `"h" + sha256(canonical_json)[:16]` so the function stays total. The speaker slug is whitespace-collapsed, case-folded, truncated to 64 chars, and appended **last**, so rows without a speaker are byte-identical to what the pre-correction format would have produced.

### 2.3 One column set, not one table per modality

**Recommendation: a single `span_kind`-discriminated column set.**

| Modality | `span_kind` | Populated columns |
|---|---|---|
| Image (whole) | `whole` | none |
| Image (region) | `box` | `bbox` |
| Video | `time` | `t_start_ms`, `t_end_ms` |
| Audio | `time` | `t_start_ms`, `t_end_ms`, `speaker` (also folded into `span_key`, per §2.2) |
| Document (page) | `page` | `page_start`, `page_end` |
| Document (offset) | `char` | `char_start`, `char_end` |

**Pros.** NULL columns are free in SQLite. The recall sub-select in §4.2 stays a single predicate. The in-repo precedent for a discriminated nullable field plus a precision qualifier is `event_date` / `event_date_precision` (`beam.py:1228`, `:1231`).

**Cons.** No database-level guarantee that a `time` moment has `t_start_ms` populated. Accepted, and enforced in `MediaStore.add_moments` instead, consistent with the application-layer integrity posture from §1.6.

Four per-modality tables would multiply the recall predicate, the filter touch points in §4.3, and the archive envelope in RFC 0004 §2 by four, to buy a constraint the repo does not enforce anywhere else.

### 2.4 Soft references, checked not enforced

`media_moments.asset_id` and `.memory_id` are convention-only per §1.6. Writers use the existence guard already used by `_store_working_embedding` (`beam.py:2216`) rather than an FK:

```sql
WHERE EXISTS (SELECT 1 FROM working_memory WHERE id = ?)
```

Integrity is *reported*, not enforced: add an orphaned-moment count to `mnemosyne doctor` and to `mnemosyne_diagnose`.

> **Correction (2026-07-30): the two orphan kinds are different health states and must not be reported alike.**
>
> A moment whose **`asset_id`** has no asset is real corruption — nothing in the design should ever produce it, and it warrants a Finding.
>
> A moment whose **`memory_id`** no longer resolves is the *expected steady state*. §3.3 Cost 2 says so itself: consolidation may summarize the memory row away, and `media_moments.text` remains authoritative. Working memory is also trimmed on a cap (`_trim_working_memory`, called from `remember`). Emitting a warning for these means `mnemosyne doctor` cries wolf at every healthy media user after their first `sleep()`, which trains people to ignore the check that catches the real corruption.
>
> **Count both; emit a Finding only for asset-orphans.**
>
> A second constraint the draft did not state: `mnemosyne doctor` opens the database **read-only** (`open_readonly_doctor_db`). The media inspection must therefore take a `not_configured` branch when the tables are absent — the same shape as the existing unverifiable-store branch — and must never attempt DDL on that connection.

---

## 3. A moment is a first-class memory row

### 3.1 Decision

**Each retained moment is written to `working_memory` as an ordinary row, with `media_moments.memory_id` bound to it. There is no `vec_moments` table, and `vec_facts` does not get a writer.**

*Philosophy: the cheapest new subsystem is the one that is not a new subsystem.*

### 3.2 Why

Per §1.4, a working-memory row already receives the dense voice, FTS, decay, tier degradation, consolidation, the veracity multiplier, sync events, and reindex coverage. Per §1.3, a new vec0 table costs four whitelist edits, a rowid-alignment discipline, and a permanently larger dim-mismatch surface.

There is a third benefit that only becomes visible in §4: because moments are ordinary rows, the media recall voice returns ordinary `memory_id` values, so RRF fusion in `_combine_voices` (`polyphonic_recall.py:727`) needs no id-space translation. A sidecar-only design would require mapping moment ids into memory ids before fusion, or fusing two incompatible id spaces.

Reusing `vec_facts` is rejected outright: it is the cautionary tale from §1.3, not an opportunity. Giving it a writer would entangle an unrelated debt with this feature, and its shape is fact-triple-oriented (see `fts_facts` at `beam.py:1194-1207`).

### 3.3 Honest costs, and the mitigations

**Cost 1: working-memory volume.** A ten-minute video at shot granularity is dozens of rows. `WORKING_MEMORY_MAX_ITEMS` defaults to 10,000 (`beam.py:276`), so a handful of videos could measurably shift eviction pressure.
*Mitigation:* cap retained moments at `MNEMOSYNE_MODALITY_MAX_MOMENTS` (default 12, RFC 0002 §3.4). Shot-level density is opt-in per call, not the default.

**Cost 2: consolidation may summarize moment text away.** Sleep compresses working into episodic.
*Mitigation:* `media_moments.text` remains authoritative. A moment whose memory row was consolidated or evicted is rehydratable by re-inserting and rebinding `memory_id`. The sidecar is the source of truth; the memory row is the index entry.

**Cost 3: moments are misclassified.** `classify_memory` (`beam.py:3267`) will label a caption an "observation".
*Mitigation:* see §3.4.

### 3.4 Three prerequisite changes to the ingest path

> **Correction (2026-07-30).** This section listed two prerequisites. There is a third, and it is the most dangerous of the three: see prerequisite 3. Line anchors throughout this section have also been corrected; they had drifted by two lines.

**Prerequisite 1: `remember()` needs a `memory_type` override.** `BeamMemory.remember` (`beam.py:3212`) accepts `content`, `source`, `importance`, `metadata`, `valid_until`, `scope`, `memory_id`, `extract_entities`, `extract`, `veracity`, and `trust_tier`. There is **no `memory_type` parameter**: the value is derived locally from `classify_memory` at `beam.py:3266-3272`. Add `memory_type: Optional[str] = None` as an explicit override so a moment can declare itself `artifact` (`core/typed_memory.py:51`, `MemoryType.ARTIFACT`, described as "references to documents, code, external resources" — verified present). The plumbing is already friendly: the dedup-update path uses `memory_type = COALESCE(?, memory_type)` at `beam.py:3301`.

An explicit value **wins outright and short-circuits the classifier**; an unrecognized label warns and falls back to classification rather than storing `NULL`, so a typo degrades to today's behaviour instead of stripping the type. Append the parameter at the **end** of the signature — positional callers exist throughout the test suite, and a `*` separator would break them.

*Estimated: ~15 lines in `beam.py` including the label-clamping helper, plus the same parameter on the `core/memory.py` facade and per-item parity in `remember_batch`.*

**Prerequisite 2: `remember_media()` must never route a URI through `content`.** Per §1.2, `sanitize_content` is invoked on the content string at `beam.py:3251`, and a `data:` URI is rewritten into a blob stub. The media reference therefore travels as a **parameter**, and the `content` of a moment row is always the moment's descriptive text. This is a hard API constraint, not a style preference, and deserves a regression test.

A second, sharper reason the draft missed: **`metadata_json` is never sanitized.** `sanitize_content` only ever inspects `content` (`beam.py:3251`, `beam.py:3523`, `memory.py:358`). A `data:` URI that arrives as a reference and gets copied into `media_assets.ref_value` or into `metadata["media"]["ref"]` therefore writes megabytes of base64 into two uncapped `TEXT` columns, with no size cap and no entropy check — precisely the outcome the size-cap and entropy branches exist to prevent, reached by a path they do not watch. The fix is to convert a `data:` URI to a `blob://` reference **at the door**, before any storage. The regression test must assert not only that no `content` contains `data:`, but that no `metadata_json` contains `base64` and that every written `metadata_json` stays small.

**Prerequisite 3: `remember()` needs a `dedupe` opt-out, or moments bind to the wrong asset.** `_find_duplicate` (`beam.py:3177`, called at `beam.py:3275`) matches on `(session_id, content)` exactly and returns the existing row's id. Moment captions collide on identical text far more often than prose does: "a black frame", "a screenshot of a terminal window", the same slide appearing in two recordings, any boilerplate title card. Two such moments **from different assets in one session** therefore produce **one** `working_memory` row, and the second `media_moments` row binds its `memory_id` to the **first asset's** memory.

The consequences are all silent. Recall answers "which video mentioned X" with the wrong video. The §4.4 enrichment join returns two `media` blocks for one result with no way to choose between them. The §4.3 `asset_ref` filter matches the wrong asset. Nothing raises, nothing is logged, and the row counts all look correct.

Add `dedupe: bool = True` and guard the `_find_duplicate` call with it; `remember_media()` passes `dedupe=False`. Note that `remember_batch` already bypasses dedup entirely (`beam.py:3531` generates an id unconditionally), so only the single-write path is exposed. Note also that leaving dedup on has a second-order hazard even when the collision is benign: the `COALESCE` at `beam.py:3301` means an explicit `memory_type='artifact'` **retypes the colliding row**, silently converting a user's conversational note into an artifact. `dedupe=False` removes both exposures at once.

*This is the highest silent-wrong-answer risk in this RFC.* It ranks above the enhanced-recall cache key in §4.3, because the cache key produces a stale answer while this one produces a confidently wrong attribution.

---

## 4. Three time axes, and recall

### 4.1 Media-internal time is not wall-clock time

`t_start_ms` is relative to its asset. "t=90s" has no meaning without knowing which asset, so it is **not comparable across assets** and must never enter `event_date` or a date-range filter. `event_date` is indexed for range queries (`beam.py:1228-1231` and the indexes immediately following) and consumed by `_temporal_voice` (`polyphonic_recall.py:617`); polluting it with media offsets would corrupt temporal recall for every user.

Three distinct times, never conflated:

| Time | Where | Meaning |
|---|---|---|
| `media_assets.captured_at` (+ precision) | new | wall clock of the recording itself |
| `working_memory.timestamp` | existing | when Mnemosyne ingested it |
| `media_moments.t_start_ms` | new | offset within the asset, asset-relative only |

The one legitimate bridge is `captured_at + t_start_ms -> event_date`, valid only when `captured_at_precision == 'exact'`. It goes behind `MNEMOSYNE_MEDIA_TIME_BRIDGE`, **default off** in v1. It is genuinely useful ("what did I see the afternoon of the 12th") and genuinely wrong when `captured_at` is approximate, so it is opt-in.

### 4.2 Recall integration: a polyphonic voice, not a linear weight

**Recommendation: add a fifth polyphonic voice, plus read-only enrichment on the linear path. Do not add a term to the linear scorer.**

Rejected alternative, linear weight: `_normalize_weights` (`beam.py:1400`) normalizes vector, FTS, and importance weights to sum to 1.0, reading them straight from `os.environ`. A fourth term renormalizes the other three and shifts ranking for every existing user and every ranking test, to serve queries that are mostly not about media.

Rejected alternative, episodic bonus: the bonus ladder at `beam.py:6237-6299` (graph bonus capped at 0.08, fact bonus at 0.1) could host an additive media bonus, but it would fire for every artifact memory regardless of whether the query is about media. That is noise, not signal.

Polyphonic is the seam that was designed for this. `_combine_voices` (`polyphonic_recall.py:727`) fuses ranked lists with Reciprocal Rank Fusion at `RRF_K = 60` (`:734`), which absorbs a new list without renormalizing anything. `_env_disabled` (`:58`) provides a free kill switch. `PolyphonicResult.voice_scores` (`:83`) provides per-signal provenance at no cost.

`_media_voice(query, query_embedding)`:

- FTS5 over moment text, via a new `fts_moments` external-content table with AI and AD triggers, copying the `fts_facts` idiom at `beam.py:1194-1207`.
- An exact-match boost when the query contains a URL or a content hash matching `media_assets.ref_value`. This is what makes "what do I know about `<url>`" work as a literal lookup.
- Emits `RecallResult(memory_id=<moment.memory_id>, voice="media", ...)` (`polyphonic_recall.py:69-75`), which fuses directly per §3.2.

Note the existing voice weights are `vector 0.35, graph 0.25, fact 0.25, temporal 0.15` (`polyphonic_recall.py:128-133`). Introducing a fifth entry rebalances all four. Ship with `MNEMOSYNE_VOICE_MEDIA` **off** for one full release so the change is measurable rather than surprising.

### 4.3 The modality filter, and its nine touch points

New recall parameters: `modality: Optional[str]` and `asset_ref: Optional[str]`.

Implement the predicate as a **correlated sub-select, not a join**, so that the six duplicated SELECT column lists in the recall path stay untouched:

```sql
EXISTS (SELECT 1
        FROM media_moments mm
        JOIN media_assets  ma ON mm.asset_id = ma.asset_id
        WHERE mm.memory_id = working_memory.id
          AND ma.modality  = ?)
```

| # | Touch point | Location | Note |
|---|---|---|---|
| 1 | `recall()` signature | `beam.py:5451` | add after `memory_type` |
| 2 | Working-memory WHERE | `beam.py:5695` | the `memory_type` clause is the literal template |
| 3 | Episodic WHERE | `beam.py:6197` | same shape |
| 4 | `_recall_polyphonic` signature and call site | `beam.py:7235`, called from `:5531` | |
| 5 | Polyphonic post-filter | `beam.py` `_polyphonic_row_passes_filters` | **batch-prefetch** an `id -> modality` map before the loop; do not query per row |
| 6 | Explain-trace `filters` dict | `beam.py:5562` | safe: `RecallExplainTrace` prunes `None` values |
| 7 | Enhanced-recall cache key | `beam.py:6717` | **see the warning below** |
| 8 | MCP surface | `tool_schemas.py` `RECALL_SCHEMA`, `mcp_tools.py`, and **both** Hermes copies | `tests/test_hermes_provider_parity.py` fails otherwise |
| 9 | Generated docs | `scripts/generate-docs.py` | it hand-duplicates the schemas as literals |

**The highest silent-wrong-answer risk in this RFC is touch point 7.** `_enhanced_recall_cache_key` (`beam.py:6717`) canonicalizes `recall_kwargs` wholesale, so a new parameter enters the digest only if it arrives inside `kwargs` rather than as a named argument. Verify that plumbing, and **bump the key's version prefix regardless**. Without the bump, cache entries written before the filter existed will be served to modality-filtered queries, returning unfiltered results with no error. Note that `invalidate()` already calls `_invalidate_query_cache` (`beam.py:3995`) for both branches, so eviction on mutation is handled; this is purely about the schema of the key.

### 4.4 Read-only enrichment on every path

Independent of filtering or scoring, every recall result whose memory id has a bound moment gains:

```json
{"media": {"asset_ref": "...", "modality": "video", "archive_locator": "...",
           "span": {"kind": "time", "t_start_ms": 90000, "t_end_ms": 96000}}}
```

Fetched with one batched `WHERE memory_id IN (...)` query, following the batching pattern already used for the `memory_type` backfill in the enhanced-recall path. This is what lets a client deep-link to a timestamp. **Recall never fetches bytes** (RFC 0004 §2.2).

---

## 5. Phasing

| Phase | Work | Est. LOC | Chief risk |
|---|---|---|---|
| **0** | `core/media.py` registry and `MediaStore`; `MediaStore` constructed in `BeamMemory.__init__` beside `self.canonical` (`beam.py:3043`), **not** in `init_beam` — see the correction in §2; `mnemosyne doctor` orphan count | ~420 + ~550 tests | Near zero. No vec tables, no recall touch, nothing in a hot path |
| **1** | RFC 0002 provider protocol and Atlas adapter; `remember_media()`; `memory_type` override on `remember()` | ~640 | Process-global not reset in conftest; a socket opening when disabled; a URI leaking into `content` (§3.4) |
| **2** | Read-only recall enrichment (§4.4) | ~80 | The result dict gains a key: audit every consumer, including both Hermes copies and the sync serializer |
| **3** | `modality` / `asset_ref` filter across the nine touch points (§4.3) | ~150 | The cache key (§4.3); the generated-docs literals |
| **4** | `fts_moments` and `_media_voice` (§4.2) | ~180 | Rebalances the four existing voice weights. Ship with `MNEMOSYNE_VOICE_MEDIA` off for one release |
| **5** | Video shot segmentation, audio diarization, voice profiles | separate release | See the concurrency rule below |

### 5.1 Concurrency rule: no threads in core

`mnemosyne/core/beam.py` has **no access lock**. It relies solely on thread-local connections. The `_beam_access_lock` that serializes Beam access against the auto-sleep daemon lives in the Hermes provider only (`hermes_memory_provider/__init__.py`, duplicated in `integrations/hermes/src/mnemosyne_hermes/`).

A background media-processing thread added inside `core/` would therefore run with no serialization whatsoever, which is precisely the shape of the WAL-checkpoint-mid-statement crash behind issues #498 and #520.

Therefore: media understanding is **synchronous inside the caller's `remember_media()`**, or it is a **separate process** (`mnemosyne media process --pending`, scanning `understanding_status='pending'`). A host that wants asynchrony owns the thread and the lock, exactly as the Hermes provider already does for auto-sleep.

### 5.2 Must not ship in the same release

- **Any new vec0 table together with an embedding model or dimension change.** Per §3.1, no new vec0 table at all.
- **The modality filter (phase 3) and `_media_voice` (phase 4) together.** One changes filter semantics, the other changes ranking. Shipped together, a recall regression is unattributable.
- **A `vec_facts` writer alongside media work.** Independent debts.
- **Media-native embeddings in phases 0 through 4.** See RFC 0002 §2.

### 5.2b Config keys this RFC introduces must be registered

> **Correction (2026-07-30).** This RFC introduces two config keys — `MNEMOSYNE_MEDIA_TIME_BRIDGE` (§4.1) and `MNEMOSYNE_VOICE_MEDIA` (§4.2) — and tabulates neither. RFC 0002 §3.4's key table covers only the eight `MNEMOSYNE_MODALITY_*` keys, so a reader following either document alone will miss these.
>
> Each must be added to **`ENV_VAR_MAP` (`config.py:62`)**, **`DEFAULTS` (`config.py:192`)**, and **`CONFIG_DESCRIPTIONS` (`scripts/generate-docs.py`)**, then `docs/api/*.mdx` regenerated. A key absent from `ENV_VAR_MAP` is YAML-only and cannot be set from the environment at all; a key absent from `CONFIG_DESCRIPTIONS` **fails the `docs-check` CI job**. The Python type of the value in `DEFAULTS` drives environment-variable coercion, so both must be registered as `bool`, not `"false"`.
>
> Register each key in the phase that consumes it (the time bridge with phase 3, the media voice with phase 4), not upfront — declaring a key with no reader is the same anti-pattern as declaring an interface with no writer (RFC 0002 §1.4).

### 5.3 CI constraint

CI runs `pytest tests/ -v` with `MNEMOSYNE_NO_EMBEDDINGS=1`. Every media test must pass with embeddings off. The caption-to-text design (RFC 0002 §2) satisfies this, because moments still land in FTS. Ruff runs `--select E9,F63,F7,F82` only.

### 5.4 Known gaps this RFC does not close

> **Added 2026-07-30.** Three gaps surfaced during implementation review that no section of this RFC addresses. They are recorded here rather than silently inherited. **All three should be resolved before phase 2**, because phase 2's enrichment join builds directly on top of them.

**1. `media_assets` and `media_moments` do not sync.** `memory_events` (`beam.py:745`) is what propagates rows to peer devices, and nothing emits events for either media table. A synced peer therefore receives the caption *memory row* and none of its span data, `archive_locator`, or asset linkage. This directly contradicts §3.3 Cost 2, which calls `media_moments.text` "authoritative" and the memory row a mere "index entry": after a sync, the index entry arrives and the authority does not. It also weakens RFC 0004 invariant 5, which requires deletion to reach synced devices. The failure is silent and degraded rather than loud and broken — timestamps and deep links simply vanish on the second device — which is exactly the kind of gap that goes unreported for a long time.

**2. `mnemosyne export` omits both tables.** Backups will silently exclude all media state. Same class of gap as (1), and it should be fixed in the same change.

**3. Deletion and cascade semantics are unspecified.** RFC 0004 §4 invariant 5 requires `forget_asset(asset_id, cascade=True)` to remove moments, their memory rows, their `memory_embeddings` and `vec_working` entries, and to emit DELETE events. Neither RFC says what happens in the *other* direction: when a moment's `working_memory` row is forgotten by the user, evicted by `_trim_working_memory`, or summarized away by consolidation. §3.3 Cost 2 asserts moments are "rehydratable" and nothing implements rehydration.

Implementations should **not** ship a `cascade=` parameter before there is an ingest path creating the bindings it would cascade over — that is a dead branch with a falsely reassuring test. Phase 0 ships sidecar-only deletion and says so in the docstring.

---

## 6. Gap analysis

### 6.1 What exists

| Capability | Status | Location |
|---|---|---|
| Content-addressed byte storage | ⚠️ write-only | `core/content_sanitizer.py:91`, `:103` |
| A `blob://` URI scheme | ⚠️ no reader | `content_sanitizer.py:126` |
| Wall-clock event time with precision | ✅ | `beam.py:1228`, `:1231` |
| Additive-column migration idiom | ✅ | `beam.py:658-662` |
| Subsystem module with idempotent init | ✅ | `core/annotations.py:124` |
| Partial unique index idiom | ✅ | `core/canonical.py:131-132` |
| Idempotent-insert-on-unique contract | ✅ | `annotations.py:159` and #538 |
| Recall filter template | ✅ | `beam.py:5695`, `:6195` |
| Pluggable recall voice with kill switch | ✅ | `polyphonic_recall.py:58`, `:727` |
| A span or offset model | ❌ | nowhere |
| A modality dimension | ❌ | nowhere |
| Any media ingest entry point | ❌ | nowhere |

### 6.2 What is missing

**Gap 1: no asset identity.** There is no way to say "these two memories are about the same video".
**Impact:** every mention of a piece of media is an unrelated text row. No deduplication, no "show me everything about this file", no deletion by asset.

**Gap 2: no span.** Text can be stored about a video but not *located* within it.
**Impact:** the core use case, pinpointing the moment a topic is mentioned, is unrepresentable. This is the gap the RFC exists to close.

**Gap 3: blobs are write-only.** `sanitize_content` quarantines bytes and nothing can retrieve them.
**Impact:** any byte already extracted by the entropy or size-cap rules is currently unreachable. Closed by RFC 0004 §2.1.

**Gap 4: `remember()` cannot declare a memory type.**
**Impact:** captions would be classified as conversational observations, polluting type-aware decay and consolidation.

---

## 7. Architecture

```
   EXPLICIT USER ACTION ONLY  (no watcher, no polling, no directory scan)
   mnemosyne media add | mnemosyne_media_register | remember_media()
                          │
                          v
   ┌──────────────────────────────────────────────────────────────────┐
   │ media_assets            [NEW, core/media.py]                     │
   │ asset_id (deterministic) · ref_kind/ref_value · modality         │
   │ captured_at + precision · understanding_status · archive_locator │
   │ *** NO BLOB COLUMN. Bytes are never stored here. ***             │
   └──────────┬───────────────────────────────────────────────────────┘
              │ call_modality_describe()   [RFC 0002]
              │   -> DescribeResult{ moments: MomentDraft[] }  ... TEXT
              v
   ┌──────────────────────────────────────────────────────────────────┐
   │ media_moments           [NEW, core/media.py]                     │
   │ span_kind: whole | time | page | char | box                      │
   │ t_start_ms/t_end_ms · page_* · char_* · bbox · speaker           │
   │ UNIQUE(asset_id, kind, span_key) + INSERT OR IGNORE => idempotent│
   │ memory_id ──────────────┐  (soft ref, no FK per #503)            │
   └─────────────────────────┼────────────────────────────────────────┘
                             v
   ┌──────────────────────────────────────────────────────────────────┐
   │ working_memory          [EXISTING, unchanged schema]             │
   │ memory_type='artifact' · content = the moment's TEXT             │
   │ inherits: vec_working · fts_working · decay · sleep ·            │
   │           veracity · memory_events(sync) · mnemosyne reindex     │
   └─────────────────────────┬────────────────────────────────────────┘
                             v
   ┌──────────────────────────────────────────────────────────────────┐
   │ recall                                                           │
   │  phase 2: enrichment  -> result["media"] = {ref, span, locator}  │
   │  phase 3: filter      -> EXISTS(...) correlated sub-select        │
   │  phase 4: _media_voice -> RRF k=60 (polyphonic_recall.py:734)     │
   │  linear scorer weights UNTOUCHED (beam.py:1400)                   │
   └──────────────────────────────────────────────────────────────────┘

   TIME AXES, never conflated:
     captured_at (wall clock)  |  timestamp (ingest)  |  t_start_ms (asset-relative)
     bridge captured_at + t_start_ms -> event_date is OPT-IN, default off

   GAP LEFT OPEN: bytes. Mnemosyne holds the reference and the text.
        The archive holds the bytes. See RFC 0004.
```

---

## 8. Key file reference

| File | Purpose | Lines |
|---|---|---|
| `mnemosyne/core/media.py` | **new.** DDL, `MediaStore`, span canonicalization | ~350 |
| `mnemosyne/core/beam.py` | 3 lines of init; `memory_type` param; `remember_media()`; recall touch points 1-7 | +~300 |
| `mnemosyne/core/polyphonic_recall.py` | `_media_voice`, fifth weight entry | +~120 |
| `mnemosyne/core/memory.py` | facade passthrough for the new params | +~15 |
| `mnemosyne/tool_schemas.py`, `mcp_tools.py` | recall filter params, media tools | +~80 |
| `hermes_memory_provider/`, `integrations/hermes/src/mnemosyne_hermes/` | **both copies** or parity tests fail | +~60 |
| `scripts/generate-docs.py` | hand-duplicated schema literals | +~20 |
| `tests/test_media_*.py` | new | ~600 |

---

## 9. Conclusion

The moment index is **additive and unusually low-risk for its ambition**: phase 0 introduces two tables, three lines in `init_beam`, and touches no recall path at all. The privacy-preserving design the product wants, reference hashes plus semantically tagged spans, happens to also be the cheapest design, because text moments reuse the entire existing retrieval engine.

Four gaps, in priority order: **identity** (no asset concept), **location** (no span model, the reason this RFC exists), **retrieval** (no modality signal or filter), and **hydration** (blobs are write-only, closed by RFC 0004).

Recommended sequencing: phase 0 and 1 land the capability, phase 2 makes it visible, phase 3 makes it filterable, phase 4 makes it rank. Phases 3 and 4 must not ship together.

**The key insight:** a moment should be an ordinary `working_memory` row. That single decision eliminates a vec0 table and its four whitelist edits, eliminates the id-space translation that RRF fusion would otherwise need, and inherits decay, consolidation, sync, and reindex for nothing. Everything expensive about this feature is expensive only if moments are given their own storage; make them memories, and the remaining work is two sidecar tables and a filter.
