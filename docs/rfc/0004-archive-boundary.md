# RFC 0004: The Archive Boundary, and a Companion Archive Product

**Status:** Draft
**Author:** abdiisan
**Related issue:** TBD
**Target version:** Part 1, next MAJOR. Part 2, separate repository
**Related:** RFC 0002 (modality providers), RFC 0003 (media moments)

---

## 0. Summary and structure

A community proposal suggested adding a "wiki" or "library" layer to Mnemosyne so it could hold and browse documents and media directly. Building it into the engine would be a mistake: Mnemosyne's entire competitive position is being a single-file, near-zero-dependency, sub-millisecond memory layer, and a document archive has the opposite characteristics. It is large, it is I/O bound, it wants a UI, and it wants to be browsed by humans rather than queried by agents.

The alternative is a boundary. **Mnemosyne is the librarian: it holds the semantic index, the associations, and the reasoning. A companion product is the archive: it holds the raw heavy files.** The librarian knows where everything is and what it means. The archive knows how to store and serve it.

This document has two parts with different statuses:

- **Part 1 (§1 through §4) is normative.** It specifies the Mnemosyne-side contract. It must be reviewable, implementable, and mergeable **whether or not the companion product is ever built**, and it must let a third party implement the archive side.
- **Part 2 (§5 through §7) is informative.** It is the plan for the companion product, recorded here because the boundary was designed alongside it. Once that product has a repository, Part 2 moves there and this document keeps only a pointer.

---

# Part 1: The contract (normative)

## 1. The boundary in one sentence

**The brain stores text and spans. The archive stores bytes.**

Every ambiguity below resolves against that sentence.

| Concern | Owner |
|---|---|
| Reference hash, mime, duration, dimensions | brain (`media_assets`, RFC 0003 §2.1) |
| Semantic text about a span | brain (`media_moments`, RFC 0003 §2.2) |
| Embeddings, recall, ranking, associations | brain |
| The actual bytes | archive |
| Thumbnails, transcodes, derived renditions | archive |
| Human-browsable structure and UI | archive |
| Deciding *what is worth remembering* | brain |
| Deciding *how to store it durably* | archive |

Mnemosyne has no dependency on the archive, and never will. An archive is an optional accelerator for byte access, not a component.

## 2. `ContentResolver`: generalizing `blob://`

### 2.1 The existing scheme, and its missing half

`core/content_sanitizer.py` already defines a content-addressed byte store and a URI scheme. `_store_blob` (`:91`) writes to `_blob_root()` (`:32`), and `sanitize_content` (`:103`) records `blob://sha256/<hash>` into memory metadata (`:126`).

It has **no reader.** A grep for `blob_ref` or `blob://` outside `content_sanitizer.py` and `tests/` returns nothing. Bytes extracted by the size-cap rule (`:141-145`) or the entropy rule (`:157-162`) are currently unreachable by any code path in the repository. This is a real hole independent of anything multimodal, and closing it is roughly forty lines.

### 2.2 The protocol

New module `mnemosyne/core/resolvers.py`, following the registry shape of `core/llm_backends.py` for the same reasons given in RFC 0002 §1.2.

```python
class ContentResolver(Protocol):
    """Resolves an opaque content URI to metadata or bytes."""

    name: str
    schemes: frozenset[str]          # {"blob"} | {"archive"} | {"file"} | ...

    def head(self, uri: str) -> Optional[ResolvedMeta]:
        """Cheap existence and metadata probe.

        None means "not my scheme, or not a URI I can parse".
        ResolvedMeta(exists=False) means "mine, and gone".
        """

    def open(self, uri: str) -> Optional[BinaryIO]:
        """Lazy byte stream. None if unavailable."""

    def presign(self, uri: str, *, ttl_s: int = 300) -> Optional[str]:
        """A short-lived URL a third party can fetch. None if unsupported."""
```

`ResolvedMeta` carries `exists`, `byte_size`, `mime`, `content_hash`, `etag`.

> **Correction (2026-07-30), two signature fixes.**
>
> **1. `head` must distinguish "not mine" from "gone".** The original docstring said only "None if unresolvable", collapsing two states that §3.2.1 depends on separating: `mnemosyne doctor` validating that a reference is still live needs "the file is missing" to be a *different answer* from "no resolver handles this scheme". `ResolvedMeta.exists` exists precisely to carry that, and returning `None` for both makes it unreachable.
>
> **2. `ttl_s` is keyword-only with a default.** As drafted it was positional, contradicting the `llm_backends` convention this section says it is copying — one positional argument, everything else keyword-only (`llm_backends.py:37-47`).

**`presign` is the load-bearing method.** It is how a cloud vision provider (RFC 0002) reaches bytes that Mnemosyne does not want to read, buffer, or proxy. Without it, either the brain streams every megabyte through its own process to hand to a provider, or media understanding only works for local files. With it, "the archive is a separate product" becomes structurally true rather than aspirational: the archive serves bytes directly to whoever needs them, and the brain only ever passes URIs around.

### 2.3 Ship `BlobResolver` in the same change

A `BlobResolver` over `content_sanitizer._blob_root()`, giving the existing blob store its first reader. Two details constrain `head`:

- **`mime` is sniffed or `None`, on all three branches.** `head` must not guess.
- `presign` returns `None`. Local files have no presignable URL. Callers must handle `None` by falling back to `open`, or by skipping the provider call.

> **Correction (2026-07-30): the original wording of the first bullet was wrong, in a way that would send an implementer looking for something that does not exist.**
>
> It read: "`mime` is populated only on the `data:` URI branch (`content_sanitizer.py:128`). For size-cap and entropy blobs, `head` must sniff the bytes or return `mime=None`." That implies the data-URI branch gives `BlobResolver` a mime it can return. It does not. That mime is written into the **memory row's `metadata_json`** (`beam.py:3251-3256`, `memory.py:358-364`), not into anything addressable by `blob://sha256/<hash>`. The blob store on disk holds bare bytes at a hash path with no sidecar and no filename.
>
> So the rule is uniform: **sniff-or-`None` on all three branches.** Sniffing means reading a small prefix and matching magic bytes; it explicitly excludes `mimetypes.guess_type` (there is no filename to guess from — the path *is* the hash) and excludes defaulting to `application/octet-stream`, which is a guess wearing a disguise.
>
> A resolver that joined against `working_memory.metadata_json` to recover the declared mime is conceivable but out of scope: it would put a database dependency inside a filesystem resolver, and the two can disagree.

**One thing this section does not say, and must.** Nothing here specifies **who registers `BlobResolver`**. Unaddressed, the forty lines that exist to close the "blobs are unreachable" bug close nothing, because no code path ever constructs one. The registry in §2.4 therefore resolves in two tiers: an explicit registration dict first, then a built-in default map containing `{"blob": BlobResolver}`. This makes byte access work with zero configuration, and it makes `clear_content_resolvers()` restore the built-in rather than *remove* blob access — which matters, because §2.4 mandates calling exactly that in `tests/conftest.py` before every test.

### 2.4 Registration and absence

Process-global registry keyed by scheme: `set_content_resolver(resolver)`, `get_resolver(scheme)`, `clear_content_resolvers()`. Per RFC 0002 §5, these globals **must** be reset in `tests/conftest.py` beside the existing `llm_backends._backend` reset at `:64-69`.

When no resolver is registered for a scheme, `get_resolver` returns `None` and every caller degrades:

- `head` is unknown, so the asset keeps `understanding_status='unavailable'` (RFC 0003 §2.1).
- Moments that already exist **recall perfectly**, because their text is local to Mnemosyne.
- A client shows a reference without a preview.
- Nothing raises. This mirrors `embeddings.available()`.

That last point is the whole design goal restated: **losing the archive costs you previews, not memories.**

## 3. The two directions of traffic

### 3.1 Archive to brain: index handoff

The archive pushes what it has learned. Two new MCP tools, `mnemosyne_media_register` and `mnemosyne_media_add_moments`, carrying a versioned envelope:

```json
{
  "schema": "mnemosyne.media/1",
  "asset": {
    "ref_kind": "sha256",
    "ref_value": "<hex>",
    "modality": "video",
    "mime": "video/mp4",
    "duration_ms": 612000,
    "captured_at": "2026-07-12",
    "captured_at_precision": "day",
    "archive_locator": "<opaque, never parsed by Mnemosyne>"
  },
  "moments": [
    {"kind": "shot", "text": "speaker introduces the memory layer diagram",
     "span_kind": "time", "t_start_ms": 90000, "t_end_ms": 96000,
     "confidence": 0.82}
  ]
}
```

Idempotent on `(ref_kind, ref_value)` and on `(asset_id, kind, span_key)` per RFC 0003 §2.1 and §2.2, so re-pushing an unchanged index is a no-op. `archive_locator` is opaque: Mnemosyne stores it and hands it back, and never interprets it.

`schema` is a version string because this envelope is the one thing a third-party implementer codes against.

### 3.2 Brain to archive: hydration, on exactly two occasions

1. **`head`**, to check that a reference is still live, during `mnemosyne doctor` or an explicit validation pass.
2. **`presign` or `open`**, when a user explicitly views a moment, or asks to re-describe an asset.

**Recall never hydrates.** A recall result carries the reference, the archive locator, and the span (RFC 0003 §4.4). The client decides whether to fetch. This single rule is what keeps recall latency unchanged, keeps Mnemosyne's "no external services required" claim true, and keeps a recall query from generating outbound network traffic about the user's private media.

### 3.3 Third-party implementability

The archive side is: two MCP tool calls to push an index, plus an HTTP contract of `GET /head` and `POST /presign`, plus a declared scheme string. Nothing Python-specific, nothing Mnemosyne-internal. An S3 bucket with a small shim, a NAS daemon, or a full knowledge-base product are all valid archives.

## 4. Privacy invariants, written as testable assertions

The point of reference hashes and moment spans is to get the utility of media memory without the surveillance posture of OS-level ambient capture. Prose promises do not survive refactors, so each invariant is stated as something a test asserts.

| # | Invariant | How it is asserted |
|---|---|---|
| 1 | **No ambient capture.** No watcher, no polling loop, no directory scan, no screenshot timer. | Ingest is reachable only from `remember_media()`, the MCP media tools, or `mnemosyne media add`. Grep-assertable: no `watchdog`, `inotify`, or scheduled scan in the media path |
| 2 | **No provider call without explicit opt-in.** | `MNEMOSYNE_MODALITY_ENABLED` defaults false (RFC 0002 §3.4). With it unset, a test asserts zero sockets opened and zero backend invocations |
| 3 | **The brain never persists media bytes.** | `media_assets` has no BLOB column (RFC 0003 §2.1), and a test asserts the media ingest path never calls `content_sanitizer._store_blob` |
| 4 | **Recall generates no outbound traffic.** | A test asserts no resolver method is called during `recall()` |
| 5 | **Deletion is complete and propagates.** | `forget_asset(asset_id, cascade=True)` removes moments, their memory rows, their `memory_embeddings` and `vec_working` entries via `_wm_vec_delete` (`beam.py:2206`), and emits DELETE rows into `memory_events` (`beam.py:745`) so the deletion reaches synced devices |
| 6 | **The index is legible.** | Everything stored about an asset is human-readable text plus a reference, per RFC 0002 §2. No opaque media vectors |

Invariant 6 is the substantive difference from tools like Microsoft Recall, and it is worth stating plainly in user-facing docs: what Mnemosyne knows about your media is a list of sentences you can read, search, edit, and delete.

---

# Part 2: The companion archive product (informative)

## 5. Positioning

A local-first tool that ingests a directory of Markdown and attachments, parses frontmatter and wikilinks, infers additional edges to produce a self-wiring knowledge graph, stores attachments content-addressed, and speaks the `mnemosyne.media/1` contract from §3.1.

Inspired by the G Brain concept of a self-organizing personal knowledge base. Differentiation must be argued against three specific things, because "another notes app" is not a position:

| Against | Difference |
|---|---|
| **Obsidian** | Links are *inferred*, not hand-authored. The graph wires itself from content, and it speaks the Mnemosyne media contract so an agent can query it semantically |
| **G Brain** | Local-first and open source. Your archive is a directory on your disk, not a hosted account |
| **Mnemosyne itself** | It owns heavy bytes and human-browsable structure, both of which Mnemosyne deliberately refuses. It is for reading; Mnemosyne is for recalling |

## 6. Scope of v1, and what to reuse

**In scope.** Ingest a directory tree of Markdown plus attachments. Parse YAML frontmatter and wikilinks. Infer edges between documents. Store attachments content-addressed. Push assets and moments to Mnemosyne over the §3.1 envelope. Serve `GET /head` and `POST /presign` per §2.2.

**Reuse rather than reinvent, in both directions:**

- `integrations/obsidian-mnemosyne/main.ts` (318 lines of TypeScript) already parses and writes YAML frontmatter (`parseYaml`, `stringifyYaml`) and manages vault folders. It is **export-only**: `syncMemories()` at `:92` writes Mnemosyne memories out as vault notes, and there is no path that reads notes back in. It is the closest existing surface to this product and the natural place to prototype the reverse direction before committing to a separate codebase.
- `mnemosyne/core/importers/` is the best-factored subsystem in the repo and the pattern the archive should imitate for its own source plugins: a `BaseImporter` ABC (`base.py:48`) with a fixed four-phase `run()` (`:64`) over abstract `extract()` (`:140`) and `transform()` (`:149`), plus a literal `PROVIDERS` registry (`__init__.py:44`, eight providers today).

**Explicit non-goals**, recorded to prevent the outcome this whole split exists to avoid:

1. The archive does **not** rank or score recall.
2. The archive does **not** generate embeddings.
3. The archive does **not** replace or duplicate BEAM.
4. Mnemosyne does **not** gain a dependency on the archive.
5. Neither product's data model is the other's source of truth. The archive owns bytes and document structure; Mnemosyne owns the semantic index.

## 7. Open decisions

These are the owner's calls and are deliberately left unresolved here.

**Name.** Candidates that fit the librarian framing without colliding with Mnemosyne's mythological register: *Stacks*, *Scriptorium*, *Codex*, *Vellum*, *Atheneum*. A name that reads as a *place where things are kept* is preferable to one that reads as a brain, since the whole point of the split is that this is not the brain.

**Repository.** Recommendation: a separate repository in the same organization, so the `mnemosyne.media/1` contract can version independently of both products, and so Mnemosyne's dependency footprint stays trivially auditable.

**License.** Recommendation: match Mnemosyne, so code can move between them without a relicensing question.

**Language.** Open. The `head` and `presign` HTTP contract is language-neutral by design. If the Obsidian plugin becomes the prototype per §6, TypeScript is the path of least resistance for a first version.

---

## 8. Gap analysis

### 8.1 What exists

| Capability | Status | Location |
|---|---|---|
| Content-addressed byte store | ⚠️ write-only | `core/content_sanitizer.py:91` |
| `blob://sha256/<hash>` URI scheme | ⚠️ no reader | `content_sanitizer.py:126` |
| Registry pattern to copy | ✅ | `core/llm_backends.py:24-123` |
| Frontmatter parse and vault I/O | ⚠️ export-only | `integrations/obsidian-mnemosyne/main.ts:92` |
| Source-plugin pattern for the archive to copy | ✅ | `core/importers/base.py:48`, `__init__.py:44` |
| Deletion propagation through sync | ✅ | `beam.py:745`, `beam.py:2206` |
| A resolver abstraction | ❌ | nowhere |
| Any way to read a stored blob | ❌ | nowhere |
| An index-handoff contract | ❌ | nowhere |

### 8.2 What is missing

**Gap 1: blobs are unreachable.** Bytes quarantined by `sanitize_content` can never be retrieved.
**Impact:** a silent data-availability hole that exists **today**, independent of multimodal work. Any user whose memory tripped the 1 MB cap has bytes on disk that no code can return. Closing this is ~40 lines and is the highest value-per-line change in all three RFCs.

**Gap 2: no seam for byte access.** Media understanding would otherwise hardcode local filesystem reads.
**Impact:** media features would work only for local files, and Mnemosyne would have to proxy every byte to a provider.

**Gap 3: no index-handoff contract.** An external tool has no defined way to contribute what it knows.
**Impact:** without it, the companion product must either write directly to Mnemosyne's SQLite (coupling to internal schema) or not integrate at all. This gap is the one that decides whether the two-product split works.

---

## 9. Architecture

```
        ┌──────────────────────────────┐        ┌───────────────────────────────┐
        │   THE LIBRARIAN              │        │   THE ARCHIVE                 │
        │   Mnemosyne                  │        │   companion product, optional │
        │                              │        │                               │
        │  media_assets   (references) │        │  the actual bytes             │
        │  media_moments  (text+spans) │        │  thumbnails, transcodes       │
        │  working_memory (the index)  │        │  human-browsable structure    │
        │  recall, ranking, reasoning  │        │  markdown graph, UI           │
        │                              │        │                               │
        │  *** NEVER stores bytes ***  │        │  *** NEVER ranks recall ***   │
        └──────────────┬───────────────┘        └───────────────┬───────────────┘
                       │                                        │
                       │   index handoff  (archive -> brain)    │
                       │ <──────────────────────────────────────┤
                       │   mnemosyne_media_register             │
                       │   mnemosyne_media_add_moments          │
                       │   {"schema": "mnemosyne.media/1", ...} │
                       │   idempotent on (ref_kind, ref_value)  │
                       │                                        │
                       │   hydration  (brain -> archive)        │
                       ├──────────────────────────────────────> │
                       │   head()      validate a reference     │
                       │   presign()   let a PROVIDER fetch     │
                       │   open()      only on explicit view    │
                       │                                        │
                       │   *** recall() NEVER crosses here ***  │
                       └────────────────────────────────────────┘

   NO ARCHIVE PRESENT:
     get_resolver(scheme) -> None
       -> understanding_status stays 'unavailable'
       -> existing moments still recall perfectly (their TEXT is local)
       -> client shows a reference with no preview
       -> nothing raises
     You lose previews. You do not lose memories.

   GAP CLOSED TODAY, independent of everything else:
     BlobResolver gives content_sanitizer's write-only blob store its
     first reader (~40 lines). Bytes extracted since that feature shipped
     are currently unreachable by any code path.
```

---

## 10. Key file reference

| File | Purpose | Lines |
|---|---|---|
| `mnemosyne/core/resolvers.py` | **new.** `ContentResolver` protocol, registry, `BlobResolver` | ~180 |
| `mnemosyne/core/content_sanitizer.py` | unchanged. Read by `BlobResolver` | 169 |
| `mnemosyne/core/media.py` | RFC 0003. Stores `archive_locator`, never parses it | ~350 |
| `mnemosyne/tool_schemas.py`, `mcp_tools.py` | 2 media tools, plus **both** Hermes copies | +~120 |
| `tests/conftest.py` | reset the resolver registry beside `:64-69` | +~6 |
| `integrations/obsidian-mnemosyne/main.ts` | export-only today. Candidate prototype for Part 2 | 318 |
| `mnemosyne/core/importers/base.py` | the plugin pattern the archive should copy | 250 |

---

## 11. Conclusion

The community was right that Mnemosyne is missing something, and wrong about where to put it. A wiki layer inside the engine would trade away the four things that make Mnemosyne worth choosing: single file, zero dependencies, sub-millisecond, no external services. A boundary keeps all four and still gets the capability.

Three gaps, in order of value per line of code:

1. **Blobs are unreachable** and always have been. Roughly forty lines of `BlobResolver` fixes a live data-availability hole, and it is worth shipping on its own merits regardless of the rest of this document.
2. **No seam for byte access**, which would otherwise force local-filesystem-only media support or byte proxying through the brain.

> **Correction (2026-07-30): gap 2 is a blocker, not a nicety, and the sequencing below understates it.**
>
> This section recommends that `ContentResolver` plus `BlobResolver` "land with RFC 0003 phase 0, because they close an existing bug" — framing byte access as independently valuable but optional to the provider work. It is not optional. RFC 0002's `DescribeRequest` carries a URI and no bytes, so without a resolver to supply them the OpenAI-compatible adapter can describe **publicly fetchable URLs and nothing else** — no local file, no `blob://` reference. For a local-first memory system, that is close to describing nothing.
>
> **`ContentResolver` is therefore a hard prerequisite of the vision adapter**, and the adapter must not be scheduled before it. RFC 0002 §3.1's `DescribeRequest.fetch` field is the seam through which the two connect.
3. **No index-handoff contract**, which is the gap that determines whether a two-product architecture is real or just a diagram.

Recommended sequencing: `ContentResolver` plus `BlobResolver` land with RFC 0003 phase 0, because they close an existing bug. The MCP handoff tools land with phase 1. Part 2 begins only after the contract has one working implementation, which per §6 should probably be the Obsidian plugin learning to read.

**The key insight:** the interface between the two products is not a database, an SDK, or a shared library. It is a URI scheme plus a versioned JSON envelope plus two HTTP verbs. That is small enough that a third party can implement an archive in an afternoon, and small enough that Mnemosyne never needs to know whether one exists.
