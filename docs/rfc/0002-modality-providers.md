# RFC 0002: Modality Providers and the Vision Endpoint Seam

> **Retitled 2026-07-30.** This document was "Modality Providers and the Atlas Cloud Seam". The seam is generic — any OpenAI-compatible vision endpoint — and naming it after one provider misrepresented the design as vendor-coupled when it never was. See the correction in §3.2. Atlas Cloud remains the reference deployment and keeps its configuration recipe in §3.5.

**Status:** Draft
**Author:** abdiisan
**Related issue:** TBD
**Target version:** next MAJOR
**Supersedes:** nothing
**Related:** RFC 0003 (media moments), RFC 0004 (archive boundary)

---

## 0. Scope and non-scope

This RFC defines **one thing**: how Mnemosyne asks an external provider to describe a piece of non-text content, and gets text back.

It does **not** define where that text is stored (RFC 0003), how it is recalled (RFC 0003), or where the bytes live (RFC 0004). Those are separate documents on purpose, so this seam can land and be tested with no schema change and no recall change.

It also establishes the `docs/rfc/` document convention, since none was written down before. See §8.

---

## 1. R&D Summary: what the codebase can and cannot do today

### 1.1 Mnemosyne cannot consume anything except text

`mnemosyne/core/embeddings.py` is the only path from content to vector, and its entire public surface is text-shaped:

```python
def embed(texts: List[str]) -> np.ndarray
def embed_query(text: str) -> np.ndarray
def serialize(vec) -> bytes
```

There is no modality parameter, no dispatch, no second code path. Two backends exist behind those functions (local `fastembed` ONNX, and an OpenAI-compatible HTTP endpoint via `_embed_api`), but the choice between them is a *transport* decision made by the `_is_api_model` heuristic, not a *modality* decision. Feeding an image to Mnemosyne today is not "unsupported", it is unrepresentable.

### 1.2 The one clean provider abstraction in the repo is `LLMBackend`

`mnemosyne/core/llm_backends.py` is 124 lines and is the shape this RFC copies. Its virtues, in the order they matter:

- **One method.** `LLMBackend.complete(...)` at `llm_backends.py:37-47`. The module docstring calls the interface "intentionally tiny: one method, prompt-shaped, returning text-or-None."
- **The caller owns the prompt.** Stated explicitly at `llm_backends.py:30-33`: the method is named `complete` rather than `summarize` "because the same backend serves both memory consolidation and structured fact extraction; the caller, not the backend, owns the system prompt."
- **Failure is a return value, not an exception.** `call_host_llm` at `llm_backends.py:95-123` returns `None` when no backend is registered *or* when the backend raises.
- **A test seam ships with it.** `CallableLLMBackend` at `llm_backends.py:50-74` wraps any callable.
- **Opt-in by default.** The registry is inert unless a host calls `set_host_llm_backend` (`llm_backends.py:80`) and the user sets `MNEMOSYNE_HOST_LLM_ENABLED`.

### 1.3 Atlas Cloud already works for text, and needs no code

Atlas exposes OpenAI-compatible endpoints. Mnemosyne already speaks that protocol in two places, so the text half of the partnership is a configuration recipe:

| Capability | Status | Mechanism |
|---|---|---|
| Chat / summarization / extraction | ✅ works today | `MNEMOSYNE_LLM_BASE_URL` + `_API_KEY` + `_MODEL`, consumed by `_call_remote_llm` in `core/local_llm.py` |
| Text embeddings | ✅ works today | `MNEMOSYNE_EMBEDDING_API_URL` + `MNEMOSYNE_EMBEDDING_API_KEY`, consumed by `_embed_api` in `core/embeddings.py:263` |
| Vision (image understanding) | ❌ no seam | this RFC |
| Video understanding | ❌ no seam | this RFC |
| Audio / speech | ❌ no seam | this RFC, phase 2 |

One trap worth documenting in the recipe: `_is_api_model` (`core/embeddings.py:113`) routes to the API when the model name starts with `openai/` or contains `text-embedding`, **or** when `MNEMOSYNE_EMBEDDING_API_URL` is set to a non-OpenRouter URL. An Atlas-hosted BGE model shares a vendor-prefix shape with the local default (`BAAI/bge-small-en-v1.5`), so operators who want Atlas to serve embeddings should set `MNEMOSYNE_EMBEDDINGS_VIA_API=1` explicitly rather than relying on name inference.

### 1.4 The `vec_facts` cautionary tale

`vec_facts` was declared as a vec0 virtual table at `beam.py:1220` and **never given a writer**. The consequence is permanent: `reindex_vectors` (`beam.py:2394`) must recreate it empty on every reindex, and the comment at `beam.py:2448` explains why, so its declared dimension "can't mismatch a query." It also occupies a slot in three hardcoded table whitelists (`beam.py:552`, `:1544`, `:2150`).

This yields a rule that governs §2 of this RFC: **do not declare an interface with no writer.** An unused abstraction is not free optionality, it is a maintenance obligation that every future refactor must carry.

### 1.5 Embedding dimension is a database-level invariant, not a config value

This is the single most important constraint on any multimodal design, and it is enforced defensively:

- `_existing_vec_dim` (`beam.py:538`) parses the *stored DDL* of the existing vec tables to recover their dimension. The stored schema, not process config, is the source of truth.
- On mismatch, `init_beam` **refuses to create the vec tables** and logs `_dim_mismatch_message` (`beam.py:566`). See the guard at `beam.py:799-802`.
- Recovery is an explicit operator action, `mnemosyne reindex`, which re-embeds everything and recreates the tables at the active dimension (`reindex_vectors`, `beam.py:2394`).

Any design in which swapping a provider can change `EMBEDDING_DIM` is a design that can silently degrade every user's recall to lexical-only. §2 exists to make that impossible.

---

## 2. Decision: two concerns, one protocol, captioning only

**Content understanding and embedding generation are different concerns, and this RFC implements only the first.**

Understanding returns **text**. Embedding returns a **vector of a fixed dimension**. Per §1.5 that dimension is a database invariant. If a single `ModalityProvider` protocol covered both, then an operator changing `MNEMOSYNE_MODALITY_VISION_MODEL` could swing the vector space out from under an existing database. Fusing them puts a footgun behind a config key.

So the pipeline for all non-text content is:

```
media reference  ->  provider.describe()  ->  text moments  ->  existing embed()  ->  existing vec_working
```

One vector space. 384 dimensions by default. Unchanged.

### Why this is the right call and not just the cautious one

1. **Moments inherit the whole engine for free.** Because a moment is text, it gets the dense voice, FTS5, the binary voice, decay, tier degradation, sleep and consolidation, and `mnemosyne reindex` coverage with no new code. RFC 0003 §3 depends on this.
2. **It is the privacy story, not merely compatible with it.** Everything Mnemosyne indexes about an image is human-readable text the user can read, search, correct, and delete. A 512-float CLIP vector is none of those things. This is the concrete difference from OS-level ambient capture tools: the index is legible to the person it describes.
3. **It passes CI.** The test job runs with `MNEMOSYNE_NO_EMBEDDINGS=1`. Text moments still land in FTS with embeddings disabled, so media tests are runnable in CI. A dedicated media vector space would not be.
4. **It degrades to something useful.** With no provider configured, an asset is still registered by reference. With a provider but no embeddings, moments are still lexically searchable.

### What is explicitly deferred

Non-text media embeddings (CLIP-style joint image/text vector spaces) are **out of scope for this RFC and for RFC 0003 phases 0 through 4.** They require their own vec table, their own reindex path, and a redesign of `_existing_vec_dim`, which today returns the first matching table's dimension from a hardcoded name list (`beam.py:538-552`) and whose premise, that there is one vector space, a second modality space would falsify. That is a separate RFC.

Per §1.4, we do not stub the embedding protocol now. It gets declared when it gets a writer.

---

## 3. Proposed design

### 3.1 `mnemosyne/core/modality_backends.py` (new, ~200 lines)

A structural mirror of `core/llm_backends.py`.

**Request and result shapes.** Three dataclasses:

- `DescribeRequest`: `modality` (`"image" | "video" | "audio" | "document"`), `uri`, `content_hash`, `mime`, `hint`, `max_moments`, `span_hint`, `timeout`, `detail`, **`fetch`**.
- `MomentDraft`: `kind` (`caption | shot | transcript | style | ocr | page | summary`), `text`, `t_start_ms`, `t_end_ms`, `page_start`, `page_end`, `char_start`, `char_end`, `bbox`, `speaker`, `confidence`, `extra`.
- `DescribeResult`: `summary`, `moments: List[MomentDraft]`, `provider`, `model`, `warnings`, **`refused`**.

> **Correction (2026-07-30): two fields the original draft omitted, one of which is load-bearing.**
>
> **`DescribeRequest.fetch: Optional[Callable[[], Optional[bytes]]]`.** As drafted, the request carried `uri`, `mime`, and `content_hash` — and no way to carry bytes or obtain them. An OpenAI-compatible vision call needs a base64 `image_url` data part for anything that is not a publicly fetchable `https://` URL, so the adapter could describe public URLs and **nothing else**: no local file, no `blob://` reference, which is to say none of the media a privacy-first local memory system actually holds. `fetch` closes this. It is lazy and optional — a backend describing a public URL never calls it, and no bytes are read.
>
> This makes RFC 0004's `ContentResolver` a **hard prerequisite of this adapter**, not the independent nicety RFC 0004 §11 implies: `fetch` is supplied from the resolver registry. Sequence accordingly.
>
> **`DescribeResult.refused: bool`.** RFC 0003 §2.1 declares `understanding_status='refused'` as a valid state and nothing in this RFC produces it. A provider that declines on safety grounds is meaningfully different from one that failed, and the distinction should survive into the asset row.

`MomentDraft` is deliberately the wire shape of a `media_moments` row in RFC 0003 §2.2, minus identity and binding. A provider proposes drafts; the store assigns ids and decides what to keep.

**The protocol.**

```python
class ModalityBackend(Protocol):
    """A provider that turns non-text content into text moments."""

    name: str
    modalities: frozenset[str]

    def describe(self, request: DescribeRequest) -> Optional[DescribeResult]:
        ...
```

One method. Text-or-None on failure. `hint` carries the caller's prompt, preserving the division of labour documented at `llm_backends.py:30-33`: the caller owns the prompt, the backend owns the routing.

**Registration.** Process-globals `_backends: Dict[str, ModalityBackend]` keyed by modality and `_default: Optional[ModalityBackend]`, with `set_modality_backend(backend, modalities=None)`, `get_modality_backend(modality)`, `clear_modality_backends()`, and `call_modality_describe(request) -> Optional[DescribeResult]` which swallows exceptions exactly as `call_host_llm` does (`llm_backends.py:113-123`).

**Test seam.** `CallableModalityBackend`, mirroring `CallableLLMBackend` (`llm_backends.py:50-74`).

*Estimated: ~200 lines in a new file, ~150 lines of tests.*

### 3.2 `mnemosyne/core/modality_openai_compat.py` (new, ~300 lines)

The OpenAI-compatible vision adapter, and the reference implementation of the protocol.

> **Correction (2026-07-30): this module was named `modality_atlas.py` in the original draft. Renamed, for a reason worth recording.**
>
> Naming a core module after one vendor makes that vendor structural. It is not: the module speaks the OpenAI-compatible `chat/completions` shape, which is served by Atlas Cloud, OpenRouter, Together, Groq, vLLM, LM Studio, Ollama, and a local `llama.cpp` server alike. §3.3's ladder already said "Atlas Cloud / **any OpenAI-compatible vision endpoint**" and §3.4's config keys were already vendor-neutral (`MNEMOSYNE_MODALITY_BASE_URL`, not `MNEMOSYNE_ATLAS_URL`), so the design was never actually coupled — only the module name and the framing were.
>
> Probing the partner endpoint made the point concrete. `GET https://api.atlascloud.ai/v1/models` returns **123 models, 30 of them accepting image input — and every one is another vendor's** (OpenAI, Anthropic, Google, Qwen, Moonshot, xAI, ByteDance, Z.ai). Atlas is an OpenAI-compatible **aggregator**. There is no "Atlas vision API" distinct from the generic one to write an adapter against.
>
> This also matches the project's partnership policy: sponsorship should produce work that benefits the whole ecosystem, not architecture that privileges one provider. **Atlas Cloud is the reference deployment and a documented recipe (§3.5), not a dependency.** The correct test of this correction is that switching endpoints requires changing environment variables only, and that test belongs in the implementing PR.

- OpenAI-compatible `POST {base_url}/chat/completions` with `image_url` content parts.
- Reuses the retry, backoff, and timeout shape of `_call_remote_llm_with_model` (`core/local_llm.py:458`) and the rate-limit classifier `_is_rate_limit_error` (`core/embeddings.py:247`). Do not invent a third retry policy.
- Requests strict JSON: a `{"summary": ..., "moments": [...]}` envelope. Parsing follows the tolerance ladder already proven in `core/extraction.py`: JSON first, then partial-JSON salvage, then give up and return `None`. Never raise into `remember()`.
- Prompt overridable by env, the affordance `core/extraction.py` already provides via `MNEMOSYNE_EXTRACTION_PROMPT`.
- Selected when `MNEMOSYNE_MODALITY_BASE_URL` and `MNEMOSYNE_MODALITY_API_KEY` are both set.
- **Optional capability preflight.** `GET {base_url}/models` exposes `input_modalities` per model on Atlas and OpenRouter alike — a convention, not a vendor extension. Use it to warn when the configured vision model is text-only. It must stay strictly optional: an endpoint that does not expose it, or exposes a different shape, must still work. Degrade to silence, never to failure.

*Estimated: ~300 lines in a new file, ~200 lines of tests against a stub HTTP server.*

### 3.3 The degradation ladder

Mirroring the fallback chain in `core/local_llm.py` and the disabled-path posture of `embeddings.available()`:

```
1. host-registered backend for this modality   (set_modality_backend)
2. Atlas Cloud / any OpenAI-compatible vision endpoint
3. (reserved: local captioner, not implemented)
4. metadata-only registration
```

Rung 4 is the important one, and it is a success case, not an error case. The asset is registered by reference, zero moments are written, and `understanding_status` is set to `unavailable` (RFC 0003 §2.1). Nothing raises. Nothing blocks. A user with no provider configured still gets a searchable record that they referenced a given file, which is strictly more than they have today.

### 3.4 Configuration

New keys. **Every one must be added to both `ENV_VAR_MAP` (`core/config.py:62`) and `DEFAULTS` (`core/config.py:192`)**, or it will not be readable from the environment at all, since a key absent from `ENV_VAR_MAP` is YAML-only.

| Config key | Env var | Default | Restart? |
|---|---|---|---|
| `modality_enabled` | `MNEMOSYNE_MODALITY_ENABLED` | `false` | no |
| `modality_base_url` | `MNEMOSYNE_MODALITY_BASE_URL` | `""` | **yes** |
| `modality_api_key` | `MNEMOSYNE_MODALITY_API_KEY` | `""` | no |
| `modality_vision_model` | `MNEMOSYNE_MODALITY_VISION_MODEL` | `""` | **yes** |
| `modality_video_model` | `MNEMOSYNE_MODALITY_VIDEO_MODEL` | `""` | **yes** |
| `modality_audio_model` | `MNEMOSYNE_MODALITY_AUDIO_MODEL` | `""` | **yes** |
| `modality_timeout` | `MNEMOSYNE_MODALITY_TIMEOUT` | `60` | no |
| `modality_max_moments` | `MNEMOSYNE_MODALITY_MAX_MOMENTS` | `12` | no |

The base-URL and model keys join `REQUIRES_RESTART` (`core/config.py:36`) for the same reason `embedding_model` is already there (`core/config.py:43`): they are read once at client construction.

`modality_enabled` defaults to **false**. Mnemosyne must not make an outbound call to describe media until the operator has said yes, once, explicitly. This is a privacy invariant, restated as a testable assertion in RFC 0004 §3.

### 3.5 Atlas Cloud configuration recipe (docs deliverable)

A `docs/integrations/atlas-cloud.md` page covering all three surfaces in one place: chat via `MNEMOSYNE_LLM_BASE_URL`, embeddings via `MNEMOSYNE_EMBEDDING_API_URL` plus the `MNEMOSYNE_EMBEDDINGS_VIA_API=1` caveat from §1.3, and vision via the new `MNEMOSYNE_MODALITY_*` keys. Follow `docs/integrations/integration-template.md`.

---

## 4. Gap analysis

### 4.1 What exists

| Capability | Status | Location |
|---|---|---|
| Provider protocol pattern to copy | ✅ | `core/llm_backends.py:24-123` |
| OpenAI-compatible chat transport | ✅ | `core/local_llm.py` (`_call_remote_llm`) |
| OpenAI-compatible embedding transport | ✅ | `core/embeddings.py:263` (`_embed_api`) |
| Rate-limit classifier | ✅ | `core/embeddings.py:247` |
| Tolerant JSON parsing from an LLM | ✅ | `core/extraction.py` (`_parse_facts`) |
| Config plumbing with restart semantics | ✅ | `core/config.py:36`, `:62`, `:192` |
| Content-addressed byte storage | ⚠️ write-only | `core/content_sanitizer.py:91-103`, see RFC 0004 |
| Any notion of modality | ❌ | nowhere |
| Any notion of a span within content | ❌ | nowhere, see RFC 0003 |

### 4.2 What is missing

**Gap 1: `embeddings.py` has no dispatch.** `embed()` takes `List[str]`. There is no place for an image to enter the system.
**Impact:** the Atlas vision, video, and audio endpoints are unreachable from Mnemosyne regardless of configuration.

**Gap 2: no provider abstraction outside of LLM completion.** `llm_backends.py` is specific to prompt-in / text-out completion, and `embeddings.py` selects transports by string heuristic rather than by a registry.
**Impact:** every new provider kind would otherwise be bolted in as another set of env vars and another bespoke HTTP function. There are already four such sets (LLM, LLM fallback, embeddings, conflict detection). A fifth without an abstraction is where the design debt becomes permanent.

**Gap 3: no test seam for provider behaviour.** There is no way to assert "with no provider configured, nothing is called."
**Impact:** the privacy invariants in RFC 0004 §3 would be unenforceable.

---

## 5. Test obligations

**Process-global reset is mandatory.** `tests/conftest.py:64-69` resets `llm_backends._backend` with the comment "The registry is a process-global; a test that forgets to unregister would otherwise bleed into the next." That block exists because this exact bug happened once. `modality_backends._backends` and `._default` must be reset in the same fixture, in the same style.

**Assert silence when disabled.** With `MNEMOSYNE_MODALITY_ENABLED` unset, a test must assert that no socket is opened and no backend method is called. CI runs with `MNEMOSYNE_NO_EMBEDDINGS=1` and no network expectation; a media path that dials out on import or on plain `remember()` would hang the matrix.

**Assert graceful degradation.** With `modality_enabled=true` and no reachable provider, `describe` returns `None` and the caller proceeds. With a backend that raises, `call_modality_describe` returns `None` and the caller proceeds.

Test naming follows the repo's feature convention: `tests/test_modality_backends.py`, `tests/test_modality_openai_compat.py`. Ruff runs `--select E9,F63,F7,F82` only, so style will not gate; correctness will.

---

## 6. Architecture

```
                        ┌──────────────────────────────────────┐
                        │  caller: remember_media()  (RFC 0003)│
                        └───────────────┬──────────────────────┘
                                        │ DescribeRequest
                                        v
   ┌────────────────────────────────────────────────────────────────────┐
   │  modality_backends.py   call_modality_describe()   [THIS RFC]      │
   │  ─────────────────────────────────────────────────────────────     │
   │  1. host backend for modality      (set_modality_backend)          │
   │  2. Any OpenAI-compatible endpoint  (modality_openai_compat.py)             │
   │  3. (reserved: local captioner)                                    │
   │  4. metadata-only  ->  understanding_status='unavailable'          │
   │     never raises, never blocks remember()                          │
   └────────────────────────────────┬───────────────────────────────────┘
                                    │ DescribeResult{ summary, moments[] }
                                    │ ... which is TEXT
                                    v
   ┌────────────────────────────────────────────────────────────────────┐
   │  embeddings.py  embed(List[str])          [UNCHANGED, text-only]   │
   │  one vector space, EMBEDDING_DIM=384                               │
   │  guarded by _existing_vec_dim (beam.py:538) which refuses on        │
   │  mismatch rather than corrupt (beam.py:799-802)                     │
   └────────────────────────────────┬───────────────────────────────────┘
                                    v
                        vec_working / memory_embeddings / fts_working
                                 (existing, untouched)

   GAP CLOSED BY THIS RFC:  a modality has somewhere to plug in.
   GAP LEFT OPEN ON PURPOSE: media-native (CLIP-style) vectors. Needs a
        second vector space, which falsifies the premise of
        _existing_vec_dim. Separate RFC.
```

---

## 7. Key file reference

| File | Purpose | Lines |
|---|---|---|
| `mnemosyne/core/modality_backends.py` | **new.** Protocol, dataclasses, registry | ~200 |
| `mnemosyne/core/modality_openai_compat.py` | **new.** OpenAI-compatible vision adapter (Atlas, OpenRouter, vLLM, LM Studio, …) | ~300 |
| `mnemosyne/core/llm_backends.py` | the pattern being copied. No change | 124 |
| `mnemosyne/core/embeddings.py` | no change. Consumed as-is for moment text | 402 |
| `mnemosyne/core/config.py` | 8 keys into `ENV_VAR_MAP` / `DEFAULTS` / `REQUIRES_RESTART` | +~15 |
| `tests/conftest.py` | reset the new process-globals beside `:64-69` | +~8 |
| `docs/integrations/atlas-cloud.md` | **new.** Configuration recipe | ~120 |

---

## 8. Document convention established by this RFC

`docs/rfc/` previously held two files in two different formats, with no convention recorded in `CONTRIBUTING.md` or `docs/README.md`, and was not linked from the docs index. This RFC and its siblings settle it:

1. **Numbered filenames**, `NNNN-kebab-title.md`, allocated in order.
2. **Header block** copied from `0001-tags-and-scope-unification.md`: an `# RFC NNNN: Title` heading followed by bold `**Status:**`, `**Author:**`, `**Related issue:**`, `**Target version:**` lines with no blank lines between them, then a `---` rule.
3. **Numbered `## N.` sections** with `### N.M` subsections, separated by `---` rules, per `noise-remediation-rnd.md`.
4. **A gap-analysis table** using ✅ / ⚠️ / ❌ with a `Location` column, and `#### Gap N:` blocks each closing on a bold **Impact:** line.
5. **Effort estimates in lines of code** on every proposal, and honest Cons.
6. **A key-file table** and an ASCII architecture diagram with inline `GAP:` annotations.
7. **Every claim about existing code carries a `file.py:line` anchor.** Line numbers drift; a reader who finds a stale anchor should trust the file over the RFC and fix the anchor.
8. Unnumbered exploratory documents may keep the `R&D:` prefix style of `noise-remediation-rnd.md`. Numbered RFCs propose; R&D notes investigate.

---

## 9. Conclusion

The text half of the Atlas partnership needs **zero lines of code**: it is `MNEMOSYNE_LLM_BASE_URL` and `MNEMOSYNE_EMBEDDING_API_URL`, both already wired, plus a documentation page. That should ship first and independently.

The non-text half needs a seam that does not exist, and the shape of that seam is already in the repo as `LLMBackend`: one method, text-or-None, caller owns the prompt, opt-in by default, inert when unconfigured.

**The key insight:** captioning and embedding must not share an interface. Understanding produces text; embedding produces a vector whose dimension is a database invariant that `init_beam` will refuse to violate (`beam.py:799-802`). Keeping them separate means every modality reduces to text and reuses the entire existing engine, which is simultaneously the cheapest implementation, the most legible privacy story, and the only version that passes CI with `MNEMOSYNE_NO_EMBEDDINGS=1`.

The single highest-impact change in this RFC is the smallest one: `modality_enabled` defaults to false, and rung 4 of the degradation ladder is a success case. Together they mean this feature can ship without changing the behaviour of a single existing installation.
