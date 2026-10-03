# Atlas Cloud

[Atlas Cloud](https://atlascloud.ai) is an OpenAI-compatible model aggregator.
This page is a **configuration recipe** — there is no
Atlas-specific code in Mnemosyne, and there is not meant to be. Everything below
is environment variables against interfaces Mnemosyne already speaks, which is
why the same three blocks work against OpenRouter, vLLM, or a local server with
only the URLs changed.

Atlas covers all three of Mnemosyne's outbound surfaces.

| Surface | Config | Notes |
|---|---|---|
| Chat — consolidation, extraction, conflict detection | `MNEMOSYNE_LLM_*` | Works today, no new code |
| Embeddings | `MNEMOSYNE_EMBEDDING_*` | Works today, one gotcha below |
| Vision | `MNEMOSYNE_MODALITY_*` | See [OpenAI-Compatible Vision](openai-compatible-vision.md) |

Two base URLs, and they are not interchangeable:

- `https://api.atlascloud.ai/v1` — OpenAI-compatible chat, embeddings, vision.
  **This is the one Mnemosyne uses.**
- `https://api.atlascloud.ai/api/v1` — Atlas's own image/video generation and
  `uploadMedia` endpoints. Mnemosyne does not call these.

---

## Chat

Drives sleep-time consolidation, fact extraction, and conflict detection.

```bash
export MNEMOSYNE_LLM_BASE_URL=https://api.atlascloud.ai/v1
export MNEMOSYNE_LLM_API_KEY=...
export MNEMOSYNE_LLM_MODEL=...
```

---

## Embeddings

```bash
export MNEMOSYNE_EMBEDDING_API_URL=https://api.atlascloud.ai/v1
export MNEMOSYNE_EMBEDDING_API_KEY=...
export MNEMOSYNE_EMBEDDING_MODEL=...
export MNEMOSYNE_EMBEDDINGS_VIA_API=1   # <- do not omit this
```

**The gotcha.** Mnemosyne decides between the local ONNX embedder and a remote
endpoint using a name heuristic: a model routes to the API when its name starts
with `openai/` or contains `text-embedding`. A hosted BGE model has the same
vendor-prefix shape as the *local* default (`BAAI/bge-small-en-v1.5`), so
inference guesses "local" and quietly ignores your endpoint. Set
`MNEMOSYNE_EMBEDDINGS_VIA_API=1` explicitly rather than relying on the heuristic.

**Dimensions must match what is already stored.** Switching embedding providers
mid-database is a reindex, not a config change — check
`MNEMOSYNE_EMBEDDING_DIM` against your existing vec tables before you switch,
or run `mnemosyne reindex` after.

---

## Vision

```bash
export MNEMOSYNE_MODALITY_ENABLED=1
export MNEMOSYNE_MODALITY_BASE_URL=https://api.atlascloud.ai/v1
export MNEMOSYNE_MODALITY_API_KEY=...
export MNEMOSYNE_MODALITY_VISION_MODEL=qwen/qwen3-vl-235b-a22b-thinking
```

> **If Mnemosyne has already run on this machine**, exporting these does
> nothing: `config.yaml` was seeded on first run and takes precedence over the
> environment. Use `mnemosyne config set modality_enabled true` and the
> matching `modality_*` keys instead, or `mnemosyne config migrate` to import
> your current variables. See
> [OpenAI-compatible vision](openai-compatible-vision.md#configyaml-wins-over-these-variables).


Atlas aggregates vision models from several vendors, so the model name is the
only thing you pick here. `GET https://api.atlascloud.ai/v1/models` lists what is
available along with each model's `input_modalities`; Mnemosyne can use that to
warn you if you configure a text-only model. See
[OpenAI-Compatible Vision](openai-compatible-vision.md) for the full surface.

---

## Keys

Every value above is a secret. Keep them in your environment or a secret
manager — never in `config.yaml` committed to a repository, and never in a
shell history file you sync. If a key has been pasted anywhere it should not
be, rotate it; that is cheaper than auditing where it went.

---

## Why there is no `modality_atlas.py`

Mnemosyne's partnership policy is that sponsored work benefits the whole
ecosystem rather than privileging one provider, so core modules are named after
protocols, not vendors. The evidence agreed with the policy: Atlas serves 123
models, 30 of them vision-capable, and **every one belongs to another vendor**.
Atlas is an aggregator of the OpenAI-compatible protocol — there was never an
Atlas-specific API to adapt to.

So Atlas is the **reference deployment**: the configuration we verify against,
documented here as one worked example among several. The acceptance test is that
switching to any other endpoint requires changing environment variables only.
