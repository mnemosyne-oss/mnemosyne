# OpenAI-Compatible Vision

Mnemosyne can turn an image, audio clip, video or document into located text, and store the
description as ordinary, recallable memory. This page is the primary guide for
configuring that.

The seam speaks one protocol — `POST {base_url}/chat/completions` with
`image_url` content parts — so **any** endpoint that implements it works. There
is no provider-specific code path, and switching providers is an environment
change, never a code change.

> **Nothing is sent until you say so.** `modality_enabled` defaults to `false`.
> With it unset, Mnemosyne makes no outbound call to describe media, opens no
> socket, and never reads the bytes of a local file. Turning it on is a
> deliberate, one-time act.

---

## What it does

Point Mnemosyne at a piece of media and it records:

- an **asset row** — the reference (URL, file path, or content hash), plus what
  it knows about the media. The bytes are never copied into the database.
- zero or more **moments** — captions, shots, transcript segments, OCR regions.
  Each retained moment becomes an ordinary memory row, so it is recalled,
  decayed, consolidated, and synced by the machinery that already exists.

If no provider is configured, or the provider is unreachable, or it declines,
the asset is still registered by reference. That is a **success state**, not an
error: you get a searchable record that you referenced a given file, which is
strictly more than you had before.

---

## What each modality does

| Modality | Sent where | Moments | Needs |
|---|---|---|---|
| Image | `POST {base_url}/chat/completions`, one `image_url` part | `caption`, `ocr` | `MNEMOSYNE_MODALITY_VISION_MODEL` |
| Audio | `POST {base_url}/audio/transcriptions`, multipart, `response_format=verbose_json` | `transcript`, timed | `MNEMOSYNE_MODALITY_AUDIO_MODEL` (e.g. `whisper-1`, `whisper-large-v3`) |
| Video | Up to 8 sampled frames to `/chat/completions` in one call; soundtrack to `/audio/transcriptions` | `shot` and `transcript`, timed | `ffmpeg` and `ffprobe` on PATH; `MNEMOSYNE_MODALITY_VIDEO_MODEL`, or the vision model if unset |
| Document | Nowhere. Read locally | `page`, located by page, slide, chapter or character range | Nothing for txt, md, docx, pptx, epub; `pip install 'mnemosyne-memory[media]'` for PDF |

Documents never leave the machine. A PDF page with no text layer (a scan) is
the one exception: it is rendered and sent to the image model like any other
image, so a scanned page is described instead of skipped.

Audio uploads over 25 MB are refused locally rather than sent and rejected.
Long documents are packed up to `MNEMOSYNE_MODALITY_MAX_MOMENTS` passages; when
that cap stops extraction early, the result warns with how far it got.

Setting the variables below is the whole setup. The adapter registers itself
the first time something is described after `MNEMOSYNE_MODALITY_ENABLED` is on;
there is no code to call.

---

## Using it: tool, CLI or SDK

| Surface | Call |
|---|---|
| MCP | `mnemosyne_remember_media` with `ref` (plus optional `modality`, `title`, `hint`, `max_moments`, `bank`) |
| Hermes | the same tool, exposed by the Mnemosyne provider |
| CLI | `mnemosyne media ./diagram.png` (add `--json` for the full result) |
| Python | `BeamMemory.remember_media(ref)` |

A tool call may come from a remote MCP client or from a model steered by text it
just read, so the tool is stricter than the CLI and SDK:

- **Local files need an allow-list.** Set `MNEMOSYNE_MEDIA_ALLOWED_PATHS` to the
  directories media may be read from (`:`-separated on Linux and macOS, `;` on
  Windows, or commas). The path is resolved through symlinks before the check,
  so a link inside an allowed folder cannot reach outside it. With nothing set,
  the tool accepts no local paths; pass an `https://` URL or a `data:` URI
  instead.
- **No internal URLs.** URLs that resolve to loopback, private or link-local
  addresses are refused, because audio and video are fetched from this machine.
  `MNEMOSYNE_MEDIA_ALLOW_PRIVATE_URLS=1` allows a trusted media server on your
  LAN.
- **Inline payloads are capped** at 25 MB decoded.

The CLI and the SDK are not restricted: there the caller is the person whose
files they are.

---

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `MNEMOSYNE_MODALITY_ENABLED` | `false` | Master switch. Nothing happens until this is on. |
| `MNEMOSYNE_MODALITY_BASE_URL` | *(unset)* | Base URL of the endpoint, including `/v1` if the provider uses one. |
| `MNEMOSYNE_MODALITY_API_KEY` | *(unset)* | Bearer token. Required — a base URL alone is treated as an unfinished config. |
| `MNEMOSYNE_MODALITY_VISION_MODEL` | *(unset)* | Model for images, scanned PDF pages, and video frames when no video model is set. |
| `MNEMOSYNE_MODALITY_VIDEO_MODEL` | *(unset)* | Vision model for sampled video frames. Falls back to the vision model. |
| `MNEMOSYNE_MODALITY_AUDIO_MODEL` | *(unset)* | Transcription model for audio and video soundtracks. |
| `MNEMOSYNE_MODALITY_TIMEOUT` | `60` | Per-call timeout, seconds. |
| `MNEMOSYNE_MODALITY_MAX_MOMENTS` | `12` | Cap on moments retained per asset. |
| `MNEMOSYNE_MEDIA_ALLOWED_PATHS` | *(unset)* | Directories the `mnemosyne_remember_media` tool may read local files from. Unset: tool calls cannot name local files. |
| `MNEMOSYNE_MEDIA_ALLOW_PRIVATE_URLS` | `false` | Let the tool fetch URLs that resolve to private or loopback addresses. |
| `MNEMOSYNE_MODALITY_PROMPT` | *(built-in)* | Override the description prompt. `{modality}` and `{max_moments}` are substituted. |

`MNEMOSYNE_MODALITY_BASE_URL` and the three model keys are read once when the
client is constructed, so changing them needs a restart. The rest apply
immediately.

### config.yaml wins over these variables

Read this before using the examples below. Mnemosyne resolves settings in the
order **`config.yaml` > environment variable > built-in default**, and
*presence* in the file decides it, not the value. Once a key exists in
`config.yaml`, the matching variable is never consulted again.

That matters here because `config.yaml` is seeded with every known key the
first time Mnemosyne runs. The seed copies whatever variables are already
exported, so:

* Export these variables **before Mnemosyne has ever run**, and the seeded file
  captures them. The examples below work as written.
* Export them **after** a config already exists, and they are silently ignored,
  because the file already contains `modality_enabled: false` and an empty
  `modality_base_url`.

If Mnemosyne has run before, set the values instead of exporting them:

```bash
mnemosyne config set modality_enabled true
mnemosyne config set modality_base_url https://your-endpoint/v1
mnemosyne config set modality_api_key ...
mnemosyne config set modality_vision_model your/model
```

`mnemosyne config migrate` re-exports your current variables into the file in
one step, which is the quickest way to adopt an existing shell setup.

The env-var form in the examples below stays correct for containers and
ephemeral filesystems, where the variables are set before the first run every
time.

---

## Worked examples

Each block below is complete. Nothing but these variables changes between them.

### Atlas Cloud

```bash
export MNEMOSYNE_MODALITY_ENABLED=1
export MNEMOSYNE_MODALITY_BASE_URL=https://api.atlascloud.ai/v1
export MNEMOSYNE_MODALITY_API_KEY=...
export MNEMOSYNE_MODALITY_VISION_MODEL=qwen/qwen3-vl-235b-a22b-thinking
```

See [Atlas Cloud](atlas-cloud.md) for the full recipe, which also covers chat
and embeddings.

### OpenRouter

```bash
export MNEMOSYNE_MODALITY_ENABLED=1
export MNEMOSYNE_MODALITY_BASE_URL=https://openrouter.ai/api/v1
export MNEMOSYNE_MODALITY_API_KEY=sk-or-...
export MNEMOSYNE_MODALITY_VISION_MODEL=google/gemini-2.0-flash-001
```

### vLLM (self-hosted)

```bash
export MNEMOSYNE_MODALITY_ENABLED=1
export MNEMOSYNE_MODALITY_BASE_URL=http://localhost:8000/v1
export MNEMOSYNE_MODALITY_API_KEY=whatever-you-configured
export MNEMOSYNE_MODALITY_VISION_MODEL=Qwen/Qwen2-VL-7B-Instruct
```

vLLM does not require a key by default, but Mnemosyne does — set
`--api-key` on the server and match it here, rather than leaving both blank.

### LM Studio (local, no cloud)

```bash
export MNEMOSYNE_MODALITY_ENABLED=1
export MNEMOSYNE_MODALITY_BASE_URL=http://localhost:1234/v1
export MNEMOSYNE_MODALITY_API_KEY=lm-studio
export MNEMOSYNE_MODALITY_VISION_MODEL=qwen2-vl-7b-instruct
```

This is the configuration to use if you want media understanding with no
network egress at all.

---

## How local files are sent

For a public `https://` URL, Mnemosyne passes the URL through and the provider
fetches it. **No bytes are read locally and nothing new leaves your machine.**

For anything else — a local path, a `blob://` reference — Mnemosyne reads the
bytes and sends them inline as a base64 data part. This is the only way an
OpenAI-compatible vision endpoint can see content it cannot fetch itself.

If you are not comfortable with that for a given file, do not ingest it: there
is no ambient capture, so nothing is described that you did not name.

---

## Checking your model can actually see

Some endpoints publish per-model input modalities on `GET {base_url}/models`.
When yours does, Mnemosyne uses it to warn if the configured model is text-only.
When it does not, nothing is reported and everything still works — the check is
strictly optional and degrades to silence.

```python
from mnemosyne.core.modality_openai_compat import warn_if_model_cannot_see

warn_if_model_cannot_see("some/model", "https://your-endpoint/v1", "your-key")
```

---

## Degradation ladder

In order:

1. A host-registered backend for this modality.
2. This adapter, when a base URL and key are configured.
3. *(reserved: a local captioner — not implemented.)*
4. Metadata-only registration.

Rung 4 is a success. The asset is registered, `understanding_status` is set to
`unavailable`, zero moments are written, and nothing raises.

A provider that *declines* on safety grounds is recorded separately, as
`refused` — it will decline again, whereas a failure is worth retrying.

---

## Bringing your own backend

The adapter is one implementation of a small protocol. A host that has its own
authenticated vision client can register it directly and skip HTTP entirely:

```python
from mnemosyne.core.modality_backends import (
    CallableModalityBackend, DescribeResult, DescribedMoment, set_modality_backend,
)

def describe(request):
    text = my_client.caption(request.uri)
    return DescribeResult(
        summary=text,
        moments=[DescribedMoment(kind="caption", text=text)],
        provider="my-host",
    )

set_modality_backend(
    CallableModalityBackend(name="my-host", func=describe,
                            modalities=frozenset({"image"})),
)
```

The opt-in gate still applies: with `MNEMOSYNE_MODALITY_ENABLED` unset, a
registered backend is never called.

---

## See also

- [Atlas Cloud recipe](atlas-cloud.md) — all three surfaces in one place
- `docs/rfc/0002-modality-providers.md` — why the seam looks like this
- `docs/rfc/0003-media-moments.md` — where the text ends up
- `docs/rfc/0004-archive-boundary.md` — why the bytes do not
