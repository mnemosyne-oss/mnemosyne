# Mnemosyne Open Issues Summary (16 issues)
Generated: 2026-07-19

**Engagement breakdown:** 13 WARM | 1 LUKEWARM | 2 COLD

---
## #491: Trim-before-embedding can trigger a foreign-key failure
**Author:** pvspencer22 | **Created:** 2026-07-18T19:09:54Z | **Updated:** 2026-07-18T19:09:54Z
**Labels:** none
**Engagement:** COLD | AxDSan replies: 0 | Community replies: 0 | Total comments: 0
**Body (first 200 chars):** ## Summary  `BeamMemory.remember()` commits a new `working_memory` row, runs `_trim_working_memory()`, and only then generates/stores the embedding. If trimming or a concurrent delete removes that par

---
## #489: Suggested title: task_progress fails: "cannot start a transaction within a transaction" (BEGIN IMMEDIATE on a connection already in a transaction)
**Author:** Darius1978 | **Created:** 2026-07-18T12:32:26Z | **Updated:** 2026-07-18T12:32:26Z
**Labels:** bug
**Engagement:** COLD | AxDSan replies: 0 | Community replies: 0 | Total comments: 0
**Body (first 200 chars):** ## Summary  `mnemosyne_task_progress` (set/create action) reliably fails with:  ``` sqlite3.OperationalError: cannot start a transaction within a transaction ```  The failure is **deterministic and n

---
## #487: [BUG]
**Author:** codxt | **Created:** 2026-07-17T22:39:34Z | **Updated:** 2026-07-18T01:57:57Z
**Labels:** bug
**Engagement:** LUKEWARM | AxDSan replies: 0 | Community replies: 1 | Total comments: 1
**Body (first 200 chars):** ## Summary  Using `mnemosyne-hermes==0.4.0` as the Hermes memory provider causes a very large Hermes gateway RSS increase on the first ordinary turn.  In a controlled enabled/off comparison:  | Mode |
**Key quotes:**
  - [dplush] @codxt thanks for the controlled comparison — I ran a small isolated follow-up against the same gene...

---
## #482: [BUG] 50 of 106 config.yaml keys are silently ignored — core modules read env vars into module-level constants at import
**Author:** Pawls | **Created:** 2026-07-16T21:27:37Z | **Updated:** 2026-07-17T03:53:04Z
**Labels:** none
**Engagement:** WARM | AxDSan replies: 1 | Community replies: 1 | Total comments: 2
**Body (first 200 chars):** ## Summary  Roughly half the `config.yaml` keys have no effect on behaviour. `mnemosyne config set <key>` writes them, `mnemosyne config get <key>` reads them back, and `MnemosyneConfig` resolves them
**Key quotes:**
  - [AxDSan] @Pawls this is a serious finding, and thank you for the thorough analysis. The root cause is correct: module-level `os.environ.get()` calls at import...
  - [dplush] I ran a read-only consumer audit to help scope the refactor. Current `main`: 106 `DEFAULTS` / 106 ma...

---
## #474: embeddings.available() reports True while vec_episodes is never created — silent sqlite-vec fallback hides hybrid retrieval regression
**Author:** sky770825 | **Created:** 2026-07-16T00:31:35Z | **Updated:** 2026-07-17T02:48:04Z
**Labels:** none
**Engagement:** WARM | AxDSan replies: 1 | Community replies: 0 | Total comments: 1
**Body (first 200 chars):** ## Environment  - mnemosyne-memory: latest from PyPI - Python: 3.11 (macOS arm64, also reproducible on Linux) - Verified in `site-packages/mnemosyne/core/embeddings.py:42` and `site-packages/mnemosyne
**Key quotes:**
  - [AxDSan] @sky770825 good catch. The issue is that `embeddings.available()` only checks whether sqlite-vec is loadable, not whether `vec_episodes` was actually...

---
## #450: [FEATURE] Session-driven memory activation with meaningful-use reinforcement
**Author:** israellot | **Created:** 2026-07-12T14:57:10Z | **Updated:** 2026-07-15T14:31:50Z
**Labels:** none
**Engagement:** WARM | AxDSan replies: 1 | Community replies: 0 | Total comments: 1
**Body (first 200 chars):** ## Problem / Motivation  Memory relevance for an interactive agent is better measured by intervening experience than by elapsed wall-clock time.  A memory should not become cognitively colder simply b
**Key quotes:**
  - [AxDSan] @israellot thanks for the thoughtful proposal. The session-driven activation model with meaningful-use reinforcement is a good direction.  A few obser...

---
## #449: Proposal: semantic dirtying and validation-aware recall
**Author:** israellot | **Created:** 2026-07-12T14:35:25Z | **Updated:** 2026-07-15T14:31:52Z
**Labels:** none
**Engagement:** WARM | AxDSan replies: 1 | Community replies: 0 | Total comments: 1
**Body (first 200 chars):** ## Summary  Mnemosyne already has strong primitives for evolving memory:  - `valid_until` - `superseded_by` - temporal triples - canonical fact history - veracity and validation metadata - consolidati
**Key quotes:**
  - [AxDSan] @israellot semantic dirtying and validation-aware recall is a strong concept. The current pipeline doesn't have a "dirty flag" propagation mechanism,...

---
## #446: Proposal: versioned, provenance-aware migrations for auto-seeded config.yaml
**Author:** dplush | **Created:** 2026-07-11T23:33:46Z | **Updated:** 2026-07-17T03:40:52Z
**Labels:** none
**Engagement:** WARM | AxDSan replies: 1 | Community replies: 2 | Total comments: 3
**Body (first 200 chars):** ## Motivation  #445 fixes the immediate Hermes provider-default drift conservatively: new `config.yaml` files receive safe defaults, while existing legacy files are detected and users receive explicit
**Key quotes:**
  - [AxDSan] @dplush agreed on the direction. Versioned, provenance-aware migrations for auto-seeded config.yaml is the right call.  The current auto-seed path (`c...
  - [dplush] @AxDSan  Yes, I’m happy to take a first pass at the migration contract. 👍 I’d keep the first step de...
  - [dplush] First-pass migration contract proposal:  **Format.** New auto-seeded files get a root `config_versio...

---
## #434: [BUG] Overuse of canonical memories in skill usage and development
**Author:** jbienz | **Created:** 2026-07-08T22:34:10Z | **Updated:** 2026-07-15T14:34:13Z
**Labels:** bug
**Engagement:** WARM | AxDSan replies: 4 | Community replies: 3 | Total comments: 7
**Body (first 200 chars):** ## Description  I understand that canonical memories are intended to be "single truths". Things like my name, or my birthdate. I see Hermes creating tons of canonical memories around what seem to be s
**Key quotes:**
  - [dplush] Thanks for the detailed write-up, @jbienz. This matches a failure mode I’ve been worried about too....
  - [AxDSan] @jbienz related to #435 (canonical deletion). The overuse of canonical in skill development is a valid concern — canonical should be a curated, durabl...
  - [jbienz] Hey @AxDSan, thanks for following up. Can you please elaborate on this part?:  > I'll add a note in...
  - [AxDSan] @jbienz appreciate the patience here. Let me unpack both of your questions.  On the "note in skill authoring docs": I meant a section in the Mnemosyne...
  - [AxDSan] @jbienz this is a real quality issue. The canonical memory extractor is too aggressive, creating entries for skill usage patterns that duplicate what'...

---
## #408: Referential integrity: missing PRAGMA foreign_keys=ON + missing FK on gists.memory_id
**Author:** Iman-Sharif | **Created:** 2026-07-04T16:26:37Z | **Updated:** 2026-07-17T03:46:58Z
**Labels:** bug
**Engagement:** WARM | AxDSan replies: 2 | Community replies: 5 | Total comments: 7
**Body (first 200 chars):** # Referential integrity: missing `PRAGMA foreign_keys=ON` + missing FK on `gists.memory_id`  ## Summary  Mnemosyne silently accumulates orphaned rows because SQLite foreign key enforcement is never en
**Key quotes:**
  - [dplush] I can confirm this from another real Mnemosyne DB.  Read-only check on my local Hermes/Mnemosyne ins...
  - [AxDSan] This is a real issue. The gists table has an implicit FK to episodic_memory via memory_id but no actual FOREIGN KEY constraint, and PRAGMA foreign_key...
  - [AxDSan] Part 1 fixed in 9b2b747: PRAGMA foreign_keys=ON added to both connection factories (beam.py and memory.py). This ensures FK enforcement is active at t...
  - [dplush] Before preparing a schema migration for #408, I found a contract conflict that seems worth resolving...
  - [gabriel-belmonte] ## Reproducible evidence + root cause (real instance)  I can reproduce the embedding FK warning on a...

---
## #403: [FEATURE] Survive Hermes Update
**Author:** jbienz | **Created:** 2026-07-02T23:05:36Z | **Updated:** 2026-07-17T15:22:58Z
**Labels:** enhancement
**Engagement:** WARM | AxDSan replies: 2 | Community replies: 6 | Total comments: 8
**Body (first 200 chars):** ## Problem / Motivation  I'm really having issues keeping mnemosyne running though a hermes update (Docker).   The way I'm doing updates is:  1. Edit compose.yml or dockerfile and change the image ver
**Key quotes:**
  - [dplush] I ran into a very similar problem on my Hermes Docker setup.  What worked best for me was to avoid i...
  - [jbienz] Thank you @dplush. It sounds like this has been at least somewhat planned for with the wrapper mode....
  - [dplush] Thanks, that makes total sense. I agree my version is still too much “operator notes” and not enough...
  - [AxDSan] Solid feature request. The install flow currently lets Hermes auto-update wipe the bundled SKILL.md. The fix from #424 (install bundled memory overrid...
  - [AxDSan] @jbienz this is a real pain point. The root cause is that `hermes update` does a full `uv pip install -e .[all]` which reinstalls Hermes from scratch,...

---
## #372: Add general-purpose health check endpoint for mnemosyne-hermes provider
**Author:** bedpan | **Created:** 2026-06-22T20:09:27Z | **Updated:** 2026-07-11T13:25:49Z
**Labels:** none
**Engagement:** WARM | AxDSan replies: 2 | Community replies: 1 | Total comments: 3
**Body (first 200 chars):** ## Problem  There is no built-in way to query the health of a running Mnemosyne instance from outside the process. When the provider degrades — dangling symlink, read-only DB, broken writes, missing p
**Key quotes:**
  - [AxDSan] @bedpan sorry for the delay, was IRL busy the last few days.  This is overdue. The 48h threshold you settled on through trial-and-error is exactly the...
  - [AxDSan] The mnemosyne diagnose command (mnemosyne diagnose --fix) and the new hygiene module (mnemosyne hygiene audit) together cover most of what a health ch...
  - [bedpan] ## Update: healthcheck still actively necessary (Jul 7, 2026)  Just wanted to share an update — our...

---
## #370: [FEATURE] Add bootstrap command to import pre-Mnemosyne session history
**Author:** gergeisabo | **Created:** 2026-06-22T18:01:40Z | **Updated:** 2026-07-08T21:26:11Z
**Labels:** none
**Engagement:** WARM | AxDSan replies: 3 | Community replies: 2 | Total comments: 5
**Body (first 200 chars):** ## Problem  When a user installs Mnemosyne on an existing Hermes Agent deployment, all conversation history from before the installation date is invisible. Mnemosyne only captures memories from the mo
**Key quotes:**
  - [AxDSan] @gergeisabo sorry for the delay, was IRL busy the last few days.  Real pain point. "Mnemosyne starts empty and the user perceives it as not working" i...
  - [Daniel15] I'd like to see this too. I'm surprised this feature didn't already exist.  In addition to the sessi...
  - [AxDSan] @Daniel15 good question on MEMORY.md and USER.md. Yes, the bootstrap command should cover those too — they're essentially pre-existing memory context...
  - [Daniel15] Thanks! I'll try mnemosync once this feature is available.   On July 1, 2026 3:03:45 PM PDT, Abdia...
  - [AxDSan] This is a solid feature ask. The install/import path from pre-Mnemosyne history (Hindsight exports, OpenClaw logs, raw session JSON) is a real onboard...

---
## #329: [BUG] Memory provider tools sometimes not injected
**Author:** jbienz | **Created:** 2026-06-16T08:10:08Z | **Updated:** 2026-07-11T13:25:48Z
**Labels:** bug
**Engagement:** WARM | AxDSan replies: 5 | Community replies: 6 | Total comments: 11
**Body (first 200 chars):** ❕**NOTE:** This is likely a Hermes bug but submitting here as well in case it's not. I have already submitted to the Hermes team at:  https://github.com/NousResearch/hermes-agent/issues/47119   ## Bug
**Key quotes:**
  - [dplush] I agree this looks more like a Hermes tool-surface/session-init issue than a Mnemosyne provider bug....
  - [AxDSan] @jbienz thanks for the thorough report and for cross-posting to Hermes. dplush's analysis is spot on.  We have verified from the Mnemosyne side that t...
  - [AxDSan] @jbienz sorry for the delay, was IRL busy the last few days.  Tracking upstream at NousResearch/hermes-agent#47119. Last status I saw on that thread:...
  - [jbienz] Thanks @AxDSan. I agree with adding that to the health check, and I think a bump would be great. FYI...
  - [AxDSan] @jbienz checked Hermes #47119 — still open, no movement since mid-June. But you mentioned v0.17.0 seemed improved, which lines up with dplush's analys...

---
## #327: Map Hermes gateway identity into Mnemosyne provider scoping
**Author:** dplush | **Created:** 2026-06-16T03:15:11Z | **Updated:** 2026-07-14T13:03:52Z
**Labels:** none
**Engagement:** WARM | AxDSan replies: 5 | Community replies: 5 | Total comments: 10
**Body (first 200 chars):** Hermes passes more runtime identity information to memory providers than just `session_id`.  Provider initialization can receive values such as:  - `platform` - `user_id` - `user_id_alt` - `agent_iden
**Key quotes:**
  - [AxDSan] @dplush High strategic value, zero urgency.  The gateway identity fields (platform, user_id, agent_identity, workspace) are available in the provider...
  - [jbienz] I came to request similar as this is highly important to me. My son and my girlfriend also use Herme...
  - [AxDSan] @dplush + the parent who commented about family memory mixing: sorry for the delay, was IRL busy the last few days.  Reassigning my own label. The par...
  - [jbienz] Hey @AxDSan, wanted to thank you again for prioritizing this issue. This one is super important to m...
  - [AxDSan] @jbienz not waiting on anything external — the Hermes gateway already passes platform, user_id, agent_identity, and workspace to the provider. The imp...

---
## #326: Use Hermes queue_prefetch/prefetch as a real background recall cache
**Author:** dplush | **Created:** 2026-06-16T03:15:10Z | **Updated:** 2026-07-15T12:46:47Z
**Labels:** none
**Engagement:** WARM | AxDSan replies: 3 | Community replies: 2 | Total comments: 5
**Body (first 200 chars):** Hermes now has a clear two-step memory flow:  1. after a turn, it calls `queue_prefetch_all(user_text)` 2. before the next model call, it calls `prefetch_all(query)` and injects the returned context i
**Key quotes:**
  - [AxDSan] @dplush Agreed this is strategically the right direction  -  Hermes' two-step memory flow is a natural fit for Mnemosyne's recall pipeline.  The chall...
  - [AxDSan] @dplush sorry for the delay, was IRL busy the last few days.  Strategic value is high: Hermes' two-step memory flow (queue_prefetch -> prefetch) is th...
  - [dplush] Yes, I’d prefer the Hermes-side RFC first.  The prefetch/cache behavior depends more on the provider...
  - [AxDSan] @dplush checking in on this one. The Hermes-side RFC for the prefetch cache hook is still pending. Any update on the Hermes side? Once the contract is...
  - [dplush] @AxDSan  I checked the current Hermes core: the provider contract already has `queue_prefetch(query,...

---
## Summary Statistics
- Total open issues: 16
- WARM (maintainer engaged): 13
- LUKEWARM (community discussion, no maintainer): 1
- COLD (no engagement): 2
- Issues with 'bug' label: 5
