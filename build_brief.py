#!/usr/bin/env python3
"""Build CEO Brief HTML for Mnemosyne board - June 30 2026"""
import json

issues = {}
for N in [387, 386, 384, 383, 382, 377, 372, 371, 370, 365, 360, 359, 358, 355, 329, 328, 327, 326, 308]:
    with open(f'/tmp/issue{N}.json') as f:
        issues[N] = json.load(f)

prs = {}
for N in [369, 367, 364, 363, 356]:
    with open(f'/tmp/pr{N}.json') as f:
        prs[N] = json.load(f)

now = "2026-06-30"

html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
@page {{
  size: A4;
  margin: 2cm 1.5cm;
}}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 10pt;
  line-height: 1.5;
  color: #1f2937;
  background: #fff;
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
}}
h1 {{ font-size: 18pt; margin-bottom: 4px; color: #111827; }}
h2 {{ font-size: 14pt; margin-top: 28px; margin-bottom: 8px; border-bottom: 2px solid #e5e7eb; padding-bottom: 4px; color: #1f2937; }}
h3 {{ font-size: 12pt; margin-top: 20px; margin-bottom: 6px; color: #374151; }}
.meta {{ color: #6b7280; font-size: 9pt; margin-bottom: 16px; }}
.badge {{ display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 8pt; font-weight: 600; margin-right: 4px; }}
.badge-green {{ background: #d1fae5; color: #065f46; }}
.badge-red {{ background: #fee2e2; color: #991b1b; }}
.badge-yellow {{ background: #fef3c7; color: #92400e; }}
.badge-blue {{ background: #dbeafe; color: #1e40af; }}
.badge-gray {{ background: #f3f4f6; color: #374151; }}
.badge-purple {{ background: #ede9fe; color: #5b21b6; }}
.item-card {{ background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px 16px; margin: 10px 0; }}
.item-card.bug {{ border-left: 4px solid #ef4444; }}
.item-card.enhancement {{ border-left: 4px solid #3b82f6; }}
.item-card.pr {{ border-left: 4px solid #10b981; }}
.item-card.cold {{ border-left: 4px solid #f97316; }}
.item-title {{ font-weight: 600; font-size: 11pt; }}
.item-author {{ color: #6b7280; font-size: 9pt; }}
.key-finding {{ background: #fffbeb; border: 1px solid #fcd34d; border-radius: 6px; padding: 10px 14px; margin: 8px 0; font-size: 10pt; }}
.contrarian {{ background: #fffbeb; border-left: 4px solid #f59e0b; padding: 10px 14px; margin: 8px 0; font-size: 10pt; }}
.contrarian::before {{ content: "CONTRARIAN: "; font-weight: 700; color: #b45309; }}
.verdict {{ background: #1f2937; color: #f9fafb; border-radius: 6px; padding: 10px 14px; margin: 8px 0; font-size: 10pt; }}
table {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 9pt; }}
th, td {{ border: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; }}
th {{ background: #f3f4f6; font-weight: 600; }}
.matrix-very-high {{ background: #d1fae5; }}
.matrix-high {{ background: #ecfdf5; }}
.matrix-med {{ background: #fef3c7; }}
.matrix-low {{ background: #fee2e2; }}
.conflict-badge {{ display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 7pt; font-weight: 600; }}
.conflict-badge.clear {{ background: #d1fae5; color: #065f46; }}
.conflict-badge.overlap {{ background: #fef3c7; color: #92400e; }}
.bug-scan {{ background: #f0fdf4; border: 1px solid #86efac; border-radius: 6px; padding: 10px 14px; margin: 8px 0; font-size: 10pt; }}
.stats-row {{ display: flex; gap: 12px; margin: 12px 0; }}
.stat-box {{ flex: 1; background: #f3f4f6; border-radius: 8px; padding: 12px; text-align: center; }}
.stat-num {{ font-size: 22pt; font-weight: 700; color: #111827; }}
.stat-label {{ font-size: 8pt; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; }}
.wave-box {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px 16px; margin: 10px 0; }}
.wave-box.w1 {{ border-left: 4px solid #ef4444; }}
.wave-box.w2 {{ border-left: 4px solid #f59e0b; }}
.wave-box.w3 {{ border-left: 4px solid #6b7280; }}
</style>
</head>
<body>

<h1>CEO Brief: Mnemosyne Board</h1>
<div class="meta">
<strong>Date:</strong> {now} | <strong>Repo:</strong> mnemosyne-oss/mnemosyne (formerly AxDSan/mnemosyne) | <strong>Mode:</strong> Cron Triage (Analysis Only)
</div>

<h2>Executive Summary</h2>

<div class="stats-row">
<div class="stat-box"><div class="stat-num">5</div><div class="stat-label">Open PRs</div></div>
<div class="stat-box"><div class="stat-num">19</div><div class="stat-label">Open Issues</div></div>
<div class="stat-box"><div class="stat-num">24</div><div class="stat-label">Total Open</div></div>
<div class="stat-box"><div class="stat-num">5/5</div><div class="stat-label">PRs CI Green</div></div>
</div>

<div class="key-finding">
<strong>Key Finding:</strong> Post-release stuck board. All 5 PRs are CI green but <strong>DIRTY</strong> behind the v3.11.0 version bump (landed today). 4 of 5 PRs already APPROVED by AxDSan. One PR (#356) is CLI-merged stale — fix already on main. The merge bottleneck is mechanical (rebase), not code quality or review.
</div>

<div class="key-finding">
<strong>Engagement State:</strong> 17/19 issues WARM (maintainer replied). 2 COLD issues filed today by freeformz (returning reporter). All 5 PRs are COLD (no maintainer replies on PR threads, though reviews were left). dplush dominates with 3 open issues + 1 approved PR.
</div>

<div class="contrarian">
The board is in an unusual state: nearly everything is already warmed up, reviewed, or replied to. The real bottleneck is rebase friction. Every PR needs a rebase cycle, then re-check, then merge. The <strong>real risk</strong> is not code quality — it's that small PRs from first-time/first-PR contributors (doziedotdev #363, timbeaulac #364) sit cold while the maintainer is IRL-busy. The rebase-after-approval gap applies to all 4 approved PRs — each was approved at an earlier commit and needs re-verification after rebase.
</div>

<h2>Weighted Decision Matrix</h2>
<p style="color:#6b7280;font-size:9pt;">Formula: 0.3*User Impact + 0.25*Strategic + 0.15*Ease + 0.15*Safety + 0.15*Urgency</p>

<table>
<tr>
  <th>#</th><th>Item</th><th>Type</th><th>User</th><th>Strat</th><th>Ease</th><th>Safe</th><th>Urgency</th><th>Score</th><th>Call</th>
</tr>
<tr>
  <td>#367</td><td>Wrapper install mode</td><td>PR</td>
  <td class="matrix-high">7</td><td class="matrix-high">8</td><td class="matrix-very-high">8</td><td class="matrix-very-high">9</td><td class="matrix-high">8</td>
  <td class="matrix-very-high"><strong>7.85</strong></td><td>MERGE</td>
</tr>
<tr>
  <td>#364</td><td>Tool whitelist</td><td>PR</td>
  <td class="matrix-high">8</td><td class="matrix-high">7</td><td class="matrix-very-high">8</td><td class="matrix-very-high">9</td><td class="matrix-high">7</td>
  <td class="matrix-very-high"><strong>7.70</strong></td><td>MERGE</td>
</tr>
<tr>
  <td>#363</td><td>CLI bank-aware</td><td>PR</td>
  <td class="matrix-high">7</td><td class="matrix-high">7</td><td class="matrix-very-high">9</td><td class="matrix-very-high">9</td><td class="matrix-high">8</td>
  <td class="matrix-very-high"><strong>7.75</strong></td><td>MERGE*</td>
</tr>
<tr>
  <td>#369</td><td>Tool schema refactor</td><td>PR</td>
  <td class="matrix-high">7</td><td class="matrix-very-high">9</td><td class="matrix-med">5</td><td class="matrix-high">7</td><td class="matrix-high">6</td>
  <td class="matrix-high"><strong>6.85</strong></td><td>MERGE</td>
</tr>
<tr>
  <td>#356</td><td>Profile install fix</td><td>PR</td>
  <td class="matrix-high">6</td><td class="matrix-high">6</td><td class="matrix-very-high">10</td><td class="matrix-very-high">10</td><td class="matrix-med">3</td>
  <td class="matrix-high"><strong>6.75</strong></td><td>CLOSE</td>
</tr>
<tr>
  <td>#386</td><td>Veracity silently dropped</td><td>Bug</td>
  <td class="matrix-high">7</td><td class="matrix-med">5</td><td class="matrix-high">7</td><td class="matrix-very-high">9</td><td class="matrix-high">7</td>
  <td class="matrix-high"><strong>6.90</strong></td><td>ENGAGE</td>
</tr>
<tr>
  <td>#387</td><td>Content mutation</td><td>Bug</td>
  <td class="matrix-high">7</td><td class="matrix-med">5</td><td class="matrix-med">5</td><td class="matrix-high">7</td><td class="matrix-high">7</td>
  <td class="matrix-med"><strong>6.20</strong></td><td>ENGAGE</td>
</tr>
</table>

<p style="color:#6b7280;font-size:8pt;">
* #363 has DISMISSED review (not re-approved). Contributor addressed feedback. Needs re-review after rebase.<br>
#356: CLI-merge stale PR. All commits on main. Close with explanation.<br>
Board is post-release stuck: all PRs DIRTY behind v3.11.0. Merge order = rebase order, not priority order.
</p>

<h2>Conflict Analysis</h2>

<table>
<tr><th>PR Pair</th><th>Status</th><th>Details</th></tr>
<tr>
  <td>#369 vs #367</td>
  <td><span class="conflict-badge clear">NO CONFLICT</span></td>
  <td>Tool schemas vs installer. Different filesets entirely.</td>
</tr>
<tr>
  <td>#369 vs #364</td>
  <td><span class="conflict-badge overlap">OVERLAPPING DOMAIN</span></td>
  <td>Both touch <code>hermes_memory_provider/__init__.py</code>. #369 refactors tool schemas, #364 adds whitelist. Must merge #369 first (it creates the shared schema), then #364 rebases onto it.</td>
</tr>
<tr>
  <td>#369 vs #363</td>
  <td><span class="conflict-badge clear">NO CONFLICT</span></td>
  <td>No file overlap.</td>
</tr>
<tr>
  <td>#367 vs #364</td>
  <td><span class="conflict-badge overlap">OVERLAPPING DOMAIN</span></td>
  <td>Both touch CHANGELOG.md. Standard post-release conflict — resolves on rebase.</td>
</tr>
<tr>
  <td>#367 vs #363</td>
  <td><span class="conflict-badge overlap">OVERLAPPING DOMAIN</span></td>
  <td>Both touch CHANGELOG.md <strong>and</strong> <code>mnemosyne/__init__.py</code> (version constant). Merge #367 first (profile-install, more urgent for retention), then #363 rebases.</td>
</tr>
<tr>
  <td>#364 vs #363</td>
  <td><span class="conflict-badge overlap">OVERLAPPING DOMAIN</span></td>
  <td>Both touch CHANGELOG.md. Resolves on rebase.</td>
</tr>
</table>

<h2>Deep Dives</h2>

<h3>PRs</h3>

<div class="item-card pr">
<div class="item-title"><span class="badge badge-green">APPROVED</span> PR #369 — Refactor tool schemas to a single source of truth</div>
<div class="item-author">rcsaquino | +723/-1203 | 4 files | 1 review (APPROVED) | CI: GREEN | MERGE: DIRTY</div>
<p><strong>What it does:</strong> Centralizes all 33 tool schemas into <code>mnemosyne.tool_schemas</code>. Fixes the standalone MCP server returning empty tools list. Replaces scattered schema definitions across Hermes provider files with a single importable source.</p>
<div class="contrarian">
Large net deletion (-1203) from deduplication. 723 additions across 4 files. The schema consolidation is architecturally correct, but any PR touching <code>hermes_memory_provider/__init__.py</code> has rebase risk — three other open items also touch this file. Approved by AxDSan at an earlier commit. Rebasing after v3.11.0 should be mechanical (CHANGELOG/version) but verify the provider file merges cleanly. 
<strong>Recommended merge order:</strong> First in the provider-file sequence (#369 before #364) to let #364 rebase onto the consolidated schema.
</div>
<div class="verdict">RECOMMEND: MERGE (after rebase). Re-verify the rebase diff on hermes_memory_provider/__init__.py — full re-review not needed.</div>
</div>

<div class="item-card pr">
<div class="item-title"><span class="badge badge-green">APPROVED</span> PR #367 — Add wrapper install mode for Hermes plugin discovery</div>
<div class="item-author">dplush | +313/-89 | 5 files | 2 reviews (DISMISSED then APPROVED) | CI: GREEN | MERGE: DIRTY</div>
<p><strong>What it does:</strong> Adds <code>mnemosyne-hermes install --mode wrapper</code> for Docker/read-only Hermes venvs. Generates a persistent plugin shim that imports from a selected Python environment. Extends status reporting to show install mode and detect stale wrappers.</p>
<div class="contrarian">
dplush is the most active contributor on the board (3 open issues + this PR). The wrapper install mode addresses a real production pain point (Docker deployments). dplush added the CHANGELOG entry and bumped the target release after rebasing. APPROVED after rebase re-review. The only concern: adding a new install mode increases the maintenance surface area for a feature that's already complex (profile-aware install, symlinks, multiple Python environments). But the wrappers are shim-only and don't add runtime complexity.
</div>
<div class="verdict">RECOMMEND: MERGE (after rebase). Chip away at the CHANGELOG conflict — simplest rebase of the bunch.</div>
</div>

<div class="item-card pr">
<div class="item-title"><span class="badge badge-yellow">DISMISSED</span> PR #363 — make mnemosyne CLI bank-aware under profile_isolation</div>
<div class="item-author">doziedotdev | +145/-4 | 4 files | 1 review (DISMISSED) | CI: GREEN | MERGE: DIRTY</div>
<p><strong>What it does:</strong> Fixes <code>hermes mnemosyne &lt;stats|inspect|sleep|export&gt;</code> to use the correct per-profile bank instead of always binding to the default. Self-shipped fix by doziedotdev who filed #362 with full line-level RCA 15 minutes before opening this PR.</p>
<div class="contrarian">
Same-minute self-ship from reporter-to-fixer pattern (doziedotdev pattern from Jun 19 2026). The DISMISSED review followed by contributor addressing feedback (added CHANGELOG + version bump) means this is effectively ready. The contributor wrote "the PR should be good to merge once you..." — they are waiting. Retention risk: this is doziedotdev's first PR. Left unreviewed/pending since the dismissal, the contributor is watching the board. 8+ hours cold after addressing review feedback is within the kirocop retention window.
<strong>Urgency +1</strong> for first-PR retention.
</div>
<div class="verdict">RECOMMEND: RE-APPROVE then MERGE (after rebase). Smallest diff (+145/-4). Do the re-review promptly — the rebase diff will be tiny.</div>
</div>

<div class="item-card pr">
<div class="item-title"><span class="badge badge-green">APPROVED</span> PR #364 — Add Hermes Mnemosyne tool whitelist</div>
<div class="item-author">timbeaulac | +234/-30 | 4 files | 2 reviews (DISMISSED then APPROVED) | CI: GREEN | MERGE: DIRTY</div>
<p><strong>What it does:</strong> Adds a provider-local <code>memory.mnemosyne.tools</code> allowlist for Hermes tool exposure. Implements AxDSan's requested shape from #358: filter-before-registration, loud validation, and both Hermes provider paths updated.</p>
<div class="contrarian">
timbeaulac's first contribution to the project (filed #358, then shipped this PR). Guidance-to-PR conversion pattern — they filed a feature request with clear description, got direction from AxDSan, and shipped a clean implementation. APPROVED after rebase re-review. The tool whitelist addresses a real pain point (23 tool schemas injected on every turn). One contrarian concern: the whitelist config adds a new YAML config surface in an already config-heavy system. But AxDSan specifically requested this shape, so the decision is settled.
<strong>Recommended merge order:</strong> After #369 (schema refactor) since both touch the provider file.
</div>
<div class="verdict">RECOMMEND: MERGE (after rebase). Rebase will be straightforward — just CHANGELOG + provider file merge.</div>
</div>

<div class="item-card pr">
<div class="item-title"><span class="badge badge-gray">STALE</span> PR #356 — fix(install): support named Hermes profiles in plugin install</div>
<div class="item-author">tvinagre | +883/-17 | 7 files | 2 reviews (APPROVED) | CI: GREEN | MERGE: DIRTY</div>
<p><strong>What it does:</strong> Makes <code>mnemosyne-install</code> and <code>mnemosyne-hermes install</code> scan all Hermes profiles for memory.provider: mnemosyne and create plugin symlinks in each.</p>
<div class="contrarian">
<strong>CLI-MERGE STALE PR:</strong> The core fix commit (89c444a) is already on main. All commits on the PR branch are ancestors of main. The diff is empty between the merge base and main. The PR should be CLOSED, not merged. Post a comment thanking tvinagre and confirming the fix landed in commit 89c444a. GitHub will NOT auto-close this PR since it was CLI-merged — must close manually with a comment. tvinagre is a first-time contributor (tvinagre filed #365 for this issue). The PR remaining OPEN is a retention risk: the contributor sees their PR unmerged while their fix is already live with no acknowledgment.
</div>
<div class="verdict">RECOMMEND: CLOSE (with comment). Fix already on main in commit 89c444a. Thank tvinagre, link the commit.</div>
</div>

<h3>New Cold Issues</h3>

<div class="item-card cold">
<div class="item-title"><span class="badge badge-red">COLD</span> #386 — veracity argument silently dropped on remember/import</div>
<div class="item-author">freeformz (returning reporter) | Filed today | 0 maintainer replies</div>
<p><strong>What it reports:</strong> The <code>veracity</code> argument to <code>remember</code>/<code>import</code> is silently coerced to "unknown" regardless of the value passed. No error or warning. Documented to give recall boost (stated > inferred > unknown) but the feature is effectively dead code at the API level.</p>
<div class="contrarian">
freeformz is a returning thorough reporter (#377 filed Jun 24, detailed analysis with Claude-assisted tracing). This bug is real and easy to verify: grep for veracity handling in the remember/import tool handlers. The fix is likely in the MCP tool handlers or the Hermes provider wrappers. Since veracity is an existing documented feature that's silently broken, the fix is a genuine API bug — not an enhancement.
<strong>Engagement urgency:</strong> freeformz's #377 got a maintainer reply (WARM). Filing two more bugs within 6 days signals active use. A quick acknowledge within 24h converts the reporter-to-fixer trajectory.
</div>
<div class="verdict">RECOMMEND: ENGAGE today. Acknowledge, confirm from source, estimate fix timeline.</div>
</div>

<div class="item-card cold">
<div class="item-title"><span class="badge badge-red">COLD</span> #387 — Stored content mutated: derived entity annotations appended to content field</div>
<div class="item-author">freeformz (returning reporter) | Filed today | 0 maintainer replies</div>
<p><strong>What it reports:</strong> Mnemosyne appends <code>[DATES: ...]</code> and <code>[DURATIONS: ...]</code> annotations directly to the stored <code>content</code> field. This mutates the caller's data — stored content is no longer byte-identical to what was written. No opt-out API.</p>
<div class="contrarian">
This is a design tension, not a simple bug. The entity annotations are useful for recall relevance, but storing them in the content field (rather than a separate metadata/annotations field) pollutes exported/retrieved content. The fix path has three options: (1) strip annotations on get/export, (2) move to separate annotation storage, (3) add an API flag to control behavior. Option (2) is architecturally cleanest but requires a schema migration. Option (1) is a stopgap. The freeformz deserves a substantive response acknowledging the design tradeoff — not a quick workaround.
</div>
<div class="verdict">RECOMMEND: ENGAGE today. Acknowledge the design tension, propose direction (option 1 as stopgap, option 2 as longer fix), ask for input.</div>
</div>

<h2>Bug Scan</h2>

<div class="bug-scan">
<strong>Open bugs that need attention:</strong><br><br>
• <strong>#387</strong> (freeformz) — Content mutation via entity annotations. Design-level issue, not a quick fix.<br>
• <strong>#386</strong> (freeformz) — Veracity silently dropped. Trivial fix expected (5-15 lines in tool handler).<br>
• <strong>#384</strong> (codxt) — Diagnostics report counts from fallback DB paths. WARM (AxDSan acknowledged alongside #383).<br>
• <strong>#383</strong> (codxt) — SSE transport crashes with Starlette during /sse + /messages flow. WARM.<br>
• <strong>#382</strong> (kirocop) — WAL checkpoint blocked after session ends. WARM (AxDSan replied).<br>
• <strong>#371</strong> (gergeisabo) — recall() session filter bug. WARM (AxDSan confirmed from source with line numbers).<br>
• <strong>#360</strong> (laurinaitis) — MCP server empty tools list. Fixed by #369 (schema refactor).<br>
• <strong>#329</strong> (jbienz) — Tools not injected. WARM. Tracking upstream Hermes #47119.<br><br>

<strong>CI status:</strong> All 5 PRs CI green. No CI regressions detected. All PRs DIRTY due to v3.11.0 bump — no infra issues.<br>
<strong>Test regressions:</strong> None detected in the current fetch scope (code inspection blocked per cron scope rules).
</div>

<h2>Retention Risks</h2>

<div class="key-finding">
<strong>1. CLI-merge stale PR (#356) — tvinagre first-time contributor:</strong> fix landed on main, PR shows OPEN. The contributor thinks their PR is unmerged. Post a confirmation comment ASAP. HIGH risk.<br><br>
<strong>2. doziedotdev #363 — first-PR cold after addressing review:</strong> DISMISSED review, contributor addressed ALL feedback (CHANGELOG + version), waiting for re-approval. 8+ hours cold on a self-ship PR. MEDIUM-HIGH risk (kirocop pattern).<br><br>
<strong>3. freeformz #386/#387 — two new COLD bugs from returning reporter:</strong> Filed #377 (WARM), now two more detailed bugs in one day. Quick acknowledgment converts to recurring contributor. MEDIUM risk.<br><br>
<strong>4. dplush over-concentration:</strong> 3 open issues + 1 approved PR from one contributor (17% of issues + 20% of PRs). Low risk (dplush is established), but bus factor concern in the contrarian section.<br><br>
<strong>5. Post-release rebase friction for all PRs:</strong> Every PR needs a manual rebase. The friction of a version-bump rebase (just CHANGELOG + __init__.py) is low, but the psychological friction of "I have to do something and wait again" after already waiting for review is real. Batch-merge in one wave to minimize this.
</div>

<h2>Action Plan</h2>

<div class="wave-box w1">
<h3>Wave 1 — Today [HUMAN]</h3>
<ol>
<li><strong>Acknowledge #386 and #387</strong> (freeformz) — Reply to both bugs. Confirm #386 is a real bug (veracity silently dropped), estimate fix timeline. For #387, acknowledge the design tension and propose direction.</li>
<li><strong>Close #356</strong> (tvinagre) — Post closing comment: fix already on main in commit 89c444a. Thank tvinagre for the work. Close the PR.</li>
<li><strong>Re-review #363</strong> (doziedotdev) — Review the rebase diff. Contributor addressed all feedback. Re-approve and merge.</li>
<li><strong>Rebase all 5 PRs</strong> — Ask all 4 remaining PR authors to rebase onto main (v3.11.0). For #369 (provider file merge), note the merge order: #369 first, then #364.</li>
</ol>
</div>

<div class="wave-box w2">
<h3>Wave 2 — This Week [HUMAN]</h3>
<ol>
<li><strong>Merge rebased PRs</strong> — After each contributor rebases, verify rebase diff, re-approve if clean, merge in order: #367 (wrapper install, smallest blocker) → #369 (schema refactor, provider file gate) → #364 (tool whitelist, after #369) → #363 (bank-aware CLI, re-approved).</li>
<li><strong>Follow up on #386 fix</strong> — Either fix the veracity handler directly (5-15 line change) or greenlight freeformz to ship a PR if they offer.</li>
<li><strong>Engage on #382</strong> (kirocop, WAL checkpoint) — kirocop is a retention-sensitive contributor (previous frustration pattern). A substantive reply on the thread matters.</li>
</ol>
</div>

<div class="wave-box w3">
<h3>Wave 3 — Next Week [HUMAN]</h3>
<ol>
<li><strong>#387 content mutation design decision</strong> — Settle on approach (strip-on-get vs separate annotations storage vs API flag) after community input.</li>
<li><strong>dplush's remaining issues</strong> (#328 sync_turn, #327 gateway identity scoping, #326 prefetch cache) — strategic but non-urgent. Batch follow-up.</li>
<li><strong>#308 Phase 2 reindex</strong> (Milgauss) — Deferred by mutual agreement. Check if demand has emerged.</li>
</ol>
</div>

<h2>State of the Repo</h2>

<div class="key-finding">
<strong>Health:</strong> GOOD but stuck. The board is clean in the sense that most items have maintainer engagement and all PRs have reviews. The v3.11.0 release (today) is the single cause of the stuck state — not code quality, not review bottleneck.<br><br>
<strong>Contributor engagement:</strong> Mostly WARM. 17/19 issues have maintainer replies. The two COLD items (freeformz #386/#387) are from a returning high-signal reporter — quick acknowledgment is important.<br><br>
<strong>PR velocity:</strong> 0 PRs merged in the last 24h. All 4 active PRs approved but stuck behind the version bump. The CLI-merge of #356's fix happened without GH tracking — the contributor doesn't know their work shipped.<br><br>
<strong>Risk score:</strong> 2/5 (low). The board risk is mechanical, not technical. No blocking bugs, no CI failures, no contentious disagreements. The main risk is contributor retention from cold PRs and stale PR display state.<br><br>
<strong>What changed since last brief:</strong> v3.11.0 shipped (PR #369's schema refactor, #367's wrapper install, #385's tool whitelist fix, plus profile isolation fixes). The release bumped the version and DIRTYed all remaining open PRs.
</div>

</body>
</html>'''

with open('/tmp/ceo-brief-2026-06-30.html', 'w') as f:
    f.write(html)
print(f'HTML written: {len(html)} chars')
