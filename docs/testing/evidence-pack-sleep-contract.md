# Reproducing the evidence-pack Sleep boundaries

This is an executable characterization of the existing contract discussed in
[#545](https://github.com/mnemosyne-oss/mnemosyne/pull/545),
[#789](https://github.com/mnemosyne-oss/mnemosyne/issues/789),
[#979](https://github.com/mnemosyne-oss/mnemosyne/issues/979), and
[#1266](https://github.com/mnemosyne-oss/mnemosyne/issues/1266).
It does not implement archive recovery or select a new admission policy.
Intentional contract changes should update these expectations with their decision.

The runtime baseline is Core commit
`e1f34fb696f72d57f2a98a17714e6c8e7b81c886` with its bundled Hermes **0.7.7**
source. The original reproduction tests were added in
`ac3229013ea72d7e0edda0d3685eaf3c8db7ed49`; the baseline commit alone does not
contain the test file. The tests do not pin a machine path, read private memories,
write result files, or require the earlier local investigation artifacts.

## Setup and run

Use Python 3.10 or later in a disposable checkout that contains the tests.
From a clone of `https://github.com/mnemosyne-oss/mnemosyne.git`, these commands
select the current PR test matrix, including its two-primary ordering checks:

```sh
git fetch origin refs/pull/1327/head
git switch --detach FETCH_HEAD
git rev-parse HEAD
```

Record the printed commit when sharing results. For the immutable original
13-case reproduction, fetch and check out
`ac3229013ea72d7e0edda0d3685eaf3c8db7ed49` instead; that historical revision
does not include the strengthened two-primary checks. In a virtual environment,
install the small test/runtime dependencies and the two local packages.
Installation may require network access; the reproduction itself uses controlled
inference and does not download models or call an inference service.

```sh
python -m pip install "pytest>=7" "pytest-timeout>=2.1" numpy "PyYAML>=6" "sqlite-vec==0.1.9"
python -m pip install --no-deps -e . -e integrations/hermes
python -m pytest -q tests/test_evidence_pack_sleep_contract.py
```

Expected with NumPy, a loadable sqlite-vec extension, and the bundled Hermes
package: **13 passed**. No environment variable or separate fixture archive is
required. Without the Hermes package, only its helper companion test skips;
the 12 Core cases still run. Without a loadable sqlite-vec extension, the native
WM arms skip. Without NumPy, the module skips. A run with skips is not evidence
that the complete matrix passed; inspect the skip reasons with `-rs`.

For a source-tree run without editable installation of the Hermes package,
on a POSIX shell use the relative import path:

```sh
PYTHONPATH=.:integrations/hermes/src python -m pytest -q tests/test_evidence_pack_sleep_contract.py
```

## What the cases establish

All 12 Core cases first use real `remember()` and successful `sleep()`. They
check retained original text, the consolidation marker, completed claim and
`summary_of`. Summary text and embeddings are deliberately fixed: the original
has cosine 0.8 to the query, the first primary has cosine 1, the second primary
has cosine 0.95, and the summary is orthogonal. These values isolate contract
boundaries, not model quality.
Entity extraction is disabled during storage.

The tests then establish three outcomes, each under linear/polyphonic recall
and native-WM-index/JSON-WM configurations:

- The original reaches a wider candidate call through explicit `source='fact'`,
  but the pack removes it because it is consolidated.
- A current detail from the session already represented in `primary` reaches
  candidates but is not added as a supplemental group.
- An authorized current fact about a colleague from another session can enter
  the pack even though the question is about the user's own configuration.

`cross_session: true` is explicitly configured only for the synthetic store;
all stored rows have `scope='global'`. This is not an unauthorized-session
exclusion test and does not promote existing rows to global scope.

Every case retrieves two primary rows with `top_k=2` and compares the exact
ordered IDs `["primary", "secondary"]` against ordinary recall and the pack.
It checks actual WM recall-count changes: both primary rows are credited once,
and supplemental candidate reads do not credit the other rows. The recall wrapper
observes real calls and does not replace their results.

The thirteenth test checks only Hermes's distinctive-lexical-evidence helper
with the colleague text. It does not wire packs into `prefetch()`, run the full
provider, or measure an answering model's error.

## Limits

The explicit source filter intentionally bypasses the default consolidated-dense
exclusion. This isolates the *later pack exclusion*; it is not an end-to-end
repair of the default Sleep-recall gap. Pack grouping and semantic admission
remain separate decisions.

Polyphonic WM already reads JSON vectors, so its two WM-index configurations
are not two independent WM algorithms. This matrix does not prove every
Float32/Int8/bit or episodic backend's score/threshold parity. No threshold,
fixed hot/cold quota, model-dependent calibration, precision estimate or latency
claim follows from these small synthetic cases.
