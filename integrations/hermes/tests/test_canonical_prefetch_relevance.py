"""Real-store regressions for #1263's function-word canonical injection."""
from __future__ import annotations

import pytest

from mnemosyne.core.canonical import CanonicalStore
from mnemosyne_hermes import (
    MnemosyneMemoryProvider,
    _canonical_prefetch_rows,
    _canonical_recall_rows,
    _prefetch_has_distinctive_lexical_evidence,
    _semantic_dedup_prefetch,
)


@pytest.fixture(autouse=True)
def clear_configuration(monkeypatch):
    for key in (
        "MNEMOSYNE_PREFETCH_CANONICAL_GENERIC_TOKENS",
        "MNEMOSYNE_PREFETCH_CANONICAL_EXTRA_GENERIC_TOKENS",
        "MNEMOSYNE_PREFETCH_MIN_DISTINCTIVE_TOKENS",
        "MNEMOSYNE_PREFETCH_MIN_QUERY_COVERAGE",
        "MNEMOSYNE_PREFETCH_CANONICAL_RARE_TOKEN_MAX_FREQUENCY",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def store(tmp_path):
    result = CanonicalStore(db_path=tmp_path / "canonical.db")
    yield result
    result.conn.close()


def seed(store, rows, owner="default"):
    for name, body in rows:
        store.remember(owner, "preference", name, body)


def names(store, query):
    return [row["canonical_name"] for row in _canonical_prefetch_rows(store, "default", query)]


def test_german_function_words_do_not_inject_an_unrelated_rule(store):
    seed(store, [("board", "Für mich muss das neue Board genutzt werden.")])

    assert names(store, "muss ich für neue Passwörter im Vaultwarden sichern") == []


def test_real_content_survives_function_word_competition(store):
    seed(store, [
        ("a_board", "Für mich muss das neue Board genutzt werden."),
        ("z_password", "Passwörter werden im Vaultwarden gesichert."),
    ])

    assert names(store, "muss ich für neue Passwörter im Vaultwarden sichern") == ["z_password"]


@pytest.mark.parametrize("form", ["neue", "neuen", "neuer", "neuem", "neues"])
def test_inflected_function_word_pair_is_not_topic_evidence(store, form):
    seed(store, [("board", f"muss {form} Boardregel beachten")])

    assert names(store, f"muss {form} Passwörter Vaultwarden sichern") == []


def test_function_word_matches_cannot_crowd_content_out_of_top_three(store):
    seed(store, [(f"a_wrong_{i}", f"muss neue Boardregel Nummer {i}") for i in range(4)])
    seed(store, [("z_password", "muss neue Passwörter Vaultwarden sichern")])

    assert names(store, "muss neue Passwörter Vaultwarden verwenden") == ["z_password"]


def test_common_domain_terms_remain_useful_in_a_small_bank(store):
    seed(store, [
        ("export", "Vaultwarden Passwort exportieren"),
        ("renew", "Vaultwarden Passwort erneuern"),
        ("save", "Vaultwarden Passwort sichern"),
    ])

    assert set(names(store, "Vaultwarden Passwort")) == {"export", "renew", "save"}


@pytest.mark.parametrize(("body", "query"), [
    ("New York restaurants", "New York restaurants"),
    ("Neue Zürcher Zeitung", "Neue Zürcher Zeitung"),
    ("Deploy pipeline configuration", "deploy pipeline"),
    ("тёмную резервную копию", "найди тёмную резервную копию"),
    ("서울 숙소", "서울 숙소 알려줘"),
])
def test_named_phrases_and_other_language_positives_survive(store, body, query):
    seed(store, [("target", body)])

    assert names(store, query) == ["target"]


def test_irrelevant_query_does_not_inject_canonical_context(store):
    seed(store, [("target", "Vaultwarden Passwort sichern")])

    assert names(store, "pancakes maple syrup") == []


def test_explicit_recall_and_ordinary_prefetch_keep_their_own_rules(store):
    seed(store, [("board", "Für mich muss das neue Board genutzt werden.")])
    query = "muss ich für neue Passwörter im Vaultwarden sichern"

    assert [r["canonical_name"] for r in _canonical_recall_rows(store, "default", query)] == ["board"]
    assert _prefetch_has_distinctive_lexical_evidence(query, "Für mich muss das neue Board genutzt werden.")


def test_replacement_configuration_remains_authoritative(store, monkeypatch):
    seed(store, [("board", "muss neue Boardregel")])
    monkeypatch.setenv("MNEMOSYNE_PREFETCH_CANONICAL_GENERIC_TOKENS", "sampleowner")

    assert names(store, "muss neue Regeln") == ["board"]


@pytest.mark.parametrize("title", ["Oder", "Die Neue"])
def test_explicit_recall_can_find_titles_made_only_of_function_words(store, title):
    seed(store, [("title", title)])
    # A bounded stopword vocabulary can also cover a river/title. Dedicated
    # search and the explicit merge must remain usable for that ambiguity.
    assert [r["canonical_name"] for r in _canonical_recall_rows(store, "default", title)] == ["title"]
    assert [r["name"] for r in store.search("default", title)] == ["title"]


def test_extra_generic_tokens_still_exclude_domain_evidence(store, monkeypatch):
    seed(store, [("password", "Vaultwarden Passwort sichern")])
    monkeypatch.setenv("MNEMOSYNE_PREFETCH_CANONICAL_EXTRA_GENERIC_TOKENS", "vaultwarden,passwort")

    assert names(store, "Vaultwarden Passwort") == []


def test_owner_scope_is_preserved(store):
    seed(store, [("password", "Vaultwarden Passwort sichern")], owner="other")

    assert names(store, "Vaultwarden Passwort") == []


def test_rare_content_evidence_outweighs_common_overlap(store):
    seed(store, [(f"a_deploy_{i}", f"deployment pipeline worker{i}") for i in range(7)])
    seed(store, [("z_vault", "Vaultwarden Passwort sichern")])
    # Both candidates match two content terms. The rare Vaultwarden/password
    # evidence must survive the three-slot limit instead of losing a name tie.
    query = "deployment pipeline Vaultwarden Passwort"

    assert names(store, query)[0] == "z_vault"


def test_two_rare_terms_outrank_three_common_terms(store):
    seed(store, [(f"a_deploy_{i}", "deployment pipeline worker") for i in range(4)])
    seed(store, [("z_vault", "Vaultwarden Passwort sichern")])
    query = "deployment pipeline worker Vaultwarden Passwort"
    rows = _canonical_prefetch_rows(store, "default", query, limit=5)
    by_name = {row["canonical_name"]: row for row in rows}
    # The rare pair covers more weighted query evidence, despite matching
    # fewer terms. A count bonus must not reverse that order before limit=3.
    assert by_name["z_vault"]["keyword_score"] > by_name["a_deploy_0"]["keyword_score"]
    assert names(store, query)[0] == "z_vault"


def test_rendered_context_retains_two_rare_terms_among_three_term_matches(tmp_path):
    provider = MnemosyneMemoryProvider()
    provider.initialize("issue-1263-count-bonus", hermes_home=str(tmp_path), agent_identity="default", profile_isolation=False)
    try:
        assert provider._beam is not None
        canonical = CanonicalStore(db_path=provider._beam.db_path, conn=provider._beam.conn)
        provider._beam.canonical = canonical
        seed(canonical, [(f"a_deploy_{i}", "deployment pipeline worker") for i in range(4)])
        seed(canonical, [("z_vault", "Vaultwarden Passwort sichern")])

        block = provider.prefetch("deployment pipeline worker Vaultwarden Passwort")

        assert "Vaultwarden Passwort" in block
    finally:
        provider.shutdown()


def test_generic_overlap_does_not_boost_a_canonical_score(store):
    seed(store, [
        ("a_padded", "muss neue Vaultwarden Passwort sichern"),
        ("z_plain", "Vaultwarden Passwort sichern"),
    ])

    rows = _canonical_prefetch_rows(store, "default", "muss neue Vaultwarden Passwort backup archive")

    assert len(rows) == 2
    assert rows[0]["score"] == rows[1]["score"]
    assert rows[0]["keyword_score"] == rows[1]["keyword_score"]


def test_other_owner_corpus_does_not_change_rarity_ranking(store):
    seed(store, [(f"a_deploy_{i}", f"deployment pipeline worker{i}") for i in range(7)])
    seed(store, [("z_vault", "Vaultwarden Passwort sichern")])
    seed(store, [(f"other_{i}", "Vaultwarden Passwort sichern") for i in range(10)], owner="other")

    assert names(store, "deployment pipeline Vaultwarden Passwort")[0] == "z_vault"


def test_rendered_context_omits_function_word_rule(tmp_path):
    provider = MnemosyneMemoryProvider()
    provider.initialize("issue-1263", hermes_home=str(tmp_path), agent_identity="default", profile_isolation=False)
    try:
        assert provider._beam is not None
        canonical = CanonicalStore(db_path=provider._beam.db_path, conn=provider._beam.conn)
        provider._beam.canonical = canonical
        seed(canonical, [
            ("a_board", "Für mich muss das neue Board genutzt werden."),
            ("z_password", "Passwörter werden im Vaultwarden gesichert."),
        ])

        block = provider.prefetch("muss ich für neue Passwörter im Vaultwarden sichern")

        assert "Vaultwarden" in block
        assert "Board" not in block
    finally:
        provider.shutdown()


@pytest.mark.parametrize("ordinary_body", [
    "Vaultwarden Passwort sichern deployment pipeline worker backup archive veraltet",
    "Vaultwarden Passwort sichern export recovery server account device offline",
])
def test_rendered_context_prioritizes_canonical_over_high_scoring_preference(tmp_path, monkeypatch, ordinary_body):
    # Supported scoring configuration makes the real working-memory result
    # outrank a canonical full match under the old blended sort.
    monkeypatch.setenv("MNEMOSYNE_VEC_WEIGHT", "0.2")
    monkeypatch.setenv("MNEMOSYNE_FTS_WEIGHT", "0.2")
    monkeypatch.setenv("MNEMOSYNE_IMPORTANCE_WEIGHT", "0.6")
    provider = MnemosyneMemoryProvider()
    provider.initialize("canonical-priority", hermes_home=str(tmp_path), agent_identity="default", profile_isolation=False)
    try:
        assert provider._beam is not None
        canonical = CanonicalStore(db_path=provider._beam.db_path, conn=provider._beam.conn)
        provider._beam.canonical = canonical
        canonical_body = "Vaultwarden Passwort sichern deployment pipeline worker backup archive aktuell"
        weaker_body = "Vaultwarden sichern lokal verschlüsselt sicher aufbewahren Tresor Zugang"
        seed(canonical, [("a_weaker", weaker_body), ("z_password", canonical_body)])
        assert provider._beam.remember(ordinary_body, source="preference", importance=0.95, scope="global")

        block = provider.prefetch("Vaultwarden Passwort sichern")

        assert canonical_body in block
        assert weaker_body in block
        assert block.index(canonical_body) < block.index(weaker_body)
        assert "[CANONICAL]" in block
        assert "[owner=default home=" in block
        if "veraltet" in ordinary_body:
            assert ordinary_body not in block
        else:
            assert ordinary_body in block
            assert block.index(canonical_body) < block.index(ordinary_body)
    finally:
        provider.shutdown()


@pytest.mark.parametrize(("coverage", "canonical_first"), [("0.30", False), ("0.15", True)])
def test_prefetch_canonical_priority_requires_configured_query_coverage(tmp_path, monkeypatch, coverage, canonical_first):
    monkeypatch.setenv("MNEMOSYNE_PREFETCH_MIN_QUERY_COVERAGE", coverage)
    provider = MnemosyneMemoryProvider()
    provider.initialize("canonical-coverage", hermes_home=str(tmp_path), agent_identity="default", profile_isolation=False)
    try:
        assert provider._beam is not None
        canonical = CanonicalStore(db_path=provider._beam.db_path, conn=provider._beam.conn)
        provider._beam.canonical = canonical
        bodies = ["Vaultwarden Passwort Tresorregel", "deployment pipeline Baustufe", "archive backup Archivregel"]
        seed(canonical, [(str(i), body) for i, body in enumerate(bodies)])
        ordinary_body = "Vaultwarden Passwort deployment pipeline worker recovery archive backup export offline"
        assert provider._beam.remember(ordinary_body, source="preference", importance=0.95, scope="global")

        block = provider.prefetch(ordinary_body)

        assert ordinary_body in block
        assert all(body in block for body in bodies)
        if canonical_first:
            assert all(block.index(body) < block.index(ordinary_body) for body in bodies)
        else:
            assert all(block.index(ordinary_body) < block.index(body) for body in bodies)
    finally:
        provider.shutdown()


def test_prefetch_retains_ordinary_recovery_instructions_containing_canonical_fact(tmp_path):
    provider = MnemosyneMemoryProvider()
    provider.initialize("canonical-extension", hermes_home=str(tmp_path), agent_identity="default", profile_isolation=False)
    try:
        assert provider._beam is not None
        canonical = CanonicalStore(db_path=provider._beam.db_path, conn=provider._beam.conn)
        provider._beam.canonical = canonical
        seed(canonical, [("password", "Vaultwarden Passwort sichern")])
        ordinary_body = "Vaultwarden Passwort sichern recovery code offline export encrypted backup restore"
        assert provider._beam.remember(ordinary_body, source="preference", importance=0.95, scope="global")

        block = provider.prefetch("Vaultwarden Passwort sichern")

        assert "[CANONICAL]" in block
        assert ordinary_body in block
        assert block.index("[CANONICAL]") < block.index(ordinary_body)
    finally:
        provider.shutdown()


@pytest.mark.parametrize("canonical", [False, True])
def test_default_dedup_keeps_existing_containment_behavior(canonical):
    shorter = {"content": "Vaultwarden Passwort sichern", "trust_tier": "CANONICAL" if canonical else "STATED"}
    longer = {"content": "Vaultwarden Passwort sichern recovery code offline export encrypted backup restore"}

    assert _semantic_dedup_prefetch([shorter, longer]) == [shorter]
    if not canonical:
        assert _semantic_dedup_prefetch([shorter, longer], preserve_canonical_extensions=True) == [shorter]


@pytest.mark.parametrize(("coverage", "expected_prefix"), [("0.30", "common"), ("0.40", "rare")])
def test_canonical_priority_is_applied_before_candidate_limit(store, monkeypatch, coverage, expected_prefix):
    monkeypatch.setenv("MNEMOSYNE_PREFETCH_MIN_QUERY_COVERAGE", coverage)
    seed(store, [(f"rare{i}", f"Vaultwarden Passwort credential{i}") for i in range(3)])
    seed(store, [(f"common{i}", f"deployment pipeline worker workload{i}") for i in range(7)])
    query = "Vaultwarden Passwort deployment pipeline worker recovery archive backup export offline"
    all_rows = _canonical_prefetch_rows(store, "default", query, limit=10)
    by_name = {row["canonical_name"]: row for row in all_rows}
    assert by_name["rare0"]["score"] > by_name["common0"]["score"]

    selected = _canonical_prefetch_rows(store, "default", query)

    assert len(selected) == 3
    assert all(row["canonical_name"].startswith(expected_prefix) for row in selected)


def test_rendered_prefetch_keeps_qualifying_facts_before_candidate_limit(tmp_path):
    provider = MnemosyneMemoryProvider()
    provider.initialize("canonical-limit", hermes_home=str(tmp_path), agent_identity="default", profile_isolation=False)
    try:
        assert provider._beam is not None
        canonical = CanonicalStore(db_path=provider._beam.db_path, conn=provider._beam.conn)
        provider._beam.canonical = canonical
        seed(canonical, [(f"rare{i}", f"Vaultwarden Passwort credential{i}") for i in range(3)])
        seed(canonical, [(f"common{i}", f"deployment pipeline worker workload{i}") for i in range(7)])

        block = provider.prefetch("Vaultwarden Passwort deployment pipeline worker recovery archive backup export offline")

        assert "deployment pipeline worker" in block
        assert "[CANONICAL]" in block
    finally:
        provider.shutdown()


def test_generic_overlap_cannot_admit_a_weak_single_token_fact(store):
    seed(store, [("photos", "muss neue Fotos sichern")])

    assert names(store, "muss neue Vaultwarden Passwörter Backup Export sichern") == []


def test_rendered_prefetch_omits_single_token_fact_admitted_only_by_generic_overlap(tmp_path):
    provider = MnemosyneMemoryProvider()
    provider.initialize("canonical-singleton", hermes_home=str(tmp_path), agent_identity="default", profile_isolation=False)
    try:
        assert provider._beam is not None
        canonical = CanonicalStore(db_path=provider._beam.db_path, conn=provider._beam.conn)
        provider._beam.canonical = canonical
        seed(canonical, [("photos", "muss neue Fotos sichern")])

        block = provider.prefetch("muss neue Vaultwarden Passwörter Backup Export sichern")

        assert block == ""
    finally:
        provider.shutdown()
