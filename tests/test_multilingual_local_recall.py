import pytest

from mnemosyne.core import embeddings
from mnemosyne.core import beam as beam_module
from mnemosyne.core.beam import (
    _expanded_query_tokens,
    _expand_hyphenated_tokens,
    _fts_query_terms,
    _hyphen_fragment_tokens,
    _leading_hyphen_fragments,
    _lexical_relevance,
    _literal_flag_bonus,
    _recall_tokens,
    _symbolic_code_tokens,
    BeamMemory,
)


def test_recall_tokens_preserve_unicode_words():
    tokens = _recall_tokens(
        "Stoßlüften im Bürgeramt: Primärquellen für den Mensa-Plan prüfen"
    )

    assert "stoßlüften" in tokens
    assert "bürgeramt" in tokens
    assert "primärquellen" in tokens
    assert "mensa-plan" in tokens
    assert "sto" not in tokens
    assert "ften" not in tokens
    assert "rgeramt" not in tokens


def test_hyphenated_query_terms_expand_for_candidate_recall_and_lexical_gate():
    query = "Welche Portnummer gehört zur Orion-Telemetrie?"
    fact = "Der Orion-Gateway nutzt Port 4831 für interne Telemetrie."
    tokens = _recall_tokens(query)

    expanded = _expanded_query_tokens(tokens)

    assert "orion-telemetrie" in expanded
    assert "orion" in expanded
    assert "telemetrie" in expanded
    score = _lexical_relevance(tokens, fact, query.lower())
    assert 0.3 <= score <= 1.0


def test_hyphen_component_score_stays_normalized_for_a_single_compound():
    score = _lexical_relevance(
        _recall_tokens("orion-telemetrie"),
        "The Orion Telemetrie gateway is healthy.",
        "orion-telemetrie",
    )
    assert score == 1.0


def test_three_component_compound_is_normalized_to_its_component_units():
    score = _lexical_relevance(
        _recall_tokens("orion-telemetrie-gateway"),
        "The Orion Telemetrie Gateway is healthy.",
        "orion-telemetrie-gateway",
    )
    assert score == 1.0


def test_partial_multi_component_overlap_yields_fractional_credit():
    score = _lexical_relevance(
        _recall_tokens("orion-telemetrie-gateway"),
        "The Orion Telemetrie service is healthy.",
        "orion-telemetrie-gateway",
    )
    assert score == 2 / 3


def test_exact_compound_scores_at_least_as_high_as_split_components():
    query = _recall_tokens("orion-gateway port")
    exact = _lexical_relevance(query, "The orion-gateway uses port 4831.", "orion-gateway port")
    split = _lexical_relevance(query, "The orion gateway uses port 4831.", "orion-gateway port")
    assert exact == split == 1.0


def test_non_hyphenated_scoring_is_unchanged():
    assert _lexical_relevance(
        _recall_tokens("atlas port"), "Atlas uses a port.", "atlas port"
    ) == 1.0


def test_hyphen_component_match_does_not_outweigh_unmatched_query_terms():
    score = _lexical_relevance(
        _recall_tokens("orion-telemetrie foobar"),
        "The Orion Telemetrie gateway is healthy.",
        "orion-telemetrie foobar",
    )
    assert score == 2 / 3


def test_single_hyphen_component_does_not_admit_a_generic_distractor():
    score = _lexical_relevance(
        _recall_tokens("orion-gateway"),
        "The gateway status page is healthy.",
        "orion-gateway",
    )
    assert score == 0.0


def test_duplicate_hyphen_components_do_not_count_as_distinct_matches():
    score = _lexical_relevance(
        _recall_tokens("orion-orion"),
        "The Orion status page is healthy.",
        "orion-orion",
    )
    assert score == 0.0


def test_hyphen_expansion_preserves_existing_non_hyphen_tokens():
    assert _expand_hyphenated_tokens(["4831", "ai", "orion-telemetrie"]) == [
        "4831", "ai", "orion-telemetrie", "orion", "telemetrie"
    ]
    assert _expand_hyphenated_tokens(["port-4831"]) == ["port-4831", "port"]
    assert _lexical_relevance(
        _recall_tokens("port-4831"), "The port is open.", "port-4831"
    ) == 0.0


def test_hyphen_expansion_filters_stopword_components_and_composes_synonyms():
    assert _expand_hyphenated_tokens(["orion-and"]) == ["orion-and", "orion"]

    expanded = _expanded_query_tokens(["branding-current"])
    assert "branding" in expanded
    assert "current" in expanded
    assert "positioning" in expanded
    assert "latest" in expanded


def test_two_hyphenated_compounds_share_their_total_lexical_unit_count():
    query = _recall_tokens("orion-telemetrie atlas-cache")
    score = _lexical_relevance(
        query,
        "Orion Telemetrie runs beside the Atlas cache.",
        "orion-telemetrie atlas-cache",
    )
    assert score == 1.0


@pytest.mark.parametrize(
    "content",
    [
        "The orion_telemetrie_api is healthy.",
        "The orion.telemetrie.api is healthy.",
        "The orion/telemetrie/api is healthy.",
    ],
)
def test_hyphenated_query_matches_structured_key_separators(content):
    query = _recall_tokens("orion-telemetrie")
    assert _lexical_relevance(query, content, "orion-telemetrie") == 1.0


def test_hyphenated_query_recalls_split_components_via_public_api(tmp_path):
    beam = BeamMemory(session_id="hyphenated-recall", db_path=tmp_path / "memory.db")
    expected_id = beam.remember(
        "Orion Gateway handles Telemetrie packets.", source="test", importance=0.5
    )
    distractor_id = beam.remember(
        "Gateway health is stable without Orion details.", source="test", importance=0.5
    )

    results = beam.recall("orion-telemetrie", top_k=5)

    assert results[0]["id"] == expected_id
    assert all(result["id"] != distractor_id for result in results)


def test_hyphen_fragment_tokens_extract_leading_hyphen_components():
    assert _hyphen_fragment_tokens("rm -rf") == ["rf"]
    assert _hyphen_fragment_tokens("install --force -rf") == ["force", "rf"]
    assert _hyphen_fragment_tokens("python -v") == ["v"]
    assert _hyphen_fragment_tokens("run -a") == []  # one-char stopword
    assert _hyphen_fragment_tokens("flag -1") == []  # numeric flag
    assert _hyphen_fragment_tokens("git--rebase") == []  # embedded, not a fragment
    assert _hyphen_fragment_tokens("git-rebase") == []
    assert _hyphen_fragment_tokens("") == []


def test_fts_query_terms_never_emit_hyphen_leading_terms():
    terms = _fts_query_terms("rm -rf")
    assert terms == ['"rf"']
    assert all(not term.startswith('"-') for term in terms)

    terms = _fts_query_terms("--force install")
    assert terms == ['"force"', '"install"']

    terms = _fts_query_terms("python -v")
    assert terms == ['"python"', '"v"']
    assert all(not term.startswith('"-') for term in terms)

    terms = _fts_query_terms("git-rebase")
    assert terms == ['"git-rebase"', '"git"', '"rebase"']
    assert all(not term.startswith('"-') for term in terms)


def test_fts_query_terms_never_emit_symbolic_code_terms():
    terms = _fts_query_terms("C++")
    assert terms == []  # unicode61 tokenizes C++ down to "c"; never emit "c" noise
    terms = _fts_query_terms("code in C++")
    assert terms == ['"code"']
    # A symbolic token with a suffix ("C++20") survives _recall_tokens() via the
    # `+` separator but must never reach the FTS MATCH builder: unicode61 splits
    # it into bare "c" + "20" tokens, flooding candidates. Exact lexical matching
    # alone handles it.
    terms = _fts_query_terms("C++20")
    assert terms == []
    terms = _fts_query_terms("code in C++20")
    assert terms == ['"code"']


def test_hyphen_fragments_score_lexically_without_admitting_distractors():
    query_lower = "rm -rf"
    assert _lexical_relevance([], "The user does not like rm -rf.", query_lower) == 1.0
    assert _lexical_relevance([], "Coffee before noon is fine.", query_lower) == 0.0


def test_leading_hyphen_fragments_recall_via_public_api(tmp_path):
    beam = BeamMemory(session_id="hyphen-fragments", db_path=tmp_path / "memory.db")
    expected_id = beam.remember(
        "The user does not like the use of `rm -rf`.", source="test", importance=0.5
    )
    distractor_id = beam.remember(
        "The user prefers git rebase over merge commits.", source="test", importance=0.5
    )

    results = beam.recall("rm -rf", top_k=5)

    assert results[0]["id"] == expected_id
    assert all(result["id"] != distractor_id for result in results)


def test_single_char_flag_recalls_via_public_api(tmp_path):
    beam = BeamMemory(session_id="single-char-flag", db_path=tmp_path / "memory.db")
    expected_id = beam.remember(
        "The user runs python with -v for verbose output.", source="test", importance=0.5
    )
    distractor_id = beam.remember(
        "The user prefers git rebase over merge commits.", source="test", importance=0.5
    )

    results = beam.recall("python -v", top_k=5)

    assert results[0]["id"] == expected_id
    assert all(result["id"] != distractor_id for result in results)


def test_symbolic_code_tokens_extract_symbolic_names_only():
    assert _symbolic_code_tokens("C++") == ["c++"]
    assert _symbolic_code_tokens("uses c# for dotnet") == ["c#"]
    assert _symbolic_code_tokens("g++ compiles") == ["g++"]
    assert _symbolic_code_tokens("F#") == ["f#"]
    assert _symbolic_code_tokens("C++20") == ["c++20"]
    assert _symbolic_code_tokens("a+b") == []  # arithmetic, not a code name
    assert _symbolic_code_tokens("git-rebase") == []  # hyphenated, not symbolic
    assert _symbolic_code_tokens("node_modules") == []
    assert _symbolic_code_tokens("python -v") == []
    assert _symbolic_code_tokens("") == []


def test_symbolic_code_queries_score_lexically_without_admitting_distractors():
    query_lower = "C++"
    assert _lexical_relevance([], "The user codes in C++.", query_lower) == 1.0
    assert _lexical_relevance([], "The user uses c# for dotnet.", query_lower) == 0.0
    assert _lexical_relevance([], "Coffee before noon is fine.", query_lower) == 0.0

    query_lower = "c#"
    assert _lexical_relevance([], "The user prefers c# for dotnet.", query_lower) == 1.0
    assert _lexical_relevance([], "The user codes in C++.", query_lower) == 0.0


def test_symbolic_code_queries_recall_via_public_api(tmp_path):
    beam = BeamMemory(session_id="symbolic-code", db_path=tmp_path / "memory.db")
    expected_id = beam.remember(
        "The user codes in C++ for performance-critical work.",
        source="test",
        importance=0.5,
    )
    distractor_id = beam.remember(
        "The user prefers git rebase over merge commits.", source="test", importance=0.5
    )

    results = beam.recall("C++", top_k=5)

    assert results[0]["id"] == expected_id
    assert all(result["id"] != distractor_id for result in results)


def test_symbolic_code_queries_recall_via_public_api_hash(tmp_path):
    beam = BeamMemory(session_id="symbolic-code-hash", db_path=tmp_path / "memory.db")
    expected_id = beam.remember(
        "The user prefers c# for dotnet services.", source="test", importance=0.5
    )
    distractor_id = beam.remember(
        "The user codes in C++.", source="test", importance=0.5
    )

    results = beam.recall("c#", top_k=5)

    assert results[0]["id"] == expected_id
    assert all(result["id"] != distractor_id for result in results)


def test_leading_hyphen_fragments_keep_the_literal_form():
    assert _leading_hyphen_fragments("--force") == ["--force"]
    assert _leading_hyphen_fragments("--dry-run") == ["--dry-run"]
    assert _leading_hyphen_fragments("rm -rf") == ["-rf"]
    assert _leading_hyphen_fragments("install --force -rf") == ["--force", "-rf"]
    assert _leading_hyphen_fragments("python -v") == ["-v"]
    assert _leading_hyphen_fragments("git--rebase") == []  # embedded, not a fragment
    assert _leading_hyphen_fragments("git-rebase") == []
    assert _leading_hyphen_fragments("") == []


def test_literal_flag_bonus_requires_an_exact_token_match():
    assert _literal_flag_bonus("--force", "Use --force now.") == 0.3
    assert _literal_flag_bonus("--force", "The --forceful approach failed.") == 0.0
    assert _literal_flag_bonus("--force", "The foo--force flag is not valid.") == 0.0
    assert _literal_flag_bonus("--force", "The force field is ready.") == 0.0
    assert _literal_flag_bonus("--force", "") == 0.0
    assert _literal_flag_bonus("deploy --force", "Deploy with --force after review.") == 0.3


def test_literal_flag_scores_above_a_bare_component_match():
    query_lower = "--force"
    assert _lexical_relevance([], "The deployment used --force to proceed.", query_lower) == 1.0
    assert _lexical_relevance([], "The force field calibration is complete.", query_lower) == 0.5

    query_lower = "deploy --force"
    query = _recall_tokens(query_lower)
    assert _lexical_relevance(query, "We deploy with --force after review.", query_lower) == 1.0
    assert _lexical_relevance(query, "We deploy using brute force.", query_lower) == 2 / 3
    assert _lexical_relevance(query, "The ship sails at noon.", query_lower) == 0.0


def test_literal_flag_recall_cannot_be_outranked_by_bare_component(tmp_path):
    beam = BeamMemory(session_id="literal-flag-precision", db_path=tmp_path / "memory.db")
    expected_id = beam.remember(
        "Use the deployment command with --force only after confirmation.",
        source="test",
        importance=0.1,
    )
    distractor_id = beam.remember(
        "The force field calibration is complete.", source="test", importance=1.0
    )

    results = beam.recall("--force", top_k=5)

    assert results[0]["id"] == expected_id
    assert all(result["id"] != distractor_id for result in results)


def test_literal_flag_recall_wins_in_a_mixed_query(tmp_path):
    beam = BeamMemory(session_id="literal-flag-mixed", db_path=tmp_path / "memory.db")
    expected_id = beam.remember(
        "We deploy with --force after review.", source="test", importance=0.2
    )
    distractor_id = beam.remember(
        "We deploy using brute force.", source="test", importance=1.0
    )

    results = beam.recall("deploy --force", top_k=5)

    assert results[0]["id"] == expected_id
    assert all(result["id"] != distractor_id for result in results)


def test_literal_flag_component_collision_requires_a_bare_component_only_match():
    collision = beam_module._is_bare_literal_flag_collision

    assert collision("--force", "The force field calibration is complete.")
    assert collision("deploy --force", "We deploy using brute force.")
    assert collision("--force", "The foo--force spelling is invalid.")
    assert collision("--dry-run", "Use --dry-clean for this command.")
    assert collision("--dry-run", "The --dry-running mode is experimental.")
    assert collision("--dry-run", "The foo--dry-run spelling is invalid.")
    assert not collision("--force", "Use --force after confirmation.")
    assert not collision("--dry-run", "Use --dry-run after confirmation.")
    assert not collision("--force", "The --forceful approach failed.")
    assert not collision("deploy", "We deploy using brute force.")


@pytest.mark.parametrize("tier", ["working", "episodic"])
def test_literal_flag_selection_uses_full_persisted_content(tmp_path, tier):
    beam = BeamMemory(session_id="literal-flag-full-content", db_path=tmp_path / "memory.db")
    exact_content = f"{'force calibration notes ' * 30}Use --force after confirmation."
    if tier == "working":
        expected_id = beam.remember(exact_content, source="test", importance=0.1)
        distractor_id = beam.remember(
            "The force field calibration is complete.", source="test", importance=1.0
        )
    else:
        expected_id = beam.consolidate_to_episodic(
            exact_content, source_wm_ids=[], source="test", importance=0.1
        )
        distractor_id = beam.consolidate_to_episodic(
            "The force field calibration is complete.",
            source_wm_ids=[],
            source="test",
            importance=1.0,
        )

    results = beam.recall(
        "--force",
        top_k=2,
        vec_weight=0.0,
        fts_weight=0.0,
        importance_weight=1.0,
    )

    assert results[0]["id"] == expected_id
    assert distractor_id not in {result["id"] for result in results}


@pytest.mark.parametrize("tier", ["working", "episodic"])
def test_multicomponent_literal_flag_rejects_a_different_flag_with_shared_prefix(
    tmp_path, tier
):
    beam = BeamMemory(session_id=f"literal-flag-multicomponent-{tier}", db_path=tmp_path / "memory.db")
    if tier == "working":
        expected_id = beam.remember(
            "Use --dry-run after confirmation.", source="test", importance=0.1
        )
        distractor_id = beam.remember(
            "Use --dry-clean for this command.", source="test", importance=1.0
        )
    else:
        expected_id = beam.consolidate_to_episodic(
            "Use --dry-run after confirmation.",
            source_wm_ids=[],
            source="test",
            importance=0.1,
        )
        distractor_id = beam.consolidate_to_episodic(
            "Use --dry-clean for this command.",
            source_wm_ids=[],
            source="test",
            importance=1.0,
        )

    results = beam.recall(
        "--dry-run",
        top_k=2,
        vec_weight=0.0,
        fts_weight=0.0,
        importance_weight=1.0,
    )

    assert results[0]["id"] == expected_id
    assert distractor_id not in {result["id"] for result in results}


def test_literal_flag_is_not_boosted_by_a_different_flag_prefix(tmp_path):
    beam = BeamMemory(session_id="literal-flag-boundary", db_path=tmp_path / "memory.db")
    expected_id = beam.remember(
        "Use --force only after confirmation.", source="test", importance=0.1
    )
    distractor_id = beam.remember(
        "The --forceful migration already ran.", source="test", importance=1.0
    )

    results = beam.recall("--force", top_k=5)

    assert results[0]["id"] == expected_id
    assert all(result["id"] != distractor_id for result in results[:1])


def test_sentence_transformers_multilingual_dimensions_are_known():
    assert embeddings._get_embedding_dim(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ) == 384
    assert embeddings._get_embedding_dim(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ) == 768
