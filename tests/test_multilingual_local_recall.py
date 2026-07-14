import pytest

from mnemosyne.core import embeddings
from mnemosyne.core.beam import (
    _expanded_query_tokens,
    _expand_hyphenated_tokens,
    _lexical_relevance,
    _recall_tokens,
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


def test_sentence_transformers_multilingual_dimensions_are_known():
    assert embeddings._get_embedding_dim(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ) == 384
    assert embeddings._get_embedding_dim(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ) == 768
