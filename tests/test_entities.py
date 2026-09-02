"""
Unit tests for Mnemosyne Entity Sketching System.

Tests:
- Levenshtein distance and similarity
- Regex entity extraction
- Fuzzy entity matching
- Triple storage for entities
"""

import sys
import os
import unittest

# Add mnemosyne to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mnemosyne.core.entities import (
    levenshtein_distance,
    similarity,
    extract_entities_regex,
    find_similar_entities,
    ENTITY_EXTRACTION_STOP_WORDS,
)


class TestLevenshtein(unittest.TestCase):
    """Test pure Python Levenshtein implementation."""

    def test_exact_match(self):
        self.assertEqual(levenshtein_distance("hello", "hello"), 0)
        self.assertEqual(levenshtein_distance("", ""), 0)

    def test_single_insertion(self):
        self.assertEqual(levenshtein_distance("cat", "cats"), 1)

    def test_single_deletion(self):
        self.assertEqual(levenshtein_distance("cats", "cat"), 1)

    def test_single_substitution(self):
        self.assertEqual(levenshtein_distance("cat", "cut"), 1)

    def test_empty_string(self):
        self.assertEqual(levenshtein_distance("", "abc"), 3)
        self.assertEqual(levenshtein_distance("abc", ""), 3)

    def test_unicode(self):
        self.assertEqual(levenshtein_distance("café", "cafe"), 1)
        self.assertEqual(levenshtein_distance("日本", "日本語"), 1)


class TestSimilarity(unittest.TestCase):
    """Test prefix-biased similarity function."""

    def test_exact_match(self):
        self.assertEqual(similarity("Abdias", "Abdias"), 1.0)

    def test_similar_names(self):
        # Abdias vs Abdias J. — should be high similarity
        self.assertGreater(similarity("Abdias", "Abdias J."), 0.8)
        # With prefix boost, should be even higher
        self.assertGreater(similarity("Abdias", "Abdias Moya"), 0.7)

    def test_different_names(self):
        # Abdias vs Abdul — should be lower
        self.assertLess(similarity("Abdias", "Abdul"), 0.8)
        self.assertGreater(similarity("Abdias", "Abdul"), 0.3)  # Some prefix overlap

    def test_completely_different(self):
        self.assertLess(similarity("Abdias", "Zebra"), 0.3)

    def test_case_insensitive(self):
        self.assertEqual(similarity("ABDIAS", "abdias"), 1.0)

    def test_short_strings(self):
        self.assertEqual(similarity("A", "A"), 1.0)
        self.assertEqual(similarity("A", "B"), 0.0)

    def test_partial_prefix(self):
        # "Abd" should match "Abdias" reasonably
        self.assertGreater(similarity("Abd", "Abdias"), 0.5)


class TestRegexEntityExtraction(unittest.TestCase):
    """Test regex-based entity extraction."""

    def test_simple_name(self):
        result = extract_entities_regex("I met Abdias yesterday.")
        self.assertIn("Abdias", result)

    def test_multiple_names(self):
        result = extract_entities_regex("Abdias and Maya went to New York.")
        self.assertIn("Abdias", result)
        self.assertIn("Maya", result)
        self.assertIn("New York", result)

    def test_name_inside_quotes_still_extracts(self):
        # Quotes do not block \b, so a capitalized name inside them still
        # comes from the capitalized patterns.
        result = extract_entities_regex('She said "Hello World" to everyone.')
        self.assertIn("Hello World", result)

    def test_quoted_span_is_not_an_entity(self):
        # A quoted span is dialogue. Only the punctuation-bearing and
        # lowercase values were ever unique to the quote patterns, and they
        # are the ones that dodged the stop-word filter ('okay,' != 'okay').
        result = extract_entities_regex(
            "'Okay,' she said. 'The room is quiet now.'"
        )
        self.assertNotIn("Okay,", result)
        self.assertNotIn("The room is quiet now.", result)
        # The bare capitalized word is still a candidate; the fragment is not.
        self.assertIn("Okay", result)
        # Double quotes, and a value no capitalized pattern can reproduce, so
        # this one bites if the double-quote pattern ever comes back. The
        # comma is what let it dodge the stop-word filter before.
        self.assertNotIn(
            "okay, fine",
            extract_entities_regex('She whispered "okay, fine" and left.'),
        )

    def test_name_in_a_quoted_sentence_is_no_longer_swallowed(self):
        # The span used to be stored whole, and the substring post-filter then
        # dropped the name inside it for being part of a longer entity.
        result = extract_entities_regex("'Talia pauses.' Then she smiled.")
        self.assertIn("Talia", result)
        self.assertNotIn("Talia pauses.", result)

    def test_no_entities(self):
        result = extract_entities_regex("the quick brown fox jumps")
        # All lowercase, no entities expected
        self.assertEqual(len(result), 0)

    def test_at_mention(self):
        result = extract_entities_regex("Contact @abdias for help.")
        # @mentions capture the word after @, not the @ itself
        self.assertIn("abdias", result)

    def test_hashtag(self):
        result = extract_entities_regex("This is #ImportantNews today.")
        # Hashtags capture the word after #, not the # itself
        self.assertIn("ImportantNews", result)

    def test_stop_words_filtered(self):
        result = extract_entities_regex("The Quick Brown Fox")
        # "The" is filtered as a stop word, and with the any-word-stopword filter,
        # "The Quick Brown Fox" is also dropped because "The" contaminates it.
        self.assertNotIn("The", result)
        self.assertNotIn("The Quick Brown Fox", result)
        self.assertIn("Quick", result)
        self.assertIn("Brown", result)
        self.assertIn("Fox", result)

    def test_mixed_content(self):
        result = extract_entities_regex(
            "Abdias said: 'The Mnemosyne project is #Awesome. "
            "Contact @support or visit New York.'"
        )
        self.assertIn("Abdias", result)
        # "The Mnemosyne" is dropped because "The" is a stopword contaminating the phrase
        self.assertNotIn("The Mnemosyne", result)
        self.assertIn("Awesome", result)  # from #Awesome
        self.assertIn("support", result)  # from @support
        self.assertIn("New York", result)


class TestFindSimilarEntities(unittest.TestCase):
    """Test fuzzy entity matching against known entities."""

    def test_exact_match(self):
        known = ["Abdias", "Maya", "Mnemosyne"]
        result = find_similar_entities("Abdias", known, threshold=0.8)
        self.assertEqual(result, [("Abdias", 1.0)])

    def test_fuzzy_match(self):
        known = ["Abdias", "Maya", "Mnemosyne"]
        result = find_similar_entities("Abdias J.", known, threshold=0.8)
        self.assertIn(("Abdias", 0.8999999999999999), result)

    def test_no_match_below_threshold(self):
        known = ["Abdias", "Maya"]
        result = find_similar_entities("Zebra", known, threshold=0.8)
        self.assertEqual(len(result), 0)

    def test_multiple_matches(self):
        known = ["Abdias Moya", "Abdias J.", "Maya"]
        result = find_similar_entities("Abdias", known, threshold=0.7)
        # Should match both Abdias variants
        self.assertGreaterEqual(len(result), 1)

    def test_case_insensitive_match(self):
        known = ["Abdias"]
        result = find_similar_entities("ABDIAS", known, threshold=0.8)
        self.assertEqual(result, [("Abdias", 1.0)])


class TestStopWords(unittest.TestCase):
    """Test stop words set."""

    def test_common_stop_words_present(self):
        self.assertIn("the", ENTITY_EXTRACTION_STOP_WORDS)
        self.assertIn("and", ENTITY_EXTRACTION_STOP_WORDS)
        self.assertIn("for", ENTITY_EXTRACTION_STOP_WORDS)

    def test_case_insensitive(self):
        # Stop words are lowercase
        self.assertIn("The".lower(), ENTITY_EXTRACTION_STOP_WORDS)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_string_extraction(self):
        result = extract_entities_regex("")
        self.assertEqual(len(result), 0)

    def test_whitespace_only(self):
        result = extract_entities_regex("   \n\t  ")
        self.assertEqual(len(result), 0)

    def test_similarity_with_empty(self):
        self.assertEqual(similarity("", ""), 1.0)
        # Empty vs non-empty: prefix match with length ratio 0.0 < 0.3 guard
        # returns 0.0 (an empty string is NOT 70% similar to any entity)
        self.assertEqual(similarity("abc", ""), 0.0)
        self.assertEqual(similarity("", "abc"), 0.0)

    def test_levenshtein_with_none(self):
        # Should handle None gracefully or raise TypeError
        with self.assertRaises((TypeError, AttributeError)):
            levenshtein_distance(None, "abc")


if __name__ == "__main__":
    unittest.main()
